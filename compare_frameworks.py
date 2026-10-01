"""Run the Exercise 3.4 RAGAS vs. DeepEval comparison on saved artifacts.

The system answers and ranked retrieval traces are loaded from disk, so this
script only calls the evaluator models. It never calls the OrbitTech answer
generator and never prints API keys.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI


ROOT = Path(__file__).resolve().parent
METRIC_NAMES = ("faithfulness", "context_precision", "context_recall")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(golden_path: Path, actual_path: Path) -> list[dict[str, Any]]:
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    actual = json.loads(actual_path.read_text(encoding="utf-8"))
    gold_by_id = {row["id"]: row for row in golden["qa_pairs"]}
    actual_by_id = {row["id"]: row for row in actual["answers"]}
    if set(gold_by_id) != set(actual_by_id):
        raise ValueError("Golden and actual answer IDs do not match.")

    rows: list[dict[str, Any]] = []
    for qa_id in sorted(gold_by_id):
        gold = gold_by_id[qa_id]
        answer = actual_by_id[qa_id]
        if answer.get("error"):
            raise ValueError(f"Actual answer {qa_id} contains a generation error.")
        if answer["question"] != gold["question"]:
            raise ValueError(f"Question mismatch for {qa_id}; regenerate actual answers.")
        retrieved = answer.get("retrieved_contexts") or []
        rows.append(
            {
                "id": qa_id,
                "question": gold["question"],
                "reference": gold["expected_answer"],
                "answer": answer["actual_answer"],
                "retrieved_contexts": [item["text"] for item in retrieved],
            }
        )
    return rows


async def score_ragas(metric: Any, sample: Any) -> float:
    value = await metric.single_turn_ascore(sample)
    return float(value)


def make_metadata(
    golden_path: Path, actual_path: Path, judge_model: str
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "judge_model": judge_model,
        "judge_temperature": 0,
        "ragas_version": importlib.metadata.version("ragas"),
        "deepeval_version": importlib.metadata.version("deepeval"),
        "langchain_community_version": importlib.metadata.version(
            "langchain-community"
        ),
        "golden_dataset_sha256": sha256(golden_path),
        "actual_answers_sha256": sha256(actual_path),
        "frameworks_script_sha256": sha256(Path(__file__)),
        "metric_mapping": {
            "faithfulness": ["Ragas.Faithfulness", "DeepEval.FaithfulnessMetric"],
            "context_precision": [
                "Ragas.LLMContextPrecisionWithReference",
                "DeepEval.ContextualPrecisionMetric",
            ],
            "context_recall": [
                "Ragas.LLMContextRecall",
                "DeepEval.ContextualRecallMetric",
            ],
        },
        "scoring_note": (
            "Same question, reference answer, generated answer, and ranked "
            "retrieved chunks were supplied to both frameworks. Metric prompts "
            "and scoring implementations differ, so raw values are diagnostic "
            "and should not be treated as interchangeable scales."
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for framework in ("ragas", "deepeval"):
        summary[framework] = {}
        for metric in METRIC_NAMES:
            values = [float(row[framework][metric]) for row in rows]
            summary[framework][metric] = {
                "average": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
                "below_0_5_ids": [
                    row["id"] for row in rows if row[framework][metric] < 0.5
                ],
            }
    summary["mean_score_delta_ragas_minus_deepeval"] = {
        metric: summary["ragas"][metric]["average"]
        - summary["deepeval"][metric]["average"]
        for metric in METRIC_NAMES
    }
    return summary


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


async def run_comparison(
    golden_path: Path,
    actual_path: Path,
    output_path: Path,
    force: bool,
) -> dict[str, Any]:
    golden_path = golden_path.resolve()
    actual_path = actual_path.resolve()
    output_path = output_path.resolve()
    partial_path = output_path.with_name(f"{output_path.stem}.partial.json")

    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing; set it in the ignored .env file.")
    judge_model = os.environ.get("OPENAI_EVAL_MODEL", "gpt-4o-mini")
    # The framework's anonymous analytics are unrelated to this comparison.
    # Disable them before importing DeepEval so only judge requests go to OpenAI.
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"

    try:
        import deepeval
        import ragas
        from deepeval.metrics import (
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
        from deepeval.models import OpenAIModel
        from deepeval.test_case import LLMTestCase
        from ragas.dataset_schema import SingleTurnSample
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional, pinned packages with "
            "`python -m pip install -r requirements-frameworks.txt`."
        ) from exc

    metadata = make_metadata(golden_path, actual_path, judge_model)
    if output_path.exists() and not force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if all(
            existing.get("run_metadata", {}).get(key) == value
            for key, value in metadata.items()
            if key != "created_at"
        ):
            print(f"Reusing completed comparison: {output_path}")
            return existing
        raise RuntimeError(
            f"{output_path} exists with different inputs or configuration; "
            "use --force to run and replace it."
        )

    rows = load_inputs(golden_path, actual_path)
    completed: dict[str, dict[str, Any]] = {}
    if partial_path.exists() and not force:
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        old_metadata = partial.get("run_metadata", {})
        if all(
            old_metadata.get(key) == value
            for key, value in metadata.items()
            if key not in {"created_at", "frameworks_script_sha256"}
        ) and old_metadata.get("frameworks_script_sha256") == metadata[
            "frameworks_script_sha256"
        ]:
            completed = {row["id"]: row for row in partial.get("results", [])}

    ragas_chat = ChatOpenAI(model=judge_model, temperature=0, seed=42)
    ragas_llm = LangchainLLMWrapper(ragas_chat)
    ragas_metrics = {
        "faithfulness": Faithfulness(llm=ragas_llm),
        "context_precision": LLMContextPrecisionWithReference(llm=ragas_llm),
        "context_recall": LLMContextRecall(llm=ragas_llm),
    }
    deepeval_model = OpenAIModel(
        model=judge_model,
        temperature=0,
        generation_kwargs={"seed": 42},
    )
    deepeval_metrics = {
        "faithfulness": FaithfulnessMetric(
            threshold=None, model=deepeval_model, include_reason=False, async_mode=False
        ),
        "context_precision": ContextualPrecisionMetric(
            threshold=None, model=deepeval_model, include_reason=False, async_mode=False
        ),
        "context_recall": ContextualRecallMetric(
            threshold=None, model=deepeval_model, include_reason=False, async_mode=False
        ),
    }

    for index, item in enumerate(rows, start=1):
        qa_id = item["id"]
        if qa_id in completed:
            print(f"[{index}/{len(rows)}] {qa_id}: resumed saved scores", flush=True)
            continue

        ragas_sample = SingleTurnSample(
            user_input=item["question"],
            response=item["answer"],
            reference=item["reference"],
            retrieved_contexts=item["retrieved_contexts"],
        )
        deep_eval_case = LLMTestCase(
            input=item["question"],
            actual_output=item["answer"],
            expected_output=item["reference"],
            retrieval_context=item["retrieved_contexts"],
        )

        ragas_scores: dict[str, float] = {}
        deepeval_scores: dict[str, float] = {}
        for metric_name, metric in ragas_metrics.items():
            ragas_scores[metric_name] = await score_ragas(metric, ragas_sample)
        for metric_name, metric in deepeval_metrics.items():
            metric.measure(deep_eval_case)
            if metric.score is None:
                raise RuntimeError(f"DeepEval returned no {metric_name} score for {qa_id}.")
            deepeval_scores[metric_name] = float(metric.score)

        completed[qa_id] = {
            "id": qa_id,
            "ragas": ragas_scores,
            "deepeval": deepeval_scores,
        }
        partial_artifact = {
            "run_metadata": metadata,
            "completed_cases": len(completed),
            "total_cases": len(rows),
            "results": [completed[row["id"]] for row in rows if row["id"] in completed],
        }
        save_json(partial_path, partial_artifact)
        print(f"[{index}/{len(rows)}] {qa_id}: both frameworks scored", flush=True)

    results = [completed[row["id"]] for row in rows]
    artifact = {
        "summary": summarize(results),
        "results": results,
        "run_metadata": metadata,
    }
    save_json(output_path, artifact)
    partial_path.unlink(missing_ok=True)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden", type=Path, default=ROOT / "golden_dataset.json"
    )
    parser.add_argument(
        "--actual", type=Path, default=ROOT / "artifacts/actual_answers.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/framework_comparison.json"
    )
    parser.add_argument(
        "--force", action="store_true", help="Run again instead of reuse/resume."
    )
    args = parser.parse_args()
    artifact = asyncio.run(
        run_comparison(args.golden, args.actual, args.output, args.force)
    )
    print(f"Compared {len(artifact['results'])} cases.")
    for framework in ("ragas", "deepeval"):
        print(framework)
        for metric in METRIC_NAMES:
            stats = artifact["summary"][framework][metric]
            print(
                f"  {metric}: avg={stats['average']:.3f}, "
                f"min={stats['min']:.3f}, max={stats['max']:.3f}, "
                f"below_0.5={len(stats['below_0_5_ids'])}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
