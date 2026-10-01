"""Connect saved lab artifacts to the reusable evaluation core.

This file deliberately contains only artifact I/O and Exercise 3.2 reporting.
All evaluation metrics, benchmark execution, and failure analysis remain in
``template.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Any

from template import (
    BenchmarkRunner,
    EVALUATION_VERSION,
    EvalResult,
    FailureAnalyzer,
    PASS_THRESHOLD,
    QAPair,
    RAGASEvaluator,
    REGRESSION_DROP_THRESHOLD,
)


def _read_json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{label} is not valid JSON: {path}:{exc.lineno}:{exc.colno} "
            f"({exc.msg})"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return value


def load_evaluation_inputs(
    golden_path: str | Path,
    actual_path: str | Path,
) -> tuple[list[QAPair], dict[str, str]]:
    """Join golden records and saved actual answers by ID."""

    golden = _read_json_file(Path(golden_path).resolve(), "Golden dataset")
    actual = _read_json_file(Path(actual_path).resolve(), "Actual answers")
    if golden.get("corpus_id") != actual.get("corpus_id"):
        raise ValueError("Golden dataset and actual answers use different corpus_id")

    golden_records = golden.get("qa_pairs")
    actual_records = actual.get("answers")
    if not isinstance(golden_records, list) or not golden_records:
        raise ValueError("Golden dataset qa_pairs must be a non-empty list")
    if not isinstance(actual_records, list) or not actual_records:
        raise ValueError("Actual answers must contain a non-empty answers list")

    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(actual_records):
        if not isinstance(record, dict):
            raise ValueError(f"Actual answers[{index}] must be an object")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"Actual answers[{index}].id must be a string")
        if record_id in actual_by_id:
            raise ValueError(f"Duplicate actual-answer ID: {record_id}")
        actual_by_id[record_id] = record

    qa_pairs: list[QAPair] = []
    answers_by_question: dict[str, str] = {}
    golden_ids: set[str] = set()
    for index, record in enumerate(golden_records):
        if not isinstance(record, dict):
            raise ValueError(f"Golden qa_pairs[{index}] must be an object")
        record_id = record.get("id")
        question = record.get("question")
        expected = record.get("expected_answer")
        contexts = record.get("contexts")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"Golden qa_pairs[{index}].id must be a string")
        if record_id in golden_ids:
            raise ValueError(f"Duplicate golden ID: {record_id}")
        golden_ids.add(record_id)
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{record_id}: question must be a non-empty string")
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(f"{record_id}: expected_answer must be non-empty")
        if not isinstance(contexts, list) or not contexts:
            raise ValueError(f"{record_id}: contexts must be a non-empty list")

        actual_record = actual_by_id.get(record_id)
        if actual_record is None:
            raise ValueError(f"Missing actual answer for {record_id}")
        if actual_record.get("question") != question:
            raise ValueError(f"{record_id}: question differs between artifacts")
        answer = actual_record.get("actual_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{record_id}: actual_answer must be non-empty")
        if actual_record.get("error") is not None:
            raise ValueError(f"{record_id}: inference contains an error")

        gold_context_texts: list[str] = []
        for context_index, context_record in enumerate(contexts):
            if not isinstance(context_record, dict):
                raise ValueError(
                    f"{record_id}: contexts[{context_index}] must be an object"
                )
            text = context_record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{record_id}: contexts[{context_index}].text must be non-empty"
                )
            gold_context_texts.append(text.strip())

        retrieved_records = actual_record.get("retrieved_contexts")
        if not isinstance(retrieved_records, list):
            raise ValueError(f"{record_id}: retrieved_contexts must be a list")
        retrieved_texts: list[str] = []
        for context_index, context_record in enumerate(retrieved_records):
            if not isinstance(context_record, dict):
                raise ValueError(
                    f"{record_id}: retrieved_contexts[{context_index}] must be an object"
                )
            text = context_record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{record_id}: retrieved_contexts[{context_index}].text "
                    "must be non-empty"
                )
            retrieved_texts.append(text.strip())

        if question in answers_by_question:
            raise ValueError(f"Duplicate question in golden dataset: {question}")
        answers_by_question[question] = answer.strip()
        qa_pairs.append(
            QAPair(
                question=question,
                expected_answer=expected.strip(),
                context="\n\n".join(gold_context_texts),
                metadata={
                    "id": record_id,
                    "difficulty": record.get("difficulty"),
                    "attack_type": record.get("attack_type"),
                },
                retrieved_contexts=retrieved_texts,
            )
        )

    unexpected_ids = sorted(set(actual_by_id) - golden_ids)
    if unexpected_ids:
        raise ValueError(
            "Actual answers contain unknown IDs: " + ", ".join(unexpected_ids)
        )
    return qa_pairs, answers_by_question


def build_evaluation_artifact(
    results: list[EvalResult],
    summary: dict[str, Any],
    analyzer: FailureAnalyzer,
) -> dict[str, Any]:
    failures = [result for result in results if not result.passed]
    suggestions = analyzer.generate_improvement_suggestions(failures)
    return {
        "summary": summary,
        "results": [
            {
                "id": result.qa_pair.metadata.get("id"),
                "difficulty": result.qa_pair.metadata.get("difficulty"),
                "question": result.qa_pair.question,
                "actual_answer": result.actual_answer,
                "faithfulness": result.faithfulness,
                "relevance": result.relevance,
                "completeness": result.completeness,
                "context_recall": result.context_recall,
                "context_precision": result.context_precision,
                "overall": result.overall_score(),
                "passed": result.passed,
                "failure_type": result.failure_type,
            }
            for result in results
        ],
        "failure_analysis": {
            "counts": analyzer.categorize_failures(failures),
            "suggestions": suggestions,
            "improvement_log": analyzer.generate_improvement_log(
                failures, suggestions
            ),
        },
    }


def _format_optional_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_exercise_3_2(results: list[EvalResult], summary: dict[str, Any]) -> None:
    """Print the table and summary requested by Exercise 3.2."""

    print(
        "| ID | Question (short) | Context Recall | Context Precision | "
        "Faithfulness | Relevance | Completeness | Overall | Passed? | Failure Type |"
    )
    print(
        "|----|------------------|----------------|-------------------|"
        "--------------|-----------|--------------|---------|---------|--------------|"
    )
    for result in results:
        pair = result.qa_pair
        question = re.sub(r"\s+", " ", pair.question).replace("|", "\\|")
        if len(question) > 48:
            question = f"{question[:45]}..."
        print(
            f"| {pair.metadata.get('id', '')} | {question} | "
            f"{_format_optional_score(result.context_recall)} | "
            f"{_format_optional_score(result.context_precision)} | "
            f"{result.faithfulness:.3f} | {result.relevance:.3f} | "
            f"{result.completeness:.3f} | {result.overall_score():.3f} | "
            f"{'Yes' if result.passed else 'No'} | "
            f"{result.failure_type or '-'} |"
        )

    print("\nAggregate Report:")
    print(f"- Overall pass rate: {summary['pass_rate']:.1%}")
    print(
        "- Avg Context Recall: "
        f"{_format_optional_score(summary['avg_context_recall'])}"
    )
    print(
        "- Avg Context Precision: "
        f"{_format_optional_score(summary['avg_context_precision'])}"
    )
    print(f"- Avg Faithfulness: {summary['avg_faithfulness']:.3f}")
    print(f"- Avg Relevance: {summary['avg_relevance']:.3f}")
    print(f"- Avg Completeness: {summary['avg_completeness']:.3f}")
    print(f"- Failure type distribution: {summary['failure_types']}")

    worst = sorted(results, key=lambda result: result.overall_score())[:3]
    print("\n3 lowest-scoring cases:")
    for index, result in enumerate(worst, start=1):
        print(
            f"{index}. ID: {result.qa_pair.metadata.get('id', '')} | "
            f"Score: {result.overall_score():.3f} | "
            f"Failure type: {result.failure_type or '-'}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved domain-assistant answers using the core in template.py."
        )
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("golden_dataset.json"),
        help="Golden dataset (default: golden_dataset.json)",
    )
    parser.add_argument(
        "--actual",
        type=Path,
        default=Path("artifacts/actual_answers.json"),
        help="Saved RAG answers (default: artifacts/actual_answers.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmark_results.json"),
        help="Benchmark output (default: artifacts/benchmark_results.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        qa_pairs, answers_by_question = load_evaluation_inputs(
            args.golden,
            args.actual,
        )

        def recorded_agent(question: str) -> str:
            return answers_by_question[question]

        evaluator = RAGASEvaluator()
        runner = BenchmarkRunner()
        results = runner.run(qa_pairs, recorded_agent, evaluator)
        summary = runner.generate_report(results)
        artifact = build_evaluation_artifact(
            results,
            summary,
            FailureAnalyzer(),
        )
        golden_digest = hashlib.sha256(args.golden.resolve().read_bytes()).hexdigest()
        actual_digest = hashlib.sha256(args.actual.resolve().read_bytes()).hexdigest()
        core_digest = hashlib.sha256(
            Path(__file__).with_name("template.py").read_bytes()
        ).hexdigest()
        adapter_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        artifact["run_metadata"] = {
            "evaluation_version": EVALUATION_VERSION,
            "overall_score": "arithmetic mean of faithfulness, relevance, completeness",
            "pass_threshold": PASS_THRESHOLD,
            "regression_drop_threshold": REGRESSION_DROP_THRESHOLD,
            "golden_dataset_sha256": golden_digest,
            "actual_answers_sha256": actual_digest,
            "evaluation_core_sha256": core_digest,
            "adapter_sha256": adapter_digest,
            "python_version": platform.python_version(),
            "replay_scope": (
                "Offline scoring is reproducible for these exact input and code "
                "hashes; regenerating answers through a hosted model may vary."
            ),
            "run_id": hashlib.sha256(
                ":".join(
                    (
                        EVALUATION_VERSION,
                        golden_digest,
                        actual_digest,
                        core_digest,
                        adapter_digest,
                    )
                ).encode("utf-8")
            ).hexdigest()[:16],
        }
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except NotImplementedError as exc:
        print(f"ERROR: Complete the required TODOs in template.py first: {exc}")
        return 2
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    print_exercise_3_2(results, summary)
    print(f"\nSaved benchmark results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
