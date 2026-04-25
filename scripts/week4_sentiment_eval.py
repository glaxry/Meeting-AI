from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meeting_ai.config import get_settings
from meeting_ai.evaluation import (
    accuracy_score,
    bootstrap_confidence_interval,
    confusion_matrix,
    load_jsonl,
    macro_f1_score,
    precision_recall_f1,
    value_counts,
)
from meeting_ai.schemas import LLMProvider, TranscriptResult, TranscriptSegment
from meeting_ai.sentiment_agent import SentimentAgent


LABEL_SPACE = ["agreement", "disagreement", "hesitation", "tension", "neutral"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Week 4 sentiment evaluation.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "data" / "eval" / "sentiment_labels.benchmark_v2.jsonl"),
        help="JSONL manifest with text and gold label.",
    )
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in LLMProvider],
        default=LLMProvider.DEEPSEEK.value,
        help="Provider used for the LLM sentiment route.",
    )
    parser.add_argument(
        "--include-qwen",
        action="store_true",
        help="Also run the Qwen LLM route when a Qwen key is configured.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "week4" / "sentiment_eval.json"),
        help="Path to the JSON output file.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=2000,
        help="Bootstrap iterations for confidence intervals.",
    )
    return parser


def _build_transcript(rows: list[dict[str, object]]) -> TranscriptResult:
    segments = [
        TranscriptSegment(
            speaker=str(row.get("speaker", "SPEAKER_00")),
            text=str(row["text"]),
            start=float(index),
            end=float(index + 1),
        )
        for index, row in enumerate(rows)
    ]
    full_text = "\n".join(f"[{segment.speaker}] {segment.text}" for segment in segments)
    return TranscriptResult(
        audio_path="sentiment_eval.jsonl",
        language="mixed",
        asr_model="manual",
        diarization_backend="manual",
        segments=segments,
        full_text=full_text,
        metadata={"source": "week4_sentiment_eval"},
    )


def _manifest_stats(rows: list[dict[str, Any]]) -> dict[str, object]:
    return {
        "sample_count": len(rows),
        "label_distribution": value_counts([str(row["label"]) for row in rows]),
        "language_distribution": value_counts([str(row.get("language", "unknown")) for row in rows]),
        "difficulty_distribution": value_counts([str(row.get("difficulty", "unspecified")) for row in rows]),
    }


def _dataset_warnings(stats: dict[str, object]) -> list[str]:
    warnings: list[str] = []
    sample_count = int(stats.get("sample_count", 0) or 0)
    language_distribution = stats.get("language_distribution", {})

    if sample_count < 100:
        warnings.append("Benchmark is still small; treat sentiment metrics as preliminary and inspect confidence intervals.")
    if isinstance(language_distribution, dict) and len(language_distribution) == 1:
        warnings.append("Benchmark is single-language only; it should not be used to claim multilingual robustness.")
    return warnings


def _route_warnings(route_name: str, accuracy: float, macro_f1: float) -> list[str]:
    warnings: list[str] = []
    if accuracy >= 0.999999 and macro_f1 >= 0.999999:
        warnings.append(
            f"{route_name} saturated the current benchmark; treat this result as a ceiling-effect signal instead of a production-grade claim."
        )
    return warnings


def _evaluate_route(
    route_name: str,
    result_labels: list[str],
    gold_labels: list[str],
    latency_seconds: float | None,
    *,
    bootstrap_iterations: int,
) -> dict[str, object]:
    accuracy = accuracy_score(gold_labels, result_labels)
    macro_f1 = macro_f1_score(gold_labels, result_labels, LABEL_SPACE)
    return {
        "route": route_name,
        "sample_count": len(gold_labels),
        "label_distribution": value_counts(gold_labels),
        "accuracy": accuracy,
        "accuracy_ci": bootstrap_confidence_interval(
            gold_labels,
            result_labels,
            accuracy_score,
            iterations=bootstrap_iterations,
        ),
        "macro_f1": macro_f1,
        "macro_f1_ci": bootstrap_confidence_interval(
            gold_labels,
            result_labels,
            lambda sample_labels, sample_predictions: macro_f1_score(sample_labels, sample_predictions, LABEL_SPACE),
            iterations=bootstrap_iterations,
        ),
        "per_label": precision_recall_f1(gold_labels, result_labels, LABEL_SPACE),
        "confusion_matrix": confusion_matrix(gold_labels, result_labels, LABEL_SPACE),
        "latency_seconds": round(latency_seconds, 3) if latency_seconds is not None else None,
        "warnings": _route_warnings(route_name, accuracy, macro_f1),
    }


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    rows = load_jsonl(args.manifest)
    transcript = _build_transcript(rows)
    gold_labels = [str(row["label"]) for row in rows]

    outputs: dict[str, dict[str, object]] = {}

    transformer_agent = SentimentAgent(settings=settings)
    started = time.perf_counter()
    transformer_result = transformer_agent.analyze(route="transformer", transcript=transcript)
    transformer_latency = transformer_result.metadata.get("latency_seconds") or (time.perf_counter() - started)
    outputs["transformer"] = _evaluate_route(
        route_name="transformer",
        result_labels=[segment.sentiment.value for segment in transformer_result.segments],
        gold_labels=gold_labels,
        latency_seconds=float(transformer_latency),
        bootstrap_iterations=args.bootstrap_iterations,
    )

    llm_provider = LLMProvider(args.provider)
    llm_agent = SentimentAgent(settings=settings, provider=llm_provider)
    llm_result = llm_agent.analyze(route="llm", transcript=transcript)
    outputs[f"llm_{llm_provider.value}"] = _evaluate_route(
        route_name=f"llm_{llm_provider.value}",
        result_labels=[segment.sentiment.value for segment in llm_result.segments],
        gold_labels=gold_labels,
        latency_seconds=float(llm_result.metadata.get("latency_seconds") or 0.0),
        bootstrap_iterations=args.bootstrap_iterations,
    )

    if args.include_qwen and settings.resolved_qwen_api_key:
        qwen_agent = SentimentAgent(settings=settings, provider=LLMProvider.QWEN)
        qwen_result = qwen_agent.analyze(route="llm", transcript=transcript)
        outputs["llm_qwen"] = _evaluate_route(
            route_name="llm_qwen",
            result_labels=[segment.sentiment.value for segment in qwen_result.segments],
            gold_labels=gold_labels,
            latency_seconds=float(qwen_result.metadata.get("latency_seconds") or 0.0),
            bootstrap_iterations=args.bootstrap_iterations,
        )

    dataset = _manifest_stats(rows)
    output = {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "label_space": LABEL_SPACE,
        "dataset": dataset,
        "warnings": _dataset_warnings(dataset),
        "routes": outputs,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
