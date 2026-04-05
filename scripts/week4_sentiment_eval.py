from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meeting_ai.config import get_settings
from meeting_ai.evaluation import accuracy_score, confusion_matrix, load_jsonl, macro_f1_score, precision_recall_f1
from meeting_ai.schemas import LLMProvider, TranscriptResult, TranscriptSegment
from meeting_ai.sentiment_agent import SentimentAgent


LABEL_SPACE = ["agreement", "disagreement", "hesitation", "tension", "neutral"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Week 4 sentiment evaluation.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "data" / "eval" / "sentiment_labels.sample.jsonl"),
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


def _evaluate_route(route_name: str, result_labels: list[str], gold_labels: list[str], latency_seconds: float | None) -> dict[str, object]:
    return {
        "route": route_name,
        "sample_count": len(gold_labels),
        "accuracy": accuracy_score(gold_labels, result_labels),
        "macro_f1": macro_f1_score(gold_labels, result_labels, LABEL_SPACE),
        "per_label": precision_recall_f1(gold_labels, result_labels, LABEL_SPACE),
        "confusion_matrix": confusion_matrix(gold_labels, result_labels, LABEL_SPACE),
        "latency_seconds": round(latency_seconds, 3) if latency_seconds is not None else None,
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
    )

    llm_provider = LLMProvider(args.provider)
    llm_agent = SentimentAgent(settings=settings, provider=llm_provider)
    llm_result = llm_agent.analyze(route="llm", transcript=transcript)
    outputs[f"llm_{llm_provider.value}"] = _evaluate_route(
        route_name=f"llm_{llm_provider.value}",
        result_labels=[segment.sentiment.value for segment in llm_result.segments],
        gold_labels=gold_labels,
        latency_seconds=float(llm_result.metadata.get("latency_seconds") or 0.0),
    )

    if args.include_qwen and settings.resolved_qwen_api_key:
        qwen_agent = SentimentAgent(settings=settings, provider=LLMProvider.QWEN)
        qwen_result = qwen_agent.analyze(route="llm", transcript=transcript)
        outputs["llm_qwen"] = _evaluate_route(
            route_name="llm_qwen",
            result_labels=[segment.sentiment.value for segment in qwen_result.segments],
            gold_labels=gold_labels,
            latency_seconds=float(qwen_result.metadata.get("latency_seconds") or 0.0),
        )

    output = {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "label_space": LABEL_SPACE,
        "routes": outputs,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
