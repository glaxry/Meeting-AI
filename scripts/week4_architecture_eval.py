from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meeting_ai.action_item_agent import ActionItemAgent
from meeting_ai.baseline import SerialMeetingPipeline
from meeting_ai.config import get_settings
from meeting_ai.orchestrator import MeetingOrchestrator
from meeting_ai.reporting import load_workflow_result
from meeting_ai.schemas import (
    ActionItemResult,
    LLMProvider,
    MeetingWorkflowResult,
    SentimentLabel,
    SentimentResult,
    SentimentSegment,
    SummaryResult,
    TranscriptResult,
    TranscriptSegment,
    TranslationResult,
)
from meeting_ai.sentiment_agent import SentimentAgent
from meeting_ai.summary_agent import SummaryAgent
from meeting_ai.text_utils import load_transcript_json, parse_labelled_lines
from meeting_ai.translation_agent import TranslationAgent
from meeting_ai.asr_agent import MeetingASRAgent


class StaticASR:
    def __init__(self, transcript: TranscriptResult):
        self.transcript = transcript

    def transcribe(self, **kwargs) -> TranscriptResult:
        return self.transcript


class FakeSummaryAgent:
    def __init__(self, provider: LLMProvider = LLMProvider.DEEPSEEK):
        self.provider = provider

    def summarize(self, **kwargs) -> SummaryResult:
        return SummaryResult(
            topics=["Architecture comparison"],
            decisions=["Keep parallel execution for independent agents"],
            follow_ups=["Document failure isolation behavior"],
            metadata={"provider": self.provider.value},
        )


class FakeTranslationAgent:
    def __init__(self, provider: LLMProvider = LLMProvider.DEEPSEEK):
        self.provider = provider

    def translate(self, transcript, **kwargs) -> TranslationResult:
        return TranslationResult(
            source_language="en",
            target_language="zh",
            segments=transcript.segments,
            full_text=transcript.full_text,
            metadata={"provider": self.provider.value},
        )


class BrokenTranslationAgent(FakeTranslationAgent):
    def translate(self, transcript, **kwargs) -> TranslationResult:
        raise RuntimeError("Injected translation failure for architecture evaluation.")


class FakeActionItemAgent:
    def __init__(self, provider: LLMProvider = LLMProvider.DEEPSEEK):
        self.provider = provider

    def extract(self, **kwargs) -> ActionItemResult:
        return ActionItemResult(metadata={"provider": self.provider.value})


class FakeSentimentAgent:
    def __init__(self, provider: LLMProvider = LLMProvider.DEEPSEEK):
        self.provider = provider

    def analyze(self, **kwargs) -> SentimentResult:
        return SentimentResult(
            route="llm",
            overall_tone=SentimentLabel.NEUTRAL,
            segments=[SentimentSegment(text="ok", sentiment=SentimentLabel.NEUTRAL, confidence=0.5)],
            metadata={"provider": self.provider.value},
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Week 4 architecture evaluation.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--audio", help="Audio file used to build the shared transcript for the runtime comparison.")
    source_group.add_argument("--transcript-json", help="Existing transcript JSON file used for the runtime comparison.")
    source_group.add_argument("--workflow-json", help="Existing workflow JSON file; its transcript section is reused.")
    parser.add_argument("--language", default="zh")
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in LLMProvider],
        default=LLMProvider.DEEPSEEK.value,
    )
    parser.add_argument("--target-language", default="en")
    parser.add_argument("--sentiment-route", choices=["llm", "transformer"], default="llm")
    parser.add_argument("--glossary", action="append", help="Glossary item in SOURCE=TARGET format.")
    parser.add_argument("--max-segments", type=int, help="Optional limit applied to the shared transcript before runtime comparison.")
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "week4" / "architecture_eval.json"),
        help="Path to the JSON output file.",
    )
    return parser


def _parse_glossary(entries: list[str] | None) -> dict[str, str]:
    glossary: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Invalid glossary entry: {entry}. Expected SOURCE=TARGET.")
        source, target = entry.split("=", 1)
        glossary[source.strip()] = target.strip()
    return glossary


def _load_runtime_transcript(args: argparse.Namespace) -> TranscriptResult:
    if args.transcript_json:
        return load_transcript_json(args.transcript_json)
    if args.workflow_json:
        workflow = load_workflow_result(Path(args.workflow_json).expanduser().resolve())
        if workflow.transcript is None:
            raise ValueError("The workflow JSON does not contain a transcript payload.")
        return workflow.transcript
    return MeetingASRAgent(get_settings()).transcribe(
        audio_path=args.audio,
        language=args.language,
        use_diarization=True,
    )


def _truncate_transcript(transcript: TranscriptResult, max_segments: int | None) -> TranscriptResult:
    if max_segments is None or max_segments <= 0 or len(transcript.segments) <= max_segments:
        return transcript
    truncated_segments = transcript.segments[:max_segments]
    full_text = "\n".join(f"[{segment.speaker}] {segment.text}" for segment in truncated_segments)
    return transcript.model_copy(
        update={
            "segments": truncated_segments,
            "full_text": full_text,
            "metadata": {
                **transcript.metadata,
                "truncated_segment_count": len(truncated_segments),
                "original_segment_count": len(transcript.segments),
            },
        }
    )


def _count_completed(result: MeetingWorkflowResult) -> int:
    return sum(
        1
        for item in [result.summary, result.translation, result.action_items, result.sentiment]
        if item is not None
    )


def main() -> None:
    args = build_parser().parse_args()
    provider = LLMProvider(args.provider)
    glossary = _parse_glossary(args.glossary)
    settings = get_settings()
    shared_transcript = _truncate_transcript(_load_runtime_transcript(args), args.max_segments)

    orchestrator = MeetingOrchestrator(
        settings=settings,
        asr_agent=StaticASR(shared_transcript),
    )
    serial = SerialMeetingPipeline(settings=settings)

    parallel_result = orchestrator.run(
        audio_path=args.audio or "shared-transcript.wav",
        language=args.language,
        provider=provider,
        target_language=args.target_language,
        glossary=glossary,
        sentiment_route=args.sentiment_route,
        persist_summary=False,
    )
    serial_result = serial.run(
        transcript=shared_transcript,
        language=args.language,
        provider=provider,
        target_language=args.target_language,
        glossary=glossary,
        sentiment_route=args.sentiment_route,
        persist_summary=False,
        fail_fast=True,
    )

    failure_text = "\n".join(
        [
            "[PM] We need a quick architecture check.",
            "[ENG] The baseline should keep running even if translation fails.",
            "[OPS] Let's document the difference clearly.",
        ]
    )
    failure_transcript = TranscriptResult(
        audio_path="synthetic_failure_demo.txt",
        language="en",
        asr_model="synthetic",
        diarization_backend="synthetic",
        segments=parse_labelled_lines(failure_text),
        full_text=failure_text,
        metadata={},
    )
    failing_orchestrator = MeetingOrchestrator(
        settings=settings,
        asr_agent=StaticASR(failure_transcript),
        summary_agent=FakeSummaryAgent(),
        translation_agent=BrokenTranslationAgent(),
        action_item_agent=FakeActionItemAgent(),
        sentiment_agent=FakeSentimentAgent(),
    )
    failing_serial = SerialMeetingPipeline(
        settings=settings,
        asr_agent=StaticASR(failure_transcript),
        summary_agent=FakeSummaryAgent(),
        translation_agent=BrokenTranslationAgent(),
        action_item_agent=FakeActionItemAgent(),
        sentiment_agent=FakeSentimentAgent(),
    )

    failure_parallel = failing_orchestrator.run(
        audio_path="synthetic_failure_demo.txt",
        language="en",
        provider=provider,
        selected_agents=["summary", "translation", "action_items", "sentiment"],
        persist_summary=False,
    )
    failure_serial = failing_serial.run(
        transcript=failure_transcript,
        language="en",
        provider=provider,
        selected_agents=["summary", "translation", "action_items", "sentiment"],
        persist_summary=False,
        fail_fast=True,
    )

    output = {
        "provider": provider.value,
        "runtime_compare": {
            "transcript_segment_count": len(shared_transcript.segments),
            "parallel_latency_seconds": parallel_result.metadata["workflow_latency_seconds"],
            "serial_latency_seconds": serial_result.metadata["workflow_latency_seconds"],
            "latency_delta_seconds": round(
                float(serial_result.metadata["workflow_latency_seconds"]) - float(parallel_result.metadata["workflow_latency_seconds"]),
                3,
            ),
            "speedup": round(
                float(serial_result.metadata["workflow_latency_seconds"]) / float(parallel_result.metadata["workflow_latency_seconds"]),
                6,
            )
            if float(parallel_result.metadata["workflow_latency_seconds"]) > 0
            else None,
            "parallel_errors": parallel_result.errors,
            "serial_errors": serial_result.errors,
        },
        "error_isolation_demo": {
            "failure_agent": "translation",
            "parallel_completed_agents": _count_completed(failure_parallel),
            "serial_completed_agents": _count_completed(failure_serial),
            "parallel_errors": failure_parallel.errors,
            "serial_errors": failure_serial.errors,
        },
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
