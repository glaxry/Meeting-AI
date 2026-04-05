from __future__ import annotations

from pathlib import Path

from meeting_ai.reporting import build_week35_metrics, export_week35_report
from meeting_ai.schemas import (
    ActionItem,
    ActionItemPriority,
    ActionItemResult,
    MeetingWorkflowResult,
    RetrievalRecord,
    SentimentLabel,
    SentimentResult,
    SentimentSegment,
    SummaryResult,
    TranscriptResult,
    TranscriptSegment,
    TranslationResult,
)


def build_result() -> MeetingWorkflowResult:
    transcript_segments = [
        TranscriptSegment(speaker="SPEAKER_00", text="Launch discussion", start=0.0, end=2.0),
        TranscriptSegment(speaker="SPEAKER_01", text="Budget concern", start=2.0, end=4.0),
    ]
    transcript = TranscriptResult(
        audio_path="demo.wav",
        language="zh",
        asr_model="mock",
        diarization_backend="mock",
        segments=transcript_segments,
        full_text="\n".join(f"[{segment.speaker}] {segment.text}" for segment in transcript_segments),
        metadata={
            "audio_duration_seconds": 12.0,
            "asr_runtime_seconds": 1.5,
            "diarization_runtime_seconds": 2.0,
        },
    )
    summary = SummaryResult(
        topics=["Launch readiness", "Budget review"],
        decisions=["Ship on Friday"],
        follow_ups=["Send the approval memo"],
        metadata={
            "strategy": "map_reduce",
            "word_count": 600,
            "chunk_count": 2,
            "map_latencies": [0.8, 0.9],
            "reduce_latency": 1.1,
        },
    )
    translation = TranslationResult(
        source_language="zh",
        target_language="en",
        segments=[
            TranscriptSegment(speaker="SPEAKER_00", text="Launch discussion", start=0.0, end=2.0),
            TranscriptSegment(speaker="SPEAKER_01", text="Budget concern", start=2.0, end=4.0),
        ],
        full_text="[SPEAKER_00] Launch discussion\n[SPEAKER_01] Budget concern",
        metadata={"chunk_count": 2, "latencies": [1.2, 1.3]},
    )
    action_items = ActionItemResult(
        items=[
            ActionItem(
                assignee="Alice",
                task="Send the approval memo",
                deadline="Friday",
                priority=ActionItemPriority.HIGH,
                source_quote="Alice will send the approval memo by Friday.",
            )
        ],
        metadata={"chunk_count": 1, "latencies": [0.7]},
    )
    sentiment = SentimentResult(
        route="llm",
        overall_tone=SentimentLabel.NEUTRAL,
        segments=[
            SentimentSegment(text="Launch discussion", sentiment=SentimentLabel.NEUTRAL, confidence=0.8, speaker="SPEAKER_00"),
            SentimentSegment(text="Budget concern", sentiment=SentimentLabel.HESITATION, confidence=0.7, speaker="SPEAKER_01"),
        ],
        metadata={"latency_seconds": 2.5},
    )
    return MeetingWorkflowResult(
        transcript=transcript,
        summary=summary,
        translation=translation,
        action_items=action_items,
        sentiment=sentiment,
        selected_agents=["summary", "translation", "action_items", "sentiment"],
        metadata={"provider": "deepseek", "workflow_latency_seconds": 12.4, "stored_meeting_id": "meeting-123"},
    )


def test_build_week35_metrics_computes_rtf_and_counts() -> None:
    result = build_result()
    retrieval_results = [RetrievalRecord(meeting_id="meeting-123", document="Launch summary", score=0.95)]

    metrics = build_week35_metrics(result, retrieval_query="What changed?", retrieval_results=retrieval_results, generated_on="2026-04-05")

    assert metrics["runtime"]["workflow_rtf"] == 1.033
    assert metrics["summary"]["total_latency_seconds"] == 2.8
    assert metrics["action_items"]["count"] == 1
    assert metrics["transcript"]["speaker_distribution"]["SPEAKER_00"] == 1
    assert metrics["sentiment"]["distribution"]["neutral"] == 1
    assert metrics["retrieval"]["result_count"] == 1


def test_export_week35_report_writes_markdown_and_svgs(tmp_path: Path) -> None:
    result = build_result()

    artifacts = export_week35_report(
        result=result,
        output_root=tmp_path,
        retrieval_query="What changed?",
        retrieval_results=[RetrievalRecord(meeting_id="meeting-123", document="Launch summary", score=0.95)],
        generated_on="2026-04-05",
    )

    assert artifacts.report_path.exists()
    assert artifacts.metrics_path.exists()
    assert artifacts.architecture_svg_path.exists()
    assert artifacts.runtime_svg_path.exists()
    assert artifacts.speaker_svg_path.exists()
    assert artifacts.snapshot_svg_path.exists()
    assert artifacts.retrieval_svg_path.exists()
    assert "Week 3.5 Progress Report" in artifacts.report_path.read_text(encoding="utf-8")
