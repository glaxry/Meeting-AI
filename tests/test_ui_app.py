from __future__ import annotations

import json

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
import ui.app as app_module
from ui.app import (
    analyze_via_api,
    format_action_items,
    format_history,
    format_sentiment,
    format_summary,
    format_transcript,
    format_translation,
    parse_glossary,
)


def test_format_action_items_returns_readable_lines() -> None:
    text = format_action_items(
        ActionItemResult(
            items=[
                ActionItem(
                    assignee="Alice",
                    task="Ship next Tuesday",
                    deadline="next Tuesday",
                    priority=ActionItemPriority.HIGH,
                    source_quote="Please ship next Tuesday.",
                )
            ]
        )
    )

    assert "Alice" in text
    assert "Ship next Tuesday" in text


def test_format_history_renders_scores() -> None:
    result = MeetingWorkflowResult(
        history=[RetrievalRecord(meeting_id="meeting-1", document="Previous decision", score=0.91)],
    )

    text = format_history(result)

    assert "meeting-1" in text
    assert "0.910" in text


def test_parse_glossary_returns_mapping() -> None:
    assert parse_glossary("语音识别=speech-recognition\n预算=budget") == {
        "语音识别": "speech-recognition",
        "预算": "budget",
    }


def test_analyze_via_api_posts_multipart_request(monkeypatch) -> None:
    audio_path = app_module.ROOT / "data" / "samples" / "asr_example_zh.wav"
    captured = {}

    class DummyResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return MeetingWorkflowResult(selected_agents=[]).model_dump(mode="json")

    def fake_post(url, data, files, timeout):
        captured["url"] = url
        captured["data"] = data
        captured["file_name"] = files["audio"][0]
        captured["file_open"] = not files["audio"][1].closed
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(app_module.requests, "post", fake_post)

    result = analyze_via_api(
        str(audio_path),
        language="zh",
        provider="deepseek",
        selected_agents=[],
        target_language="en",
        sentiment_route="llm",
        history_query="last decision",
        glossary={"语音识别": "speech-recognition"},
        use_diarization=True,
        num_speakers=None,
    )

    assert captured["url"].endswith("/meetings/analyze")
    assert json.loads(captured["data"]["selected_agents"]) == []
    assert json.loads(captured["data"]["glossary"]) == {"语音识别": "speech-recognition"}
    assert captured["file_name"] == "asr_example_zh.wav"
    assert captured["file_open"] is True
    assert captured["timeout"] is None
    assert result.selected_agents == []


def test_format_summary_renders_sections() -> None:
    result = MeetingWorkflowResult(
        selected_agents=["summary"],
        summary=SummaryResult(topics=["Launch"], decisions=["Ship"], follow_ups=["Send memo"]),
    )

    text = format_summary(result)

    assert "Topics:" in text
    assert "- Ship" in text


def test_format_summary_respects_unselected_agent() -> None:
    result = MeetingWorkflowResult(
        selected_agents=["translation"],
        summary=SummaryResult(topics=["Launch"], decisions=["Ship"], follow_ups=[]),
    )

    assert format_summary(result) == "Summary was not selected."


def test_format_translation_returns_readable_lines() -> None:
    result = TranslationResult(
        source_language="zh",
        target_language="en",
        segments=[TranscriptSegment(speaker="SPEAKER_00", text="Hello", start=0.0, end=1.0)],
        full_text="[SPEAKER_00] Hello",
    )

    text = format_translation(result, selected_agents=["translation"])

    assert "Source language: zh" in text
    assert "SPEAKER_00" in text
    assert "{" not in text


def test_format_sentiment_returns_readable_lines() -> None:
    result = SentimentResult(
        route="llm",
        overall_tone=SentimentLabel.AGREEMENT,
        segments=[
            SentimentSegment(
                text="Sounds good.",
                sentiment=SentimentLabel.AGREEMENT,
                confidence=0.9,
                speaker="SPEAKER_00",
                start=0.0,
                end=1.0,
            )
        ],
        timeline=[],
    )

    text = format_sentiment(result, selected_agents=["sentiment"])

    assert "Overall tone: agreement" in text
    assert "agreement (0.90)" in text
    assert "{" not in text


def test_format_transcript_shows_week2_annotations() -> None:
    result = MeetingWorkflowResult(
        transcript=TranscriptResult(
            audio_path="demo.wav",
            language="zh",
            asr_model="iic/SenseVoiceSmall",
            diarization_backend="mock",
            segments=[
                TranscriptSegment(
                    speaker="SPEAKER_00",
                    text="欢迎大家",
                    start=0.0,
                    end=1.0,
                    emotion="neutral",
                    event="speech",
                    metadata={"speaker_confidence": "high"},
                )
            ],
            full_text="[SPEAKER_00] 欢迎大家",
            metadata={},
        )
    )

    text = format_transcript(result)

    assert "emotion=neutral" in text
    assert "event=speech" in text
    assert "speaker_confidence=high" in text


def test_format_sentiment_includes_timeline_snapshot_lines() -> None:
    result = SentimentResult(
        route="transformer",
        overall_tone=SentimentLabel.TENSION,
        segments=[
            SentimentSegment(
                text="This delay is a serious risk.",
                sentiment=SentimentLabel.TENSION,
                confidence=0.88,
                speaker="SPEAKER_01",
                start=2.0,
                end=3.0,
            )
        ],
        timeline=[
            {
                "window_start": 0.0,
                "window_end": 120.0,
                "dominant_label": "tension",
                "label_distribution": {"tension": 1.0},
                "speakers_involved": ["SPEAKER_01"],
            }
        ],
    )

    text = format_sentiment(result, selected_agents=["sentiment"])

    assert "Timeline snapshots:" in text
    assert "speakers=SPEAKER_01" in text


def test_format_action_items_respects_unselected_agent() -> None:
    assert format_action_items(ActionItemResult(), selected_agents=["summary"]) == "Action item extraction was not selected."
