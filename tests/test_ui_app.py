from __future__ import annotations

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
    TranscriptSegment,
    TranslationResult,
)
from ui.app import format_action_items, format_history, format_sentiment, format_summary, format_translation


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
    )

    text = format_sentiment(result, selected_agents=["sentiment"])

    assert "Overall tone: agreement" in text
    assert "agreement (0.90)" in text
    assert "{" not in text


def test_format_action_items_respects_unselected_agent() -> None:
    assert format_action_items(ActionItemResult(), selected_agents=["summary"]) == "Action item extraction was not selected."
