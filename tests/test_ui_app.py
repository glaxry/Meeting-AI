from __future__ import annotations

from meeting_ai.schemas import (
    ActionItem,
    ActionItemPriority,
    ActionItemResult,
    MeetingWorkflowResult,
    RetrievalRecord,
    SummaryResult,
)
from ui.app import format_action_items, format_history, format_summary


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
        summary=SummaryResult(topics=["Launch"], decisions=["Ship"], follow_ups=["Send memo"]),
    )

    text = format_summary(result)

    assert "Topics:" in text
    assert "- Ship" in text
