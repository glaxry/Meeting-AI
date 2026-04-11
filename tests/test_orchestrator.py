from __future__ import annotations

from meeting_ai.orchestrator import MeetingOrchestrator
from meeting_ai.schemas import (
    ActionItem,
    ActionItemResult,
    ActionItemPriority,
    LLMProvider,
    RetrievalRecord,
    SentimentLabel,
    SentimentResult,
    SentimentSegment,
    SummaryResult,
    TranscriptResult,
    TranscriptSegment,
    TranslationResult,
)


def build_transcript() -> TranscriptResult:
    segment = TranscriptSegment(speaker="SPEAKER_00", text="Please ship next Tuesday.", start=0.0, end=1.0)
    return TranscriptResult(
        audio_path="demo.wav",
        language="en",
        asr_model="mock",
        diarization_backend="mock",
        segments=[segment],
        full_text="[SPEAKER_00] Please ship next Tuesday.",
        metadata={},
    )


class FakeASR:
    def transcribe(self, **kwargs):
        return build_transcript()


class FakeSummaryAgent:
    def __init__(self):
        self.provider = LLMProvider.DEEPSEEK

    def summarize(self, **kwargs):
        return SummaryResult(
            topics=["Launch"],
            decisions=["Ship next Tuesday"],
            follow_ups=["Send plan"],
            metadata={"provider": self.provider.value},
        )


class FakeTranslationAgent:
    def __init__(self):
        self.provider = LLMProvider.DEEPSEEK

    def translate(self, **kwargs):
        transcript = kwargs["transcript"]
        return TranslationResult(
            source_language="en",
            target_language="zh",
            segments=transcript.segments,
            full_text=transcript.full_text,
            metadata={"provider": self.provider.value},
        )


class BrokenTranslationAgent:
    def __init__(self):
        self.provider = LLMProvider.DEEPSEEK

    def translate(self, **kwargs):
        raise RuntimeError("translation failed")


class FakeActionItemAgent:
    def __init__(self):
        self.provider = LLMProvider.DEEPSEEK

    def extract(self, **kwargs):
        return ActionItemResult(
            items=[
                ActionItem(
                    assignee="Alice",
                    task="Ship next Tuesday",
                    deadline="next Tuesday",
                    priority=ActionItemPriority.HIGH,
                    source_quote="Please ship next Tuesday.",
                )
            ],
            metadata={"provider": self.provider.value},
        )


class FakeSentimentAgent:
    def __init__(self):
        self.provider = LLMProvider.DEEPSEEK

    def analyze(self, **kwargs):
        return SentimentResult(
            route="llm",
            overall_tone=SentimentLabel.AGREEMENT,
            segments=[
                SentimentSegment(
                    text="Please ship next Tuesday.",
                    sentiment=SentimentLabel.AGREEMENT,
                    confidence=0.9,
                    speaker="SPEAKER_00",
                )
            ],
            metadata={"provider": self.provider.value},
        )


class FakeStore:
    def __init__(self):
        self.stored: list[str] = []
        self.queries: list[str] = []

    def add_summary(self, summary, transcript=None, meeting_id=None, metadata=None):
        self.stored.append(summary.decisions[0])
        return "stored-1"

    def query(self, question, top_k=3):
        self.queries.append(question)
        return [RetrievalRecord(meeting_id="previous-1", document="Previous launch decision", score=0.9)]


def test_orchestrator_runs_selected_agents_and_persists_summary() -> None:
    store = FakeStore()
    orchestrator = MeetingOrchestrator(
        asr_agent=FakeASR(),
        summary_agent=FakeSummaryAgent(),
        translation_agent=FakeTranslationAgent(),
        action_item_agent=FakeActionItemAgent(),
        sentiment_agent=FakeSentimentAgent(),
        vector_store=store,
    )

    result = orchestrator.run(
        audio_path="demo.wav",
        selected_agents=["summary", "action_items"],
        history_query="What did we decide last time?",
    )

    assert result.transcript is not None
    assert result.summary is not None
    assert result.action_items is not None
    assert result.translation is None
    assert result.sentiment is None
    assert result.history[0].meeting_id == "previous-1"
    assert result.metadata["stored_meeting_id"] == "stored-1"
    assert store.stored == ["Ship next Tuesday"]


def test_orchestrator_isolates_agent_errors() -> None:
    store = FakeStore()
    orchestrator = MeetingOrchestrator(
        asr_agent=FakeASR(),
        summary_agent=FakeSummaryAgent(),
        translation_agent=BrokenTranslationAgent(),
        action_item_agent=FakeActionItemAgent(),
        sentiment_agent=FakeSentimentAgent(),
        vector_store=store,
    )

    result = orchestrator.run(
        audio_path="demo.wav",
        selected_agents=["summary", "translation", "sentiment"],
    )

    assert result.summary is not None
    assert result.sentiment is not None
    assert result.translation is None
    assert "translation" in result.errors


def test_orchestrator_propagates_provider_to_agents() -> None:
    store = FakeStore()
    summary_agent = FakeSummaryAgent()
    translation_agent = FakeTranslationAgent()
    action_item_agent = FakeActionItemAgent()
    sentiment_agent = FakeSentimentAgent()
    orchestrator = MeetingOrchestrator(
        asr_agent=FakeASR(),
        summary_agent=summary_agent,
        translation_agent=translation_agent,
        action_item_agent=action_item_agent,
        sentiment_agent=sentiment_agent,
        vector_store=store,
    )

    result = orchestrator.run(
        audio_path="demo.wav",
        provider=LLMProvider.QWEN,
        selected_agents=["summary", "translation", "action_items", "sentiment"],
    )

    assert result.summary.metadata["provider"] == "qwen"
    assert result.translation.metadata["provider"] == "qwen"
    assert result.action_items.metadata["provider"] == "qwen"
    assert result.sentiment.metadata["provider"] == "qwen"


def test_orchestrator_treats_empty_selected_agents_as_none_selected() -> None:
    store = FakeStore()
    orchestrator = MeetingOrchestrator(
        asr_agent=FakeASR(),
        summary_agent=FakeSummaryAgent(),
        translation_agent=FakeTranslationAgent(),
        action_item_agent=FakeActionItemAgent(),
        sentiment_agent=FakeSentimentAgent(),
        vector_store=store,
    )

    result = orchestrator.run(audio_path="demo.wav", selected_agents=[])

    assert result.selected_agents == []
    assert result.summary is None
    assert result.translation is None
    assert result.action_items is None
    assert result.sentiment is None
