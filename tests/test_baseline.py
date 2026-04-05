from __future__ import annotations

from meeting_ai.baseline import SerialMeetingPipeline
from meeting_ai.schemas import (
    ActionItemResult,
    LLMProvider,
    SentimentLabel,
    SentimentResult,
    SentimentSegment,
    SummaryResult,
    TranscriptResult,
    TranscriptSegment,
    TranslationResult,
)


def build_transcript() -> TranscriptResult:
    segment = TranscriptSegment(speaker="SPEAKER_00", text="Ship next Tuesday.", start=0.0, end=1.0)
    return TranscriptResult(
        audio_path="demo.wav",
        language="en",
        asr_model="mock",
        diarization_backend="mock",
        segments=[segment],
        full_text="[SPEAKER_00] Ship next Tuesday.",
        metadata={},
    )


class FakeSummaryAgent:
    def __init__(self):
        self.provider = LLMProvider.DEEPSEEK

    def summarize(self, **kwargs):
        return SummaryResult(topics=["Launch"], decisions=["Ship"], follow_ups=[], metadata={"provider": self.provider.value})


class FakeTranslationAgent:
    def __init__(self):
        self.provider = LLMProvider.DEEPSEEK

    def translate(self, transcript, **kwargs):
        return TranslationResult(
            source_language="en",
            target_language="zh",
            segments=transcript.segments,
            full_text=transcript.full_text,
            metadata={"provider": self.provider.value},
        )


class BrokenTranslationAgent(FakeTranslationAgent):
    def translate(self, transcript, **kwargs):
        raise RuntimeError("translation failed")


class FakeActionAgent:
    def __init__(self):
        self.provider = LLMProvider.DEEPSEEK

    def extract(self, **kwargs):
        return ActionItemResult(metadata={"provider": self.provider.value})


class FakeSentimentAgent:
    def __init__(self):
        self.provider = LLMProvider.DEEPSEEK

    def analyze(self, **kwargs):
        return SentimentResult(
            route="llm",
            overall_tone=SentimentLabel.NEUTRAL,
            segments=[SentimentSegment(text="Ship next Tuesday.", sentiment=SentimentLabel.NEUTRAL, confidence=0.5)],
            metadata={"provider": self.provider.value},
        )


class FakeStore:
    def add_summary(self, summary, transcript=None, meeting_id=None, metadata=None):
        return "stored-serial"

    def query(self, question, top_k=3):
        return []


def test_serial_pipeline_fail_fast_stops_after_error() -> None:
    pipeline = SerialMeetingPipeline(
        summary_agent=FakeSummaryAgent(),
        translation_agent=BrokenTranslationAgent(),
        action_item_agent=FakeActionAgent(),
        sentiment_agent=FakeSentimentAgent(),
        vector_store=FakeStore(),
    )

    result = pipeline.run(
        transcript=build_transcript(),
        selected_agents=["summary", "translation", "action_items", "sentiment"],
        fail_fast=True,
        persist_summary=False,
    )

    assert result.summary is not None
    assert result.translation is None
    assert result.action_items is None
    assert result.sentiment is None
    assert "translation" in result.errors


def test_serial_pipeline_propagates_provider() -> None:
    pipeline = SerialMeetingPipeline(
        summary_agent=FakeSummaryAgent(),
        translation_agent=FakeTranslationAgent(),
        action_item_agent=FakeActionAgent(),
        sentiment_agent=FakeSentimentAgent(),
        vector_store=FakeStore(),
    )

    result = pipeline.run(
        transcript=build_transcript(),
        provider=LLMProvider.QWEN,
        selected_agents=["summary", "translation", "action_items", "sentiment"],
        persist_summary=False,
    )

    assert result.summary.metadata["provider"] == "qwen"
    assert result.translation.metadata["provider"] == "qwen"
    assert result.action_items.metadata["provider"] == "qwen"
    assert result.sentiment.metadata["provider"] == "qwen"
