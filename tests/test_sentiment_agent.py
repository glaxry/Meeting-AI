from __future__ import annotations

import json

from meeting_ai.config import MeetingAISettings
from meeting_ai.schemas import LLMProvider, LLMResponse, SentimentLabel, TranscriptSegment
from meeting_ai.sentiment_agent import SentimentAgent, TransformersSentimentClassifier


class QueuedLLMClient:
    def __init__(self, payloads: list[dict[str, object]]):
        self.payloads = payloads

    def prompt(self, provider, prompt, system_prompt=None, temperature=0.2, max_tokens=None, response_format=None):
        payload = self.payloads.pop(0)
        return LLMResponse(
            provider=provider,
            model="mock-model",
            content=json.dumps(payload, ensure_ascii=False),
            latency_seconds=0.01,
            raw={},
        )


class FakePipeline:
    def __call__(self, texts):
        outputs = []
        for text in texts:
            if "agree" in text.lower():
                outputs.append(
                    [
                        {"label": "positive", "score": 0.91},
                        {"label": "negative", "score": 0.09},
                    ]
                )
            elif "risk" in text.lower():
                outputs.append(
                    [
                        {"label": "negative", "score": 0.88},
                        {"label": "positive", "score": 0.12},
                    ]
                )
            else:
                outputs.append(
                    [
                        {"label": "positive", "score": 0.55},
                        {"label": "negative", "score": 0.45},
                    ]
                )
        return outputs


def build_segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(speaker="A", text="I agree with this plan.", start=0.0, end=1.0),
        TranscriptSegment(speaker="B", text="Maybe we should wait one more day.", start=1.0, end=2.0),
        TranscriptSegment(speaker="C", text="This delay is a serious risk.", start=2.0, end=3.0),
    ]


def test_sentiment_agent_llm_route_returns_structured_output() -> None:
    llm_client = QueuedLLMClient(
        payloads=[
            {
                "overall_tone": "disagreement",
                "segments": [
                    {"index": 0, "sentiment": "agreement", "confidence": 0.91},
                    {"index": 1, "sentiment": "hesitation", "confidence": 0.77},
                    {"index": 2, "sentiment": "disagreement", "confidence": 0.85},
                ],
            }
        ]
    )
    agent = SentimentAgent(
        settings=MeetingAISettings(),
        llm_client=llm_client,
        provider=LLMProvider.DEEPSEEK,
    )

    result = agent.analyze(route="llm", transcript=build_segments())

    assert result.route == "llm"
    assert result.overall_tone == SentimentLabel.DISAGREEMENT
    assert result.segments[1].sentiment == SentimentLabel.HESITATION
    assert result.segments[2].speaker == "C"


def test_sentiment_agent_transformer_route_normalizes_to_five_labels() -> None:
    classifier = TransformersSentimentClassifier(
        settings=MeetingAISettings(sentiment_transformer_model="fake-model"),
        classifier_pipeline=FakePipeline(),
    )
    agent = SentimentAgent(
        settings=MeetingAISettings(sentiment_transformer_model="fake-model"),
        transformer_classifier=classifier,
    )

    result = agent.analyze(route="transformer", transcript=build_segments())

    assert result.route == "transformer"
    assert result.segments[0].sentiment == SentimentLabel.AGREEMENT
    assert result.segments[1].sentiment == SentimentLabel.HESITATION
    assert result.segments[2].sentiment == SentimentLabel.TENSION
    assert result.overall_tone == SentimentLabel.TENSION


def test_sentiment_agent_llm_route_tolerates_missing_overall_tone() -> None:
    llm_client = QueuedLLMClient(
        payloads=[
            {
                "segments": [
                    {"index": 0, "sentiment": "agreement", "confidence": 0.88},
                    {"index": 1, "sentiment": "neutral", "confidence": 0.51},
                    {"index": 2, "sentiment": "tension", "confidence": 0.9},
                ]
            }
        ]
    )
    agent = SentimentAgent(
        settings=MeetingAISettings(),
        llm_client=llm_client,
        provider=LLMProvider.DEEPSEEK,
    )

    result = agent.analyze(route="llm", transcript=build_segments())

    assert result.overall_tone == SentimentLabel.TENSION
    assert len(result.segments) == 3
