from __future__ import annotations

import json

from meeting_ai.config import MeetingAISettings
from meeting_ai.schemas import LLMProvider, LLMResponse, TranscriptResult, TranscriptSegment
from meeting_ai.translation_agent import TranslationAgent


class QueuedLLMClient:
    def __init__(self, payloads: list[dict[str, object]]):
        self.payloads = payloads
        self.calls: list[dict[str, object]] = []

    def prompt(self, provider, prompt, system_prompt=None, temperature=0.2, max_tokens=None, response_format=None):
        self.calls.append({"provider": provider, "prompt": prompt, "system_prompt": system_prompt})
        payload = self.payloads.pop(0)
        return LLMResponse(
            provider=provider,
            model="mock-model",
            content=json.dumps(payload, ensure_ascii=False),
            latency_seconds=0.01,
            raw={},
        )


def build_transcript() -> TranscriptResult:
    segments = [
        TranscriptSegment(speaker="Alice", text="今天先看预算。", start=0.0, end=1.0),
        TranscriptSegment(speaker="Bob", text="下周二前给我更新。", start=1.0, end=2.0),
    ]
    return TranscriptResult(
        audio_path="demo.wav",
        language="zh",
        asr_model="mock-asr",
        diarization_backend="mock",
        segments=segments,
        full_text="\n".join(f"[{segment.speaker}] {segment.text}" for segment in segments),
        metadata={},
    )


def test_translation_agent_preserves_speaker_labels_and_offsets() -> None:
    llm_client = QueuedLLMClient(
        payloads=[
            {
                "segments": [
                    {"speaker": "Alice", "text": "Let's review the budget first."},
                    {"speaker": "Bob", "text": "Send me an update by next Tuesday."},
                ]
            }
        ]
    )
    agent = TranslationAgent(
        settings=MeetingAISettings(summary_chunk_target_words=100),
        llm_client=llm_client,
        provider=LLMProvider.DEEPSEEK,
    )

    result = agent.translate(
        source_language="zh",
        target_language="en",
        transcript=build_transcript(),
        glossary={"预算": "budget"},
    )

    assert result.segments[0].speaker == "Alice"
    assert result.segments[0].start == 0.0
    assert result.segments[0].raw_text == "今天先看预算。"
    assert "预算 => budget" in llm_client.calls[0]["prompt"]
    assert result.full_text.startswith("[Alice] Let's review the budget first.")


def test_translation_agent_can_translate_labelled_text_input() -> None:
    llm_client = QueuedLLMClient(
        payloads=[{"segments": [{"speaker": "SPEAKER_00", "text": "Please check with legal."}]}]
    )
    agent = TranslationAgent(
        settings=MeetingAISettings(summary_chunk_target_words=100),
        llm_client=llm_client,
        provider=LLMProvider.DEEPSEEK,
    )

    result = agent.translate(
        source_language="zh",
        target_language="en",
        text="[SPEAKER_00] 你去和法务确认一下。",
    )

    assert result.segments[0].speaker == "SPEAKER_00"
    assert result.segments[0].text == "Please check with legal."
