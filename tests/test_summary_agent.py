from __future__ import annotations

import json

from meeting_ai.config import MeetingAISettings
from meeting_ai.schemas import LLMProvider, LLMResponse, TranscriptResult, TranscriptSegment
from meeting_ai.summary_agent import SummaryAgent


class QueuedLLMClient:
    def __init__(self, payloads: list[dict[str, object]]):
        self.payloads = payloads
        self.calls: list[dict[str, object]] = []

    def prompt(self, provider, prompt, system_prompt=None, temperature=0.2, max_tokens=None, response_format=None):
        self.calls.append(
            {
                "provider": provider,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "response_format": response_format,
            }
        )
        payload = self.payloads.pop(0)
        return LLMResponse(
            provider=provider,
            model="mock-model",
            content=json.dumps(payload, ensure_ascii=False),
            latency_seconds=0.01,
            raw={},
        )


def build_transcript(texts: list[str]) -> TranscriptResult:
    segments = [
        TranscriptSegment(
            speaker=f"SPEAKER_{index:02d}",
            text=text,
            start=float(index),
            end=float(index + 1),
        )
        for index, text in enumerate(texts)
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


def test_summary_agent_uses_single_pass_for_short_input() -> None:
    llm_client = QueuedLLMClient(
        payloads=[
            {
                "topics": ["project status", "project status"],
                "decisions": ["ship the pilot next week"],
                "follow_ups": ["Alice sends the notes"],
            }
        ]
    )
    agent = SummaryAgent(
        settings=MeetingAISettings(summary_map_reduce_threshold=100),
        llm_client=llm_client,
        provider=LLMProvider.DEEPSEEK,
    )

    result = agent.summarize(text="[SPEAKER_00] Brief status sync.")

    assert result.topics == ["project status"]
    assert result.metadata["strategy"] == "single_pass"
    assert len(llm_client.calls) == 1


def test_summary_agent_uses_map_reduce_for_long_transcript() -> None:
    transcript = build_transcript(
        [
            "First we reviewed the launch blockers in detail.",
            "Then we assigned owners for the vendor follow-up and QA sign-off.",
            "Finally we agreed to move the pilot to next Tuesday.",
        ]
    )
    llm_client = QueuedLLMClient(
        payloads=[
            {"topics": ["launch blockers"], "decisions": [], "follow_ups": ["vendor follow-up"]},
            {"topics": ["owners"], "decisions": ["QA signs off"], "follow_ups": []},
            {"topics": ["pilot plan"], "decisions": [], "follow_ups": []},
            {"topics": ["launch blockers", "pilot plan"], "decisions": ["pilot moves to next Tuesday"], "follow_ups": []},
        ]
    )
    agent = SummaryAgent(
        settings=MeetingAISettings(summary_map_reduce_threshold=3, summary_chunk_target_words=6),
        llm_client=llm_client,
        provider=LLMProvider.DEEPSEEK,
    )

    result = agent.summarize(transcript=transcript)

    assert result.metadata["strategy"] == "map_reduce"
    assert result.decisions == ["pilot moves to next Tuesday"]
    assert len(llm_client.calls) == 4
