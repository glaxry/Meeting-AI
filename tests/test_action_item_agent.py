from __future__ import annotations

import json

from meeting_ai.action_item_agent import ActionItemAgent
from meeting_ai.config import MeetingAISettings
from meeting_ai.schemas import LLMProvider, LLMResponse


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


def test_action_item_agent_extracts_and_deduplicates_items() -> None:
    llm_client = QueuedLLMClient(
        payloads=[
            {
                "items": [
                    {
                        "assignee": "Alice",
                        "task": "Follow up with the vendor on the pricing sheet",
                        "deadline": "tomorrow",
                        "priority": "high",
                        "source_quote": "Alice, can you follow up with the vendor tomorrow?",
                    },
                    {
                        "assignee": "Alice",
                        "task": "Follow up with the vendor on the pricing sheet",
                        "deadline": "tomorrow",
                        "priority": "high",
                        "source_quote": "You sync with the vendor tomorrow.",
                    },
                ]
            }
        ]
    )
    agent = ActionItemAgent(
        settings=MeetingAISettings(summary_chunk_target_words=100),
        llm_client=llm_client,
        provider=LLMProvider.DEEPSEEK,
    )

    result = agent.extract(text="[Alice] 你明天跟供应商同步一下报价单。")

    assert len(result.items) == 1
    assert result.items[0].assignee == "Alice"
    assert result.items[0].priority.value == "high"
    assert result.metadata["strategy"] == "single_pass"
