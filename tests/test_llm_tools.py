from __future__ import annotations

import json

import httpx
import pytest

from meeting_ai.config import MeetingAISettings
from meeting_ai.llm_tools import UnifiedLLMClient
from meeting_ai.schemas import LLMProvider


def test_llm_client_uses_openai_compatible_payload() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-chat",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    settings = MeetingAISettings(deepseek_api_key="test-key")
    client = UnifiedLLMClient(settings=settings, http_clients={LLMProvider.DEEPSEEK: http_client})

    response = client.prompt(
        provider=LLMProvider.DEEPSEEK,
        prompt="hello",
        system_prompt="system",
    )

    assert response.content == "ok"
    assert captured["authorization"] == "Bearer test-key"
    assert str(captured["url"]).endswith("/chat/completions")
    assert captured["body"]["messages"][0]["role"] == "system"
    assert captured["body"]["messages"][1]["content"] == "hello"


def test_llm_client_requires_configured_key() -> None:
    settings = MeetingAISettings(deepseek_api_key=None, deepseek_key_file="missing-key-file")
    client = UnifiedLLMClient(settings=settings)

    with pytest.raises(ValueError):
        client.prompt(provider=LLMProvider.DEEPSEEK, prompt="hello")
