from __future__ import annotations

import json
import sys
import types

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


class FakeGeneration:
    def __init__(self, observed: dict[str, object], start_kwargs: dict[str, object]):
        self.observed = observed
        self.start_kwargs = start_kwargs

    def __enter__(self):
        self.observed["start_kwargs"] = self.start_kwargs
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, **kwargs):
        self.observed.setdefault("updates", []).append(kwargs)


def test_llm_client_emits_langfuse_generation_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class FakeLangfuse:
        def __init__(self, **kwargs):
            observed["init_kwargs"] = kwargs

        def start_as_current_observation(self, **kwargs):
            return FakeGeneration(observed, kwargs)

    monkeypatch.setitem(sys.modules, "langfuse", types.SimpleNamespace(Langfuse=FakeLangfuse))

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
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
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )
    )
    http_client = httpx.Client(transport=transport)
    settings = MeetingAISettings(
        deepseek_api_key="test-key",
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
        langfuse_host="http://localhost:3000",
    )
    client = UnifiedLLMClient(settings=settings, http_clients={LLMProvider.DEEPSEEK: http_client})

    response = client.prompt(
        provider=LLMProvider.DEEPSEEK,
        prompt="hello",
        system_prompt="system",
        temperature=0.3,
        max_tokens=123,
    )

    assert response.content == "ok"
    assert observed["init_kwargs"] == {
        "public_key": "pk-lf-test",
        "secret_key": "sk-lf-test",
        "host": "http://localhost:3000",
    }
    assert observed["start_kwargs"]["as_type"] == "generation"
    assert observed["start_kwargs"]["name"] == "llm.chat"
    assert observed["start_kwargs"]["metadata"]["provider"] == "deepseek"
    assert observed["start_kwargs"]["model_parameters"]["max_tokens"] == 123
    assert observed["updates"][0]["output"] == "ok"
    assert observed["updates"][0]["usage_details"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
