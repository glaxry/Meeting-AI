from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import MeetingAISettings, get_settings
from .schemas import ChatMessage, LLMProvider, LLMResponse


LOGGER = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class ProviderConfig(BaseModel):
    provider: LLMProvider
    api_key: str
    base_url: str
    model: str


class UnifiedLLMClient:
    def __init__(
        self,
        settings: MeetingAISettings | None = None,
        http_clients: dict[LLMProvider, httpx.Client] | None = None,
    ):
        self.settings = settings or get_settings()
        self.http_clients = http_clients or {}
        self._clients: dict[LLMProvider, OpenAI] = {}

    def _provider_config(self, provider: LLMProvider) -> ProviderConfig:
        if provider == LLMProvider.DEEPSEEK:
            api_key = self.settings.resolved_deepseek_api_key
            if not api_key:
                raise ValueError("DeepSeek API key is missing. Set DEEPSEEK_API_KEY or provide api-key-deepseek.")
            return ProviderConfig(
                provider=provider,
                api_key=api_key,
                base_url=self.settings.deepseek_base_url,
                model=self.settings.deepseek_model,
            )

        if provider == LLMProvider.QWEN:
            api_key = self.settings.resolved_qwen_api_key
            if not api_key:
                raise ValueError("Qwen API key is missing. Set QWEN_API_KEY in .env or your shell.")
            return ProviderConfig(
                provider=provider,
                api_key=api_key,
                base_url=self.settings.qwen_base_url,
                model=self.settings.qwen_model,
            )

        raise ValueError(f"Unsupported provider: {provider}")

    def _get_client(self, provider: LLMProvider) -> tuple[OpenAI, ProviderConfig]:
        config = self._provider_config(provider)
        if provider not in self._clients:
            kwargs: dict[str, Any] = {
                "api_key": config.api_key,
                "base_url": config.base_url,
                "timeout": self.settings.llm_timeout_seconds,
                "max_retries": 0,
            }
            if provider in self.http_clients:
                kwargs["http_client"] = self.http_clients[provider]
            self._clients[provider] = OpenAI(**kwargs)
        return self._clients[provider], config

    def _request_with_retry(
        self,
        client: OpenAI,
        model: str,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ):
        retryer = Retrying(
            stop=stop_after_attempt(self.settings.llm_max_retries),
            wait=wait_exponential(
                multiplier=self.settings.llm_retry_backoff_seconds,
                min=self.settings.llm_retry_backoff_seconds,
                max=8,
            ),
            retry=retry_if_exception_type(
                (APIConnectionError, APIError, APITimeoutError, RateLimitError, httpx.HTTPError)
            ),
            reraise=True,
        )

        for attempt in retryer:
            with attempt:
                return client.chat.completions.create(
                    model=model,
                    messages=[message.model_dump() for message in messages],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )

        raise RuntimeError("Retry loop exited unexpectedly.")

    def chat(
        self,
        provider: LLMProvider,
        messages: list[ChatMessage | dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        normalized_messages = [
            message if isinstance(message, ChatMessage) else ChatMessage.model_validate(message)
            for message in messages
        ]
        client, config = self._get_client(provider)

        started = time.perf_counter()
        response = self._request_with_retry(
            client=client,
            model=config.model,
            messages=normalized_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        latency = round(time.perf_counter() - started, 3)

        message_content = response.choices[0].message.content
        if isinstance(message_content, str):
            content = message_content
        elif message_content is None:
            content = ""
        else:
            content = json.dumps(message_content, ensure_ascii=False)

        return LLMResponse(
            provider=provider,
            model=config.model,
            content=content,
            latency_seconds=latency,
            raw=response.model_dump(),
        )

    def prompt(
        self,
        provider: LLMProvider,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))
        return self.chat(
            provider=provider,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call a configured LLM provider.")
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in LLMProvider],
        default=LLMProvider.DEEPSEEK.value,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", help="Inline user prompt.")
    group.add_argument("--prompt-file", help="Path to a UTF-8 text file containing the prompt.")
    parser.add_argument("--system", help="Optional system prompt.")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--raw-json", help="Optional file path to store the raw JSON response.")
    return parser


def main() -> None:
    _setup_logging()
    args = build_parser().parse_args()

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).expanduser().resolve().read_text(encoding="utf-8")

    provider = LLMProvider(args.provider)
    client = UnifiedLLMClient(settings=get_settings())
    response = client.prompt(
        provider=provider,
        prompt=prompt,
        system_prompt=args.system,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.raw_json:
        raw_path = Path(args.raw_json).expanduser().resolve()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(response.raw, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Raw response saved to %s", raw_path)

    print(response.content)


if __name__ == "__main__":
    main()
