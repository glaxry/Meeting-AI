from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from .llm_tools import UnifiedLLMClient
from .schemas import LLMProvider, LLMResponse
from .text_utils import extract_json_payload


ModelT = TypeVar("ModelT", bound=BaseModel)


def prompt_json(
    llm_client: UnifiedLLMClient,
    provider: LLMProvider,
    schema: type[ModelT],
    prompt: str,
    system_prompt: str,
    temperature: float = 0.1,
    max_tokens: int | None = None,
) -> tuple[ModelT, LLMResponse]:
    response = llm_client.prompt(
        provider=provider,
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    payload = extract_json_payload(response.content)
    return schema.model_validate(payload), response
