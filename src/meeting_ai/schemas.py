from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    DEEPSEEK = "deepseek"
    QWEN = "qwen"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class DiarizationSegment(BaseModel):
    speaker: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    speaker: str
    text: str
    start: float
    end: float
    raw_text: str | None = None


class TranscriptResult(BaseModel):
    audio_path: str
    language: str
    asr_model: str
    diarization_backend: str
    segments: list[TranscriptSegment]
    full_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    provider: LLMProvider
    model: str
    content: str
    latency_seconds: float
    raw: dict[str, Any] = Field(default_factory=dict)

