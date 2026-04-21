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


class SummaryResult(BaseModel):
    topics: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TranslationResult(BaseModel):
    source_language: str
    target_language: str
    segments: list[TranscriptSegment]
    full_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionItemPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionItem(BaseModel):
    assignee: str | None = None
    task: str
    deadline: str | None = None
    priority: ActionItemPriority = ActionItemPriority.MEDIUM
    source_quote: str
    implicit: bool = False


class ActionItemResult(BaseModel):
    items: list[ActionItem] = Field(default_factory=list)
    reasoning: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SentimentLabel(str, Enum):
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    HESITATION = "hesitation"
    TENSION = "tension"
    NEUTRAL = "neutral"


class SentimentSegment(BaseModel):
    text: str
    sentiment: SentimentLabel
    confidence: float
    speaker: str | None = None
    start: float | None = None
    end: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SentimentResult(BaseModel):
    route: str
    overall_tone: SentimentLabel
    segments: list[SentimentSegment] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalRecord(BaseModel):
    meeting_id: str
    document: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MeetingWorkflowResult(BaseModel):
    transcript: TranscriptResult | None = None
    summary: SummaryResult | None = None
    translation: TranslationResult | None = None
    action_items: ActionItemResult | None = None
    sentiment: SentimentResult | None = None
    history: list[RetrievalRecord] = Field(default_factory=list)
    selected_agents: list[str] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
