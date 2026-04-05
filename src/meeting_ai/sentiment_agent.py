from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import MeetingAISettings, get_settings
from .llm_tools import UnifiedLLMClient
from .schemas import LLMProvider, SentimentLabel, SentimentResult, SentimentSegment, TranscriptResult, TranscriptSegment
from .text_utils import extract_json_payload, load_text_input, load_transcript_json, parse_labelled_lines, transcript_to_segments


LOGGER = logging.getLogger(__name__)


SENTIMENT_SYSTEM_PROMPT = """You are a meeting sentiment analysis agent.
Return valid JSON only.
Use only these labels: agreement, disagreement, hesitation, tension, neutral.
The output schema must be:
{
  "overall_tone": "agreement|disagreement|hesitation|tension|neutral",
  "segments": [
    {
      "index": 0,
      "sentiment": "agreement|disagreement|hesitation|tension|neutral",
      "confidence": 0.0
    }
  ]
}
Rules:
- Classify each segment independently.
- Use agreement for alignment, approval, acceptance, and buy-in.
- Use disagreement for explicit pushback, rejection, or direct contradiction.
- Use hesitation for uncertainty, tentativeness, or lack of commitment.
- Use tension for conflict, urgency, risk escalation, or emotionally strained exchanges.
- Use neutral when no clear sentiment signal is present.
- confidence must be a float between 0 and 1.
"""

_AGREEMENT_PATTERNS = [
    r"\b(agree|agreed|yes|yep|ok|okay|works for me|sounds good|makes sense)\b",
    r"(同意|可以|没问题|好|行|赞成)",
]
_DISAGREEMENT_PATTERNS = [
    r"\b(disagree|don't agree|do not agree|no way|not acceptable|i don't think)\b",
    r"(不同意|不行|不对|不接受|不是这样|但是我不觉得)",
]
_HESITATION_PATTERNS = [
    r"\b(maybe|perhaps|might|not sure|i think|probably|possibly|let me check)\b",
    r"(可能|也许|不确定|先看看|我想|大概|再确认一下)",
]
_TENSION_PATTERNS = [
    r"\b(risk|blocked|blocking|urgent|deadline|issue|problem|conflict|late)\b",
    r"(风险|卡住|紧急|来不及|问题|冲突|压力|担心|拖期)",
]


class _LLMSentimentSegment(BaseModel):
    index: int
    sentiment: SentimentLabel
    confidence: float


class _LLMSentimentPayload(BaseModel):
    overall_tone: SentimentLabel
    segments: list[_LLMSentimentSegment] = Field(default_factory=list)


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


AGREEMENT_PATTERNS = _compile_patterns(_AGREEMENT_PATTERNS)
DISAGREEMENT_PATTERNS = _compile_patterns(_DISAGREEMENT_PATTERNS)
HESITATION_PATTERNS = _compile_patterns(_HESITATION_PATTERNS)
TENSION_PATTERNS = _compile_patterns(_TENSION_PATTERNS)


def _keyword_label(text: str) -> tuple[SentimentLabel, float] | None:
    if any(pattern.search(text) for pattern in TENSION_PATTERNS):
        return SentimentLabel.TENSION, 0.82
    if any(pattern.search(text) for pattern in DISAGREEMENT_PATTERNS):
        return SentimentLabel.DISAGREEMENT, 0.8
    if any(pattern.search(text) for pattern in HESITATION_PATTERNS):
        return SentimentLabel.HESITATION, 0.72
    if any(pattern.search(text) for pattern in AGREEMENT_PATTERNS):
        return SentimentLabel.AGREEMENT, 0.78
    return None


def _label_from_score_name(label: str, available_labels: set[str]) -> SentimentLabel | None:
    normalized = label.strip().lower()
    if normalized in {value.value for value in SentimentLabel}:
        return SentimentLabel(normalized)

    if normalized in {"positive", "very positive", "pos", "approval", "approve", "entailment"}:
        return SentimentLabel.AGREEMENT
    if normalized in {"negative", "very negative", "neg", "contradiction"}:
        return SentimentLabel.DISAGREEMENT
    if normalized in {"neutral", "objective"}:
        return SentimentLabel.NEUTRAL

    match = re.match(r"^(?P<stars>[1-5])\s*star[s]?$", normalized)
    if match:
        stars = int(match.group("stars"))
        if stars <= 2:
            return SentimentLabel.DISAGREEMENT
        if stars == 3:
            return SentimentLabel.NEUTRAL
        return SentimentLabel.AGREEMENT

    if available_labels == {"label_0", "label_1"}:
        return SentimentLabel.AGREEMENT if normalized == "label_1" else SentimentLabel.DISAGREEMENT

    return None


def _score_rows(output: Any) -> list[dict[str, Any]]:
    if isinstance(output, dict):
        return [output]
    if isinstance(output, list):
        if output and isinstance(output[0], dict):
            return output
    raise ValueError(f"Unsupported classifier output: {output!r}")


def _resolve_overall_tone(segments: list[SentimentSegment]) -> SentimentLabel:
    if not segments:
        return SentimentLabel.NEUTRAL

    if any(segment.sentiment == SentimentLabel.TENSION and segment.confidence >= 0.75 for segment in segments):
        return SentimentLabel.TENSION
    if any(segment.sentiment == SentimentLabel.DISAGREEMENT and segment.confidence >= 0.8 for segment in segments):
        return SentimentLabel.DISAGREEMENT

    weights = {label: 0.0 for label in SentimentLabel}
    for segment in segments:
        weights[segment.sentiment] += max(segment.confidence, 0.0)

    return max(
        weights,
        key=lambda label: (
            weights[label],
            1 if label != SentimentLabel.NEUTRAL else 0,
        ),
    )


def _coerce_sentiment_label(value: Any) -> SentimentLabel:
    try:
        return SentimentLabel(str(value).strip().lower())
    except Exception:
        return SentimentLabel.NEUTRAL


def _normalize_llm_payload(payload: Any, segment_count: int) -> _LLMSentimentPayload:
    raw_segments: list[dict[str, Any]] = []
    overall_tone: SentimentLabel | None = None

    if isinstance(payload, dict):
        if "overall_tone" in payload:
            overall_tone = _coerce_sentiment_label(payload.get("overall_tone"))
        if isinstance(payload.get("segments"), list):
            raw_segments = [item for item in payload["segments"] if isinstance(item, dict)]
        elif isinstance(payload.get("segments"), dict):
            raw_segments = [payload["segments"]]
        elif {"index", "sentiment", "confidence"}.intersection(payload.keys()):
            raw_segments = [payload]
    elif isinstance(payload, list):
        raw_segments = [item for item in payload if isinstance(item, dict)]

    segments: list[_LLMSentimentSegment] = []
    for position, item in enumerate(raw_segments):
        index = int(item.get("index", position))
        sentiment = _coerce_sentiment_label(item.get("sentiment"))
        confidence = float(item.get("confidence", 0.5))
        segments.append(
            _LLMSentimentSegment(
                index=index,
                sentiment=sentiment,
                confidence=round(min(max(confidence, 0.0), 1.0), 3),
            )
        )

    if not segments and segment_count > 0:
        segments = [
            _LLMSentimentSegment(
                index=index,
                sentiment=SentimentLabel.NEUTRAL,
                confidence=0.5,
            )
            for index in range(segment_count)
        ]

    if overall_tone is None:
        pseudo_segments = [
            SentimentSegment(
                text="",
                sentiment=segment.sentiment,
                confidence=segment.confidence,
            )
            for segment in segments
        ]
        overall_tone = _resolve_overall_tone(pseudo_segments)

    return _LLMSentimentPayload(overall_tone=overall_tone, segments=segments)


class TransformersSentimentClassifier:
    def __init__(self, settings: MeetingAISettings, classifier_pipeline: Any | None = None):
        self.settings = settings
        self._classifier_pipeline = classifier_pipeline

    def _load(self) -> Any:
        if self._classifier_pipeline is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            device = 0 if self.settings.device.startswith("cuda") else -1
            tokenizer = AutoTokenizer.from_pretrained(self.settings.sentiment_transformer_model)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.settings.sentiment_transformer_model,
                use_safetensors=True,
            )
            self._classifier_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=device,
                top_k=None,
                truncation=True,
            )
        return self._classifier_pipeline

    def classify_segments(self, segments: list[TranscriptSegment]) -> tuple[list[SentimentSegment], dict[str, Any]]:
        classifier = self._load()
        started = time.perf_counter()
        outputs = classifier([segment.text for segment in segments])
        elapsed = round(time.perf_counter() - started, 3)
        sentiment_segments: list[SentimentSegment] = []

        for segment, output in zip(segments, outputs):
            heuristic = _keyword_label(segment.text)
            rows = _score_rows(output)
            available_labels = {str(row.get("label", "")).strip().lower() for row in rows}

            scored_labels: list[tuple[SentimentLabel, float]] = []
            for row in rows:
                label_name = str(row.get("label", "")).strip()
                mapped = _label_from_score_name(label_name, available_labels)
                if mapped is None:
                    continue
                score = float(row.get("score", 0.0))
                scored_labels.append((mapped, score))

            if heuristic is not None:
                chosen_label, heuristic_score = heuristic
                model_score = max((score for label, score in scored_labels if label == chosen_label), default=0.0)
                final_label = chosen_label
                confidence = max(heuristic_score, model_score)
            elif scored_labels:
                final_label, confidence = max(scored_labels, key=lambda item: item[1])
            else:
                final_label, confidence = SentimentLabel.NEUTRAL, 0.5

            sentiment_segments.append(
                SentimentSegment(
                    text=segment.text,
                    sentiment=final_label,
                    confidence=round(min(max(confidence, 0.0), 1.0), 3),
                    speaker=segment.speaker,
                    start=segment.start,
                    end=segment.end,
                    metadata={"route": "transformer"},
                )
            )

        return sentiment_segments, {
            "classifier_model": self.settings.sentiment_transformer_model,
            "device": self.settings.device,
            "latency_seconds": elapsed,
        }


class SentimentAgent:
    def __init__(
        self,
        settings: MeetingAISettings | None = None,
        llm_client: UnifiedLLMClient | None = None,
        provider: LLMProvider = LLMProvider.DEEPSEEK,
        transformer_classifier: TransformersSentimentClassifier | None = None,
    ):
        self.settings = settings or get_settings()
        self.llm_client = llm_client or UnifiedLLMClient(self.settings)
        self.provider = provider
        self.transformer_classifier = transformer_classifier or TransformersSentimentClassifier(self.settings)

    def analyze(
        self,
        route: str,
        transcript: TranscriptResult | list[TranscriptSegment] | None = None,
        text: str | None = None,
    ) -> SentimentResult:
        if transcript is None and text is None:
            raise ValueError("Either transcript or text must be provided.")

        segments = transcript_to_segments(transcript) if transcript is not None else parse_labelled_lines(text or "")
        if route == "llm":
            return self._analyze_with_llm(segments)
        if route == "transformer":
            return self._analyze_with_transformer(segments)
        raise ValueError(f"Unsupported sentiment route: {route}")

    def _analyze_with_llm(self, segments: list[TranscriptSegment]) -> SentimentResult:
        prompt_lines = [
            f"{index}\t[{segment.speaker}] {segment.text}"
            for index, segment in enumerate(segments)
        ]
        response = self.llm_client.prompt(
            provider=self.provider,
            prompt=(
                "Classify the sentiment of each meeting segment and the overall meeting tone.\n"
                "Input segments:\n"
                f"{chr(10).join(prompt_lines)}"
            ),
            system_prompt=SENTIMENT_SYSTEM_PROMPT,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        payload = _normalize_llm_payload(extract_json_payload(response.content), len(segments))

        by_index = {entry.index: entry for entry in payload.segments}
        result_segments: list[SentimentSegment] = []
        for index, segment in enumerate(segments):
            classified = by_index.get(index)
            if classified is None:
                classified = _LLMSentimentSegment(
                    index=index,
                    sentiment=SentimentLabel.NEUTRAL,
                    confidence=0.5,
                )
            result_segments.append(
                SentimentSegment(
                    text=segment.text,
                    sentiment=classified.sentiment,
                    confidence=round(min(max(classified.confidence, 0.0), 1.0), 3),
                    speaker=segment.speaker,
                    start=segment.start,
                    end=segment.end,
                    metadata={"route": "llm"},
                )
            )

        return SentimentResult(
            route="llm",
            overall_tone=payload.overall_tone,
            segments=result_segments,
            metadata={
                "provider": self.provider.value,
                "latency_seconds": response.latency_seconds,
                "segment_count": len(result_segments),
            },
        )

    def _analyze_with_transformer(self, segments: list[TranscriptSegment]) -> SentimentResult:
        result_segments, metadata = self.transformer_classifier.classify_segments(segments)
        return SentimentResult(
            route="transformer",
            overall_tone=_resolve_overall_tone(result_segments),
            segments=result_segments,
            metadata={"segment_count": len(result_segments), **metadata},
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Week 2 sentiment agent.")
    parser.add_argument("--route", choices=["llm", "transformer"], default="llm")
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in LLMProvider],
        default=LLMProvider.DEEPSEEK.value,
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--transcript-json", help="Path to a transcript JSON file.")
    source_group.add_argument("--text", help="Inline transcript text.")
    source_group.add_argument("--text-file", help="Path to a UTF-8 text file.")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    provider = LLMProvider(args.provider)
    agent = SentimentAgent(settings=get_settings(), provider=provider)

    transcript = load_transcript_json(args.transcript_json) if args.transcript_json else None
    text = load_text_input(text=args.text, text_file=args.text_file) if not transcript else None
    result = agent.analyze(route=args.route, transcript=transcript, text=text)

    payload = result.model_dump_json(indent=2)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        LOGGER.info("Sentiment JSON saved to %s", output_path)

    print(payload)


if __name__ == "__main__":
    main()
