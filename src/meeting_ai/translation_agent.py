from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from .config import MeetingAISettings, get_settings
from .llm_tools import UnifiedLLMClient
from .schemas import LLMProvider, TranscriptSegment, TranslationResult, TranscriptResult
from .structured_llm import prompt_json
from .text_utils import (
    chunk_text,
    chunk_transcript_segments,
    load_text_input,
    load_transcript_json,
    parse_labelled_lines,
    transcript_to_segments,
)


LOGGER = logging.getLogger(__name__)


TRANSLATION_SYSTEM_PROMPT = """You are a meeting translation agent.
Return valid JSON only.
The output schema must be:
{
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "text": "translated text"
    }
  ]
}
Rules:
- Preserve segment order exactly.
- Preserve each speaker label exactly as provided.
- Translate only the spoken content, not the speaker label.
- Keep specialized meeting terminology consistent with the glossary when provided.
"""


class _TranslatedSegment(BaseModel):
    speaker: str
    text: str


class _TranslationPayload(BaseModel):
    segments: list[_TranslatedSegment] = Field(default_factory=list)


def _format_glossary(glossary: dict[str, str] | None) -> str:
    if not glossary:
        return "No glossary provided."
    return "\n".join(f"- {source} => {target}" for source, target in glossary.items())


class TranslationAgent:
    def __init__(
        self,
        settings: MeetingAISettings | None = None,
        llm_client: UnifiedLLMClient | None = None,
        provider: LLMProvider = LLMProvider.DEEPSEEK,
    ):
        self.settings = settings or get_settings()
        self.llm_client = llm_client or UnifiedLLMClient(self.settings)
        self.provider = provider

    def translate(
        self,
        source_language: str,
        target_language: str,
        transcript: TranscriptResult | list[TranscriptSegment] | None = None,
        text: str | None = None,
        glossary: dict[str, str] | None = None,
    ) -> TranslationResult:
        if transcript is None and text is None:
            raise ValueError("Either transcript or text must be provided.")

        base_segments = transcript_to_segments(transcript) if transcript is not None else parse_labelled_lines(text or "")
        if not base_segments:
            return TranslationResult(
                source_language=source_language,
                target_language=target_language,
                segments=[],
                full_text="",
                metadata={
                    "provider": self.provider.value,
                    "chunk_count": 0,
                    "glossary_size": len(glossary or {}),
                },
            )

        translated_segments: list[TranscriptSegment] = []
        latencies: list[float] = []
        chunks = chunk_transcript_segments(base_segments, target_words=self.settings.summary_chunk_target_words)

        for chunk in chunks:
            prompt_lines = []
            for index, segment in enumerate(chunk):
                prompt_lines.append(f"{index}\t[{segment.speaker}] {segment.text}")

            payload, response = prompt_json(
                llm_client=self.llm_client,
                provider=self.provider,
                schema=_TranslationPayload,
                prompt=(
                    f"Translate the transcript from {source_language} to {target_language}.\n"
                    "Glossary:\n"
                    f"{_format_glossary(glossary)}\n\n"
                    "Input segments:\n"
                    f"{chr(10).join(prompt_lines)}"
                ),
                system_prompt=TRANSLATION_SYSTEM_PROMPT,
                temperature=0.1,
            )
            if len(payload.segments) != len(chunk):
                raise ValueError(
                    f"Translation output segment count mismatch. Expected {len(chunk)}, got {len(payload.segments)}."
                )

            latencies.append(response.latency_seconds)
            for original, translated in zip(chunk, payload.segments):
                translated_segments.append(
                    original.model_copy(
                        update={
                            "speaker": translated.speaker,
                            "text": translated.text.strip(),
                            "raw_text": original.text,
                        }
                    )
                )

        full_text = "\n".join(f"[{segment.speaker}] {segment.text}" for segment in translated_segments)
        return TranslationResult(
            source_language=source_language,
            target_language=target_language,
            segments=translated_segments,
            full_text=full_text,
            metadata={
                "provider": self.provider.value,
                "chunk_count": len(chunks),
                "glossary_size": len(glossary or {}),
                "latencies": latencies,
            },
        )


def _load_glossary(glossary_arg: list[str] | None) -> dict[str, str]:
    glossary: dict[str, str] = {}
    for entry in glossary_arg or []:
        if "=" not in entry:
            raise ValueError(f"Invalid glossary entry: {entry}. Expected SOURCE=TARGET.")
        source, target = entry.split("=", 1)
        glossary[source.strip()] = target.strip()
    return glossary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Week 2 translation agent.")
    parser.add_argument("--source-language", required=True)
    parser.add_argument("--target-language", required=True)
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in LLMProvider],
        default=LLMProvider.DEEPSEEK.value,
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--transcript-json", help="Path to a transcript JSON file.")
    source_group.add_argument("--text", help="Inline transcript text.")
    source_group.add_argument("--text-file", help="Path to a UTF-8 text file.")
    parser.add_argument("--glossary", action="append", help="Glossary item in SOURCE=TARGET format.")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    provider = LLMProvider(args.provider)
    agent = TranslationAgent(settings=get_settings(), provider=provider)

    transcript = load_transcript_json(args.transcript_json) if args.transcript_json else None
    text = load_text_input(text=args.text, text_file=args.text_file) if not transcript else None
    glossary = _load_glossary(args.glossary)

    result = agent.translate(
        source_language=args.source_language,
        target_language=args.target_language,
        transcript=transcript,
        text=text,
        glossary=glossary,
    )

    payload = result.model_dump_json(indent=2)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        LOGGER.info("Translation JSON saved to %s", output_path)

    print(payload)


if __name__ == "__main__":
    main()
