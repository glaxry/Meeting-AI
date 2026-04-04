from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import MeetingAISettings, get_settings
from .llm_tools import UnifiedLLMClient
from .schemas import LLMProvider, SummaryResult, TranscriptResult
from .structured_llm import prompt_json
from .text_utils import (
    chunk_text,
    chunk_transcript_segments,
    deduplicate_preserve_order,
    estimate_word_count,
    load_text_input,
    load_transcript_json,
    transcript_to_segments,
    transcript_to_text,
)


LOGGER = logging.getLogger(__name__)


SUMMARY_SYSTEM_PROMPT = """You are a meeting summary agent.
Return valid JSON only.
The output schema must be:
{
  "topics": ["..."],
  "decisions": ["..."],
  "follow_ups": ["..."]
}
Rules:
- Each value must be an array of concise strings.
- Do not invent facts that are not supported by the input.
- Put unresolved work, owners to follow up with, and next steps into follow_ups.
- If a field has no content, return an empty array.
"""


class SummaryAgent:
    def __init__(
        self,
        settings: MeetingAISettings | None = None,
        llm_client: UnifiedLLMClient | None = None,
        provider: LLMProvider = LLMProvider.DEEPSEEK,
    ):
        self.settings = settings or get_settings()
        self.llm_client = llm_client or UnifiedLLMClient(self.settings)
        self.provider = provider

    def summarize(
        self,
        transcript: TranscriptResult | list | None = None,
        text: str | None = None,
    ) -> SummaryResult:
        if transcript is None and text is None:
            raise ValueError("Either transcript or text must be provided.")

        if transcript is not None:
            segments = transcript_to_segments(transcript)
            source_text = transcript_to_text(segments)
            chunks = [
                transcript_to_text(chunk)
                for chunk in chunk_transcript_segments(
                    segments,
                    target_words=self.settings.summary_chunk_target_words,
                )
            ]
        else:
            source_text = text or ""
            chunks = chunk_text(source_text, target_words=self.settings.summary_chunk_target_words)

        word_count = estimate_word_count(source_text)
        use_map_reduce = word_count >= self.settings.summary_map_reduce_threshold and len(chunks) > 1

        if use_map_reduce:
            map_results: list[SummaryResult] = []
            map_latencies: list[float] = []
            for index, chunk_text_value in enumerate(chunks, start=1):
                prompt = (
                    f"Summarize meeting chunk {index}/{len(chunks)}.\n"
                    "Focus on concrete topics, explicit decisions, and follow-up work.\n\n"
                    f"{chunk_text_value}"
                )
                partial, response = prompt_json(
                    llm_client=self.llm_client,
                    provider=self.provider,
                    schema=SummaryResult,
                    prompt=prompt,
                    system_prompt=SUMMARY_SYSTEM_PROMPT,
                    temperature=0.1,
                )
                map_results.append(partial)
                map_latencies.append(response.latency_seconds)

            reduce_payload = {
                "partials": [partial.model_dump(exclude={"metadata"}) for partial in map_results],
            }
            reduced, reduce_response = prompt_json(
                llm_client=self.llm_client,
                provider=self.provider,
                schema=SummaryResult,
                prompt=(
                    "Merge the partial meeting summaries into a final meeting summary.\n"
                    "Deduplicate overlapping content and keep the output concise.\n\n"
                    f"{json.dumps(reduce_payload, ensure_ascii=False, indent=2)}"
                ),
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                temperature=0.1,
            )
            result = reduced
            metadata = {
                "strategy": "map_reduce",
                "word_count": word_count,
                "chunk_count": len(chunks),
                "provider": self.provider.value,
                "map_latencies": map_latencies,
                "reduce_latency": reduce_response.latency_seconds,
            }
        else:
            result, response = prompt_json(
                llm_client=self.llm_client,
                provider=self.provider,
                schema=SummaryResult,
                prompt=(
                    "Summarize this meeting transcript.\n"
                    "Capture discussion topics, explicit decisions, and follow-up actions.\n\n"
                    f"{source_text}"
                ),
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                temperature=0.1,
            )
            metadata = {
                "strategy": "single_pass",
                "word_count": word_count,
                "chunk_count": 1 if source_text.strip() else 0,
                "provider": self.provider.value,
                "latency_seconds": response.latency_seconds,
            }

        return SummaryResult(
            topics=deduplicate_preserve_order(result.topics),
            decisions=deduplicate_preserve_order(result.decisions),
            follow_ups=deduplicate_preserve_order(result.follow_ups),
            metadata=metadata,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Week 2 summary agent.")
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
    agent = SummaryAgent(settings=get_settings(), provider=provider)

    transcript = load_transcript_json(args.transcript_json) if args.transcript_json else None
    text = load_text_input(text=args.text, text_file=args.text_file) if not transcript else None
    result = agent.summarize(transcript=transcript, text=text)

    payload = result.model_dump_json(indent=2)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        LOGGER.info("Summary JSON saved to %s", output_path)

    print(payload)


if __name__ == "__main__":
    main()
