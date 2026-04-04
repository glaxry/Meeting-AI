from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import MeetingAISettings, get_settings
from .llm_tools import UnifiedLLMClient
from .schemas import ActionItem, ActionItemResult, LLMProvider, TranscriptResult
from .structured_llm import prompt_json
from .text_utils import (
    chunk_text,
    chunk_transcript_segments,
    load_text_input,
    load_transcript_json,
    transcript_to_segments,
    transcript_to_text,
)


LOGGER = logging.getLogger(__name__)


ACTION_ITEM_SYSTEM_PROMPT = """You are an action item extraction agent.
Return valid JSON only.
The output schema must be:
{
  "items": [
    {
      "assignee": "name or null",
      "task": "concrete task",
      "deadline": "deadline or null",
      "priority": "low|medium|high",
      "source_quote": "short supporting quote from the transcript"
    }
  ]
}
Rules:
- Extract explicit tasks and implied follow-up work.
- When the transcript implies ownership without a formal imperative, still create an item.
- Keep task text concrete and actionable.
- If no assignee is clear, use null.
- If no deadline is clear, use null.
- Use source_quote as a short verbatim quote supporting the item.
"""


def _deduplicate_items(items: list[ActionItem]) -> list[ActionItem]:
    deduplicated: list[ActionItem] = []
    seen: set[tuple[str | None, str, str | None]] = set()

    for item in items:
        key = (
            item.assignee.strip().lower() if item.assignee else None,
            item.task.strip().lower(),
            item.deadline.strip().lower() if item.deadline else None,
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)

    return deduplicated


class ActionItemAgent:
    def __init__(
        self,
        settings: MeetingAISettings | None = None,
        llm_client: UnifiedLLMClient | None = None,
        provider: LLMProvider = LLMProvider.DEEPSEEK,
    ):
        self.settings = settings or get_settings()
        self.llm_client = llm_client or UnifiedLLMClient(self.settings)
        self.provider = provider

    def extract(
        self,
        transcript: TranscriptResult | list | None = None,
        text: str | None = None,
    ) -> ActionItemResult:
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

        extracted_items: list[ActionItem] = []
        latencies: list[float] = []

        for index, chunk_text_value in enumerate(chunks or [source_text], start=1):
            result, response = prompt_json(
                llm_client=self.llm_client,
                provider=self.provider,
                schema=ActionItemResult,
                prompt=(
                    f"Extract action items from meeting chunk {index}/{max(len(chunks), 1)}.\n"
                    "Pay attention to implied ownership and follow-up obligations.\n\n"
                    "Example of an implied task:\n"
                    '[SPEAKER_A] This one you sync with the vendor tomorrow.\n'
                    "Should become an action item assigned to SPEAKER_A or the named person if the transcript identifies them.\n\n"
                    f"{chunk_text_value}"
                ),
                system_prompt=ACTION_ITEM_SYSTEM_PROMPT,
                temperature=0.1,
            )
            extracted_items.extend(result.items)
            latencies.append(response.latency_seconds)

        items = _deduplicate_items(extracted_items)
        strategy = "chunked" if len(chunks) > 1 else "single_pass"
        return ActionItemResult(
            items=items,
            metadata={
                "provider": self.provider.value,
                "strategy": strategy,
                "chunk_count": len(chunks) or 1,
                "latencies": latencies,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Week 2 action item agent.")
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
    agent = ActionItemAgent(settings=get_settings(), provider=provider)

    transcript = load_transcript_json(args.transcript_json) if args.transcript_json else None
    text = load_text_input(text=args.text, text_file=args.text_file) if not transcript else None
    result = agent.extract(transcript=transcript, text=text)

    payload = result.model_dump_json(indent=2)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        LOGGER.info("Action items JSON saved to %s", output_path)

    print(payload)


if __name__ == "__main__":
    main()
