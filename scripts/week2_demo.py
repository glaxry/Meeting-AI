from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meeting_ai.action_item_agent import ActionItemAgent
from meeting_ai.config import get_settings
from meeting_ai.schemas import LLMProvider
from meeting_ai.sentiment_agent import SentimentAgent
from meeting_ai.summary_agent import SummaryAgent
from meeting_ai.text_utils import load_transcript_json
from meeting_ai.translation_agent import TranslationAgent


def parse_glossary(entries: list[str] | None) -> dict[str, str]:
    glossary: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Invalid glossary entry: {entry}. Expected SOURCE=TARGET.")
        source, target = entry.split("=", 1)
        glossary[source.strip()] = target.strip()
    return glossary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all Week 2 agents on one transcript JSON file.")
    parser.add_argument("--transcript-json", required=True, help="Path to an existing transcript JSON file.")
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in LLMProvider],
        default=LLMProvider.DEEPSEEK.value,
    )
    parser.add_argument("--source-language", default="zh")
    parser.add_argument("--target-language", default="en")
    parser.add_argument("--translation-glossary", action="append", help="Glossary item in SOURCE=TARGET format.")
    parser.add_argument("--sentiment-route", choices=["llm", "transformer"], default="llm")
    parser.add_argument("--output-dir", help="Optional output directory. Defaults to data/outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    provider = LLMProvider(args.provider)
    transcript = load_transcript_json(args.transcript_json)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else settings.ensure_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    glossary = parse_glossary(args.translation_glossary)

    summary_agent = SummaryAgent(settings=settings, provider=provider)
    translation_agent = TranslationAgent(settings=settings, provider=provider)
    action_item_agent = ActionItemAgent(settings=settings, provider=provider)
    sentiment_agent = SentimentAgent(settings=settings, provider=provider)

    summary = summary_agent.summarize(transcript=transcript)
    translation = translation_agent.translate(
        source_language=args.source_language,
        target_language=args.target_language,
        transcript=transcript,
        glossary=glossary,
    )
    action_items = action_item_agent.extract(transcript=transcript)
    sentiment = sentiment_agent.analyze(route=args.sentiment_route, transcript=transcript)

    outputs = {
        "week2_summary.json": summary.model_dump(),
        "week2_translation.json": translation.model_dump(),
        "week2_action_items.json": action_items.model_dump(),
        f"week2_sentiment_{args.sentiment_route}.json": sentiment.model_dump(),
    }

    for filename, payload in outputs.items():
        path = output_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
