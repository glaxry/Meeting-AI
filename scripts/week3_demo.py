from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meeting_ai.orchestrator import MeetingOrchestrator
from meeting_ai.schemas import LLMProvider


def parse_glossary(entries: list[str] | None) -> dict[str, str]:
    glossary: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Invalid glossary entry: {entry}. Expected SOURCE=TARGET.")
        source, target = entry.split("=", 1)
        glossary[source.strip()] = target.strip()
    return glossary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Week 3 orchestrator demo.")
    parser.add_argument("--audio", required=True, help="Path to an audio file.")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--provider", choices=[provider.value for provider in LLMProvider], default=LLMProvider.DEEPSEEK.value)
    parser.add_argument("--target-language", default="en")
    parser.add_argument("--agent", action="append", dest="agents", help="Agent to run. Repeatable.")
    parser.add_argument("--glossary", action="append", help="Glossary item in SOURCE=TARGET format.")
    parser.add_argument("--sentiment-route", choices=["llm", "transformer"], default="llm")
    parser.add_argument("--history-query", help="Optional retrieval query.")
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--disable-diarization", action="store_true")
    parser.add_argument("--enable-voiceprint", action="store_true")
    parser.add_argument("--output", help="Optional output JSON path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    orchestrator = MeetingOrchestrator()
    result = orchestrator.run(
        audio_path=args.audio,
        language=args.language,
        provider=LLMProvider(args.provider),
        selected_agents=args.agents,
        target_language=args.target_language,
        glossary=parse_glossary(args.glossary),
        sentiment_route=args.sentiment_route,
        history_query=args.history_query,
        use_diarization=not args.disable_diarization,
        num_speakers=args.num_speakers,
        enable_voiceprint=args.enable_voiceprint,
    )
    payload = result.model_dump_json(indent=2)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
