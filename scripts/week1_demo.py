from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meeting_ai.asr_agent import MeetingASRAgent
from meeting_ai.config import get_settings
from meeting_ai.llm_tools import UnifiedLLMClient
from meeting_ai.schemas import LLMProvider, TranscriptResult


def build_demo_prompt(transcript: TranscriptResult) -> str:
    return (
        "你是一名会议助手。请根据下面的会议转录，输出一个简短总结，包含：\n"
        "1. 会议主题\n"
        "2. 关键决定\n"
        "3. 后续动作\n\n"
        f"{transcript.full_text}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Week 1 end-to-end demo.")
    parser.add_argument("--audio", required=True, help="Path to an audio file.")
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in LLMProvider],
        default=LLMProvider.DEEPSEEK.value,
    )
    parser.add_argument("--language", default="zh")
    parser.add_argument("--disable-diarization", action="store_true")
    parser.add_argument("--num-speakers", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    output_dir = settings.ensure_output_dir()

    asr_agent = MeetingASRAgent(settings=settings)
    transcript = asr_agent.transcribe(
        audio_path=args.audio,
        language=args.language,
        use_diarization=not args.disable_diarization,
        num_speakers=args.num_speakers,
    )

    transcript_path = output_dir / "week1_transcript.json"
    transcript_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")

    llm_client = UnifiedLLMClient(settings=settings)
    llm_response = llm_client.prompt(
        provider=LLMProvider(args.provider),
        prompt=build_demo_prompt(transcript),
        system_prompt="你是严谨的会议纪要助手，输出简洁明确。",
    )

    summary_path = output_dir / "week1_llm_summary.md"
    summary_path.write_text(llm_response.content, encoding="utf-8")

    print(f"Transcript JSON: {transcript_path}")
    print(f"LLM summary: {summary_path}")


if __name__ == "__main__":
    main()

