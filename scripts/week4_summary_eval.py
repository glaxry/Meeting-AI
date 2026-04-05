from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meeting_ai.config import get_settings
from meeting_ai.evaluation import compute_rouge, load_jsonl, resolve_manifest_path, summary_to_eval_text
from meeting_ai.llm_tools import UnifiedLLMClient
from meeting_ai.schemas import LLMProvider, SummaryResult
from meeting_ai.structured_llm import prompt_json
from meeting_ai.summary_agent import SummaryAgent


JUDGE_SYSTEM_PROMPT = """You are evaluating a generated meeting summary against a reference summary.
Return valid JSON only.
The output schema must be:
{
  "overall": 1,
  "faithfulness": 1,
  "coverage": 1,
  "conciseness": 1,
  "justification": "short explanation"
}
Rules:
- Scores must be integers from 1 to 5.
- faithfulness measures factual consistency with the transcript.
- coverage measures whether key topics, decisions, and follow-ups are captured.
- conciseness measures whether the summary is compact without losing important content.
- overall should reflect the overall usefulness of the candidate summary.
"""


class SummaryJudgeResult(BaseModel):
    overall: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    coverage: int = Field(ge=1, le=5)
    conciseness: int = Field(ge=1, le=5)
    justification: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Week 4 summary evaluation.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "data" / "eval" / "summary_manifest.sample.jsonl"),
        help="JSONL manifest with transcript_file or transcript_text and reference_summary.",
    )
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in LLMProvider],
        default=LLMProvider.DEEPSEEK.value,
    )
    parser.add_argument(
        "--judge-provider",
        choices=[provider.value for provider in LLMProvider],
        help="Optional LLM provider to score generated summaries.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "week4" / "summary_eval.json"),
        help="Path to the JSON output file.",
    )
    return parser


def _load_transcript_text(manifest_path: Path, row: dict[str, object]) -> str:
    transcript_text = row.get("transcript_text")
    if transcript_text:
        return str(transcript_text)
    transcript_file = row.get("transcript_file")
    if transcript_file:
        return resolve_manifest_path(manifest_path, str(transcript_file)).read_text(encoding="utf-8")
    raise ValueError("Each summary sample must provide transcript_text or transcript_file.")


def _run_judge(
    llm_client: UnifiedLLMClient,
    provider: LLMProvider,
    transcript_text: str,
    reference_summary: dict[str, object],
    candidate_summary: SummaryResult,
) -> tuple[SummaryJudgeResult, float]:
    result, response = prompt_json(
        llm_client=llm_client,
        provider=provider,
        schema=SummaryJudgeResult,
        prompt=(
            "Evaluate the candidate meeting summary.\n\n"
            f"Transcript:\n{transcript_text}\n\n"
            f"Reference summary:\n{summary_to_eval_text(reference_summary)}\n\n"
            f"Candidate summary:\n{summary_to_eval_text(candidate_summary)}"
        ),
        system_prompt=JUDGE_SYSTEM_PROMPT,
        temperature=0.0,
    )
    return result, response.latency_seconds


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    provider = LLMProvider(args.provider)
    judge_provider = LLMProvider(args.judge_provider) if args.judge_provider else None
    settings = get_settings()
    llm_client = UnifiedLLMClient(settings)
    rows = load_jsonl(manifest_path)

    strategies = {
        "default": settings,
        "single_pass": settings.model_copy(update={"summary_map_reduce_threshold": 10**9}),
    }
    results_by_strategy: dict[str, dict[str, object]] = {}

    for strategy_name, strategy_settings in strategies.items():
        agent = SummaryAgent(settings=strategy_settings, llm_client=llm_client, provider=provider)
        sample_results: list[dict[str, object]] = []
        for row in rows:
            transcript_text = _load_transcript_text(manifest_path, row)
            reference_summary = row["reference_summary"]
            candidate = agent.summarize(text=transcript_text)
            candidate_text = summary_to_eval_text(candidate)
            reference_text = summary_to_eval_text(reference_summary)
            rouge_scores = compute_rouge(reference_text, candidate_text)
            sample_result: dict[str, object] = {
                "id": row["id"],
                "language": row.get("language", "unknown"),
                "word_count": candidate.metadata.get("word_count"),
                "actual_strategy": candidate.metadata.get("strategy"),
                "chunk_count": candidate.metadata.get("chunk_count"),
                "reference_summary": reference_summary,
                "candidate_summary": candidate.model_dump(exclude={"metadata"}),
                "rouge": rouge_scores,
            }
            if judge_provider is not None:
                judge_result, judge_latency = _run_judge(
                    llm_client=llm_client,
                    provider=judge_provider,
                    transcript_text=transcript_text,
                    reference_summary=reference_summary,
                    candidate_summary=candidate,
                )
                sample_result["judge"] = judge_result.model_dump()
                sample_result["judge_latency_seconds"] = judge_latency
            sample_results.append(sample_result)

        results_by_strategy[strategy_name] = {
            "strategy": strategy_name,
            "provider": provider.value,
            "sample_count": len(sample_results),
            "mean_rouge1": round(sum(item["rouge"]["rouge1"] for item in sample_results) / len(sample_results), 6),
            "mean_rouge2": round(sum(item["rouge"]["rouge2"] for item in sample_results) / len(sample_results), 6),
            "mean_rougeL": round(sum(item["rouge"]["rougeL"] for item in sample_results) / len(sample_results), 6),
            "mean_judge_overall": (
                round(sum(item["judge"]["overall"] for item in sample_results if "judge" in item) / len(sample_results), 6)
                if judge_provider is not None
                else None
            ),
            "samples": sample_results,
        }

    output = {
        "manifest": str(manifest_path),
        "provider": provider.value,
        "judge_provider": judge_provider.value if judge_provider else None,
        "strategies": results_by_strategy,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
