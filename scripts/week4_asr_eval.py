from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meeting_ai.asr_agent import MeetingASRAgent
from meeting_ai.config import get_settings
from meeting_ai.evaluation import compute_error_rates, load_jsonl, resolve_manifest_path


MODEL_ALIASES = {
    "sensevoicesmall": "iic/SenseVoiceSmall",
    "sensevoice": "iic/SenseVoiceSmall",
}


def resolve_model_name(value: str) -> str:
    normalized = value.strip()
    return MODEL_ALIASES.get(normalized.lower(), normalized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Week 4 ASR evaluation.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "data" / "eval" / "asr_manifest.sample.jsonl"),
        help="JSONL manifest with audio_path, language, and reference_text.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="FunASR model to evaluate. Repeatable. Defaults to paraformer-zh and iic/SenseVoiceSmall.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "week4" / "asr_eval.json"),
        help="Path to the JSON output file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    models = [resolve_model_name(value) for value in (args.models or ["paraformer-zh", "iic/SenseVoiceSmall"])]
    settings = get_settings()
    rows = load_jsonl(manifest_path)

    aggregate: dict[str, dict[str, object]] = {}
    for model_name in models:
        agent = MeetingASRAgent(settings=settings.model_copy(update={"funasr_model": model_name}))
        per_sample: list[dict[str, object]] = []

        for row in rows:
            audio_path = resolve_manifest_path(manifest_path, row["audio_path"])
            try:
                transcript = agent.transcribe(
                    audio_path=audio_path,
                    language=row.get("language", "zh"),
                    use_diarization=False,
                )
                hypothesis_text = " ".join(segment.text for segment in transcript.segments)
                metrics = compute_error_rates(row["reference_text"], hypothesis_text)
                audio_duration = float(transcript.metadata.get("audio_duration_seconds") or 0.0)
                asr_runtime = float(transcript.metadata.get("asr_runtime_seconds") or 0.0)
                per_sample.append(
                    {
                        "id": row["id"],
                        "audio_path": str(audio_path),
                        "reference_text": row["reference_text"],
                        "hypothesis_text": hypothesis_text,
                        "wer": metrics["wer"],
                        "cer": metrics["cer"],
                        "audio_duration_seconds": audio_duration,
                        "asr_runtime_seconds": asr_runtime,
                        "rtf": round(asr_runtime / audio_duration, 6) if audio_duration else None,
                        "warnings": transcript.metadata.get("warnings", []),
                    }
                )
            except Exception as exc:
                per_sample.append(
                    {
                        "id": row["id"],
                        "audio_path": str(audio_path),
                        "error": str(exc),
                    }
                )

        successful = [item for item in per_sample if "error" not in item]
        aggregate[model_name] = {
            "model": model_name,
            "sample_count": len(per_sample),
            "success_count": len(successful),
            "mean_wer": round(sum(item["wer"] for item in successful) / len(successful), 6) if successful else None,
            "mean_cer": round(sum(item["cer"] for item in successful) / len(successful), 6) if successful else None,
            "mean_rtf": round(sum(item["rtf"] for item in successful if item["rtf"] is not None) / len(successful), 6) if successful else None,
            "samples": per_sample,
        }

    output = {
        "manifest": str(manifest_path),
        "models": aggregate,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
