from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torchaudio
import torch
from funasr import AutoModel
from pyannote.audio import Pipeline

from .config import MeetingAISettings, get_settings
from .schemas import DiarizationSegment, TranscriptResult, TranscriptSegment
from .voiceprint import VoiceprintIdentifier, apply_voiceprint_identities


LOGGER = logging.getLogger(__name__)
_CONTROL_TOKEN_PATTERN = re.compile(r"<\|[^|]+?\|>")
_SENSEVOICE_EMOTION_TOKENS = {
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "surprised",
    "disgusted",
}
_SENSEVOICE_EVENT_TOKENS = {
    "speech",
    "laughter",
    "applause",
    "crying",
    "cough",
    "sneeze",
    "music",
    "noise",
}
MIN_OVERLAP_RATIO = 0.1


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def get_audio_duration(audio_path: Path) -> float:
    return float(sf.info(str(audio_path)).duration)


def _uses_sensevoice_model(model_name: str) -> bool:
    return "sensevoice" in model_name.strip().lower()


def _sensevoice_language_hint(language: str) -> str:
    normalized = language.strip().lower()
    if normalized in {"zh", "zh-cn", "zh_cn", "cn", "中文", "chinese"}:
        return "zn"
    if normalized in {"en", "english"}:
        return "en"
    return "auto"


def _funasr_model_kwargs(settings: MeetingAISettings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": settings.funasr_model,
        "device": settings.device,
        "hub": settings.funasr_hub,
        "disable_update": True,
    }

    if _uses_sensevoice_model(settings.funasr_model):
        kwargs["vad_model"] = settings.funasr_vad_model
        kwargs["vad_kwargs"] = {"max_single_segment_time": 30_000}
        kwargs["trust_remote_code"] = True
    else:
        kwargs["vad_model"] = settings.funasr_vad_model
        kwargs["punc_model"] = settings.funasr_punc_model

    return kwargs


def _funasr_generate_kwargs(settings: MeetingAISettings, language: str) -> dict[str, Any]:
    if _uses_sensevoice_model(settings.funasr_model):
        return {
            "input": None,
            "cache": {},
            "language": _sensevoice_language_hint(language),
            "use_itn": False,
            "batch_size_s": 0,
        }
    return {
        "input": None,
        "sentence_timestamp": True,
        "return_raw_text": True,
        "en_post_proc": language.lower().startswith("en"),
    }


def _coerce_seconds(value: Any, audio_duration: float | None) -> float:
    seconds = float(value)
    if audio_duration is not None and seconds > (audio_duration + 5):
        seconds = seconds / 1000.0
    return round(max(seconds, 0.0), 3)


def _clean_transcript_text(text: str) -> str:
    normalized = _CONTROL_TOKEN_PATTERN.sub("", text)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _control_tokens(text: str) -> list[str]:
    return [match[2:-2].strip().lower() for match in _CONTROL_TOKEN_PATTERN.findall(text or "")]


def _normalize_optional_tag(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _extract_sensevoice_annotations(item: dict[str, Any]) -> tuple[str | None, str | None]:
    emotion = _normalize_optional_tag(item.get("emotion")) or _normalize_optional_tag(item.get("emo"))
    event = _normalize_optional_tag(item.get("event")) or _normalize_optional_tag(item.get("audio_event"))

    token_candidates: list[str] = []
    token_candidates.extend(_control_tokens(str(item.get("raw_text", ""))))
    token_candidates.extend(_control_tokens(str(item.get("text", ""))))

    if emotion is None:
        emotion = next((token for token in token_candidates if token in _SENSEVOICE_EMOTION_TOKENS), None)
    if event is None:
        event = next((token for token in token_candidates if token in _SENSEVOICE_EVENT_TOKENS), None)
    return emotion, event


def normalize_sentence_info(sentence_info: list[dict[str, Any]], audio_duration: float | None) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for item in sentence_info:
        text = _clean_transcript_text(str(item.get("text", "")))
        if not text:
            continue
        emotion, event = _extract_sensevoice_annotations(item)
        segments.append(
            TranscriptSegment(
                speaker="SPEAKER_00",
                text=text,
                start=_coerce_seconds(item.get("start", 0.0), audio_duration),
                end=_coerce_seconds(item.get("end", 0.0), audio_duration),
                raw_text=_clean_transcript_text(str(item.get("raw_text", "")).strip()) or None,
                emotion=emotion,
                event=event,
            )
        )
    return sorted(segments, key=lambda seg: (seg.start, seg.end))


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(
    transcript_segments: list[TranscriptSegment],
    diarization_segments: list[DiarizationSegment],
    min_overlap_ratio: float = MIN_OVERLAP_RATIO,
) -> list[TranscriptSegment]:
    if not diarization_segments:
        return transcript_segments

    assigned: list[TranscriptSegment] = []
    for segment in transcript_segments:
        duration = max(segment.end - segment.start, 1e-6)
        best_speaker = diarization_segments[0].speaker
        best_overlap = 0.0
        for diarized in diarization_segments:
            overlap = _overlap(segment.start, segment.end, diarized.start, diarized.end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diarized.speaker

        best_ratio = round(best_overlap / duration, 3)
        metadata = dict(segment.metadata)
        if best_ratio < min_overlap_ratio:
            center = (segment.start + segment.end) / 2
            nearest = min(
                diarization_segments,
                key=lambda item: abs(((item.start + item.end) / 2) - center),
            )
            best_speaker = nearest.speaker
            metadata.update(
                {
                    "speaker_confidence": "low",
                    "overlap_ratio": best_ratio,
                    "assignment_strategy": "nearest_segment",
                }
            )
        else:
            metadata.update(
                {
                    "speaker_confidence": "high",
                    "overlap_ratio": best_ratio,
                    "assignment_strategy": "overlap_match",
                }
            )

        assigned.append(segment.model_copy(update={"speaker": best_speaker, "metadata": metadata}))

    return assigned


class FunASRTranscriber:
    def __init__(self, settings: MeetingAISettings):
        self.settings = settings
        self._model: AutoModel | None = None

    def _load(self) -> AutoModel:
        if self._model is None:
            LOGGER.info(
                "Loading FunASR model=%s vad=%s punc=%s on %s",
                self.settings.funasr_model,
                self.settings.funasr_vad_model,
                self.settings.funasr_punc_model,
                self.settings.device,
            )
            self._model = AutoModel(**_funasr_model_kwargs(self.settings))
        return self._model

    def transcribe(self, audio_path: Path, language: str, audio_duration: float) -> tuple[list[TranscriptSegment], dict[str, Any]]:
        model = self._load()
        started = time.perf_counter()
        generate_kwargs = _funasr_generate_kwargs(self.settings, language)
        generate_kwargs["input"] = str(audio_path)
        result = model.generate(**generate_kwargs)
        elapsed = round(time.perf_counter() - started, 3)
        payload = result[0] if result else {}
        sentence_info = payload.get("sentence_info") or []
        segments = normalize_sentence_info(sentence_info, audio_duration)

        if not segments and payload.get("text"):
            emotion, event = _extract_sensevoice_annotations(payload)
            segments = [
                TranscriptSegment(
                    speaker="SPEAKER_00",
                    text=_clean_transcript_text(str(payload["text"])),
                    start=0.0,
                    end=round(audio_duration, 3),
                    raw_text=_clean_transcript_text(str(payload.get("raw_text", "")).strip()) or None,
                    emotion=emotion,
                    event=event,
                )
            ]

        metadata = {
            "asr_runtime_seconds": elapsed,
            "sentence_count": len(segments),
            "raw_result_keys": sorted(payload.keys()),
            "sensevoice_enrichment": _uses_sensevoice_model(self.settings.funasr_model),
            "emotion_segment_count": sum(1 for segment in segments if segment.emotion),
            "event_segment_count": sum(1 for segment in segments if segment.event),
        }
        return segments, metadata


class PyannoteDiarizer:
    def __init__(self, settings: MeetingAISettings):
        self.settings = settings
        self._pipeline: Pipeline | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings.huggingface_token)

    def _load(self) -> Pipeline:
        if not self.enabled:
            raise RuntimeError("HUGGINGFACE_TOKEN is not configured.")
        if self._pipeline is None:
            LOGGER.info("Loading pyannote pipeline=%s on %s", self.settings.pyannote_model, self.settings.device)
            self._pipeline = Pipeline.from_pretrained(
                self.settings.pyannote_model,
                use_auth_token=self.settings.huggingface_token,
            )
            if self.settings.device.startswith("cuda"):
                self._pipeline.to(torch.device("cuda"))
        return self._pipeline

    def diarize(
        self,
        audio_path: Path,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> tuple[list[DiarizationSegment], dict[str, Any]]:
        pipeline = self._load()
        waveform, sample_rate = torchaudio.load(str(audio_path))
        kwargs: dict[str, int] = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        started = time.perf_counter()
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
        elapsed = round(time.perf_counter() - started, 3)
        segments = [
            DiarizationSegment(
                speaker=speaker,
                start=round(float(turn.start), 3),
                end=round(float(turn.end), 3),
            )
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        return segments, {"diarization_runtime_seconds": elapsed, "diarization_segment_count": len(segments)}


class MeetingASRAgent:
    def __init__(
        self,
        settings: MeetingAISettings | None = None,
        transcriber: FunASRTranscriber | None = None,
        diarizer: PyannoteDiarizer | None = None,
        voiceprint_identifier: VoiceprintIdentifier | None = None,
    ):
        self.settings = settings or get_settings()
        self.transcriber = transcriber or FunASRTranscriber(self.settings)
        self.diarizer = diarizer or PyannoteDiarizer(self.settings)
        self.voiceprint_identifier = voiceprint_identifier or VoiceprintIdentifier(self.settings)

    def transcribe(
        self,
        audio_path: str | Path,
        language: str = "zh",
        use_diarization: bool = True,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        enable_voiceprint: bool = False,
    ) -> TranscriptResult:
        path = Path(audio_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        audio_duration = get_audio_duration(path)
        transcript_segments, asr_metadata = self.transcriber.transcribe(path, language, audio_duration)

        warnings: list[str] = []
        diarization_backend = "disabled"
        diarization_segments: list[DiarizationSegment] = []
        diarization_metadata: dict[str, Any] = {}
        voiceprint_matches: dict[str, dict[str, Any]] = {}
        voiceprint_summary: dict[str, Any] = {
            "enabled": bool(enable_voiceprint),
            "matched_speakers": 0,
            "unknown_speakers": 0,
            "profile_count": 0,
        }

        if use_diarization:
            if self.diarizer.enabled:
                try:
                    diarization_segments, diarization_metadata = self.diarizer.diarize(
                        path,
                        num_speakers=num_speakers,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                    if diarization_segments:
                        transcript_segments = assign_speakers(transcript_segments, diarization_segments)
                        diarization_backend = self.settings.pyannote_model
                    else:
                        warnings.append("pyannote returned no speaker segments; fallback to a single speaker.")
                        diarization_backend = "single-speaker-fallback"
                except Exception as exc:  # pragma: no cover
                    warnings.append(f"pyannote diarization failed: {exc}")
                    diarization_backend = "single-speaker-fallback"
            else:
                warnings.append("HUGGINGFACE_TOKEN is missing; speaker diarization skipped.")
                diarization_backend = "single-speaker-fallback"
        elif enable_voiceprint:
            warnings.append("Voiceprint matching requires diarization; enable_diarization is false.")

        if enable_voiceprint and diarization_segments:
            try:
                voiceprint_matches = self.voiceprint_identifier.identify(
                    audio_path=path,
                    diarization_segments=diarization_segments,
                )
                if voiceprint_matches:
                    transcript_segments = apply_voiceprint_identities(transcript_segments, voiceprint_matches)
                    voiceprint_summary["matched_speakers"] = sum(
                        1 for payload in voiceprint_matches.values() if payload.get("status") == "matched"
                    )
                    voiceprint_summary["unknown_speakers"] = sum(
                        1 for payload in voiceprint_matches.values() if payload.get("status") != "matched"
                    )
                    voiceprint_summary["profile_count"] = max(
                        (int(payload.get("profile_count", 0)) for payload in voiceprint_matches.values()),
                        default=0,
                    )
                else:
                    profile_count = self.voiceprint_identifier.registry.get_profile_count()
                    voiceprint_summary["profile_count"] = profile_count
                    if profile_count == 0:
                        warnings.append("Voiceprint matching enabled, but no enrolled profiles were found.")
                    else:
                        warnings.append("Voiceprint matching skipped because no diarized speaker had enough audio.")
            except Exception as exc:  # pragma: no cover
                warnings.append(f"voiceprint matching failed: {exc}")

        full_text = "\n".join(f"[{segment.speaker}] {segment.text}" for segment in transcript_segments)
        low_confidence_assignments = sum(
            1 for segment in transcript_segments if segment.metadata.get("speaker_confidence") == "low"
        )
        metadata: dict[str, Any] = {
            **asr_metadata,
            **diarization_metadata,
            "device": self.settings.device,
            "audio_duration_seconds": round(audio_duration, 3),
            "warnings": warnings,
            "diarization_segments": [segment.model_dump() for segment in diarization_segments],
            "speaker_confidence_low_count": low_confidence_assignments,
            "speaker_confidence_high_count": sum(
                1 for segment in transcript_segments if segment.metadata.get("speaker_confidence") == "high"
            ),
            "voiceprint": {
                **voiceprint_summary,
                "matches": voiceprint_matches,
            },
        }

        return TranscriptResult(
            audio_path=str(path),
            language=language,
            asr_model=self.settings.funasr_model,
            diarization_backend=diarization_backend,
            segments=transcript_segments,
            full_text=full_text,
            metadata=metadata,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Week 1 ASR agent.")
    parser.add_argument("--audio", required=True, help="Path to the input audio file.")
    parser.add_argument("--language", default="zh", help="Language hint for post-processing. Default: zh")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--disable-diarization", action="store_true", help="Skip pyannote diarization.")
    parser.add_argument("--num-speakers", type=int, help="Known number of speakers.")
    parser.add_argument("--min-speakers", type=int, help="Lower bound for the speaker count.")
    parser.add_argument("--max-speakers", type=int, help="Upper bound for the speaker count.")
    parser.add_argument("--enable-voiceprint", action="store_true", help="Match diarized speakers against enrolled voiceprints.")
    return parser


def main() -> None:
    _setup_logging()
    args = build_parser().parse_args()

    settings = get_settings()
    agent = MeetingASRAgent(settings=settings)
    result = agent.transcribe(
        audio_path=args.audio,
        language=args.language,
        use_diarization=not args.disable_diarization,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        enable_voiceprint=args.enable_voiceprint,
    )

    payload = result.model_dump_json(indent=2)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        LOGGER.info("Transcript JSON saved to %s", output_path)

    print(payload)


if __name__ == "__main__":
    main()
