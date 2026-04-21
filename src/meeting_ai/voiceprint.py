from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchaudio
import torch.nn.functional as F

from .config import MeetingAISettings, get_settings
from .schemas import DiarizationSegment, TranscriptSegment


TARGET_SAMPLE_RATE = 16000


def _coerce_mono_waveform(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    if waveform.ndim == 2:
        if waveform.shape[0] == 1:
            return waveform
        return waveform.mean(dim=0, keepdim=True)
    raise ValueError("Waveform must be one-dimensional or channel-first two-dimensional.")


def _resample_waveform(waveform: torch.Tensor, sample_rate: int, target_sample_rate: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    if sample_rate == target_sample_rate:
        return waveform
    return torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_tensor = torch.tensor(left, dtype=torch.float32)
    right_tensor = torch.tensor(right, dtype=torch.float32)
    score = F.cosine_similarity(left_tensor.unsqueeze(0), right_tensor.unsqueeze(0)).item()
    return round(float(score), 6)


class SpeechBrainVoiceEncoder:
    def __init__(self, settings: MeetingAISettings | None = None):
        self.settings = settings or get_settings()
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy

            self._model = EncoderClassifier.from_hparams(
                source=self.settings.voiceprint_model,
                savedir=str(self.settings.ensure_voiceprint_model_cache_dir()),
                run_opts={"device": self.settings.device},
                local_strategy=LocalStrategy.COPY,
            )
        return self._model

    def encode_waveform(self, waveform: torch.Tensor, sample_rate: int) -> list[float]:
        classifier = self._load()
        normalized = _coerce_mono_waveform(waveform.float())
        normalized = _resample_waveform(normalized, sample_rate, TARGET_SAMPLE_RATE)
        target_device = torch.device(self.settings.device)
        normalized = normalized.to(target_device)
        wav_lens = torch.ones(normalized.shape[0], device=target_device)
        with torch.inference_mode():
            embedding = classifier.encode_batch(
                normalized,
                wav_lens=wav_lens,
                normalize=True,
            )
        flattened = embedding.squeeze().detach().cpu().flatten()
        return [float(value) for value in flattened.tolist()]

    def encode_audio_file(
        self,
        audio_path: str | Path,
        *,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> tuple[list[float], dict[str, Any]]:
        path = Path(audio_path).expanduser().resolve()
        waveform, sample_rate = torchaudio.load(str(path))
        waveform = _coerce_mono_waveform(waveform)

        start_index = 0 if start_seconds is None else max(0, int(float(start_seconds) * sample_rate))
        end_index = waveform.shape[1] if end_seconds is None else min(waveform.shape[1], int(float(end_seconds) * sample_rate))
        clipped = waveform[:, start_index:end_index]
        if clipped.numel() == 0:
            raise ValueError("Enrollment audio clip is empty after applying the requested time range.")

        duration_seconds = round(clipped.shape[1] / float(sample_rate), 3)
        if duration_seconds < self.settings.voiceprint_min_total_seconds:
            raise ValueError(
                f"Enrollment clip is too short ({duration_seconds}s). "
                f"Provide at least {self.settings.voiceprint_min_total_seconds}s of reference audio."
            )

        return self.encode_waveform(clipped, sample_rate), {
            "audio_path": str(path),
            "sample_rate": sample_rate,
            "duration_seconds": duration_seconds,
            "start_seconds": None if start_seconds is None else round(float(start_seconds), 3),
            "end_seconds": None if end_seconds is None else round(float(end_seconds), 3),
        }

    def encode_speaker_segments(
        self,
        *,
        audio_path: str | Path,
        diarization_segments: list[DiarizationSegment],
    ) -> dict[str, dict[str, Any]]:
        if not diarization_segments:
            return {}

        path = Path(audio_path).expanduser().resolve()
        waveform, sample_rate = torchaudio.load(str(path))
        waveform = _coerce_mono_waveform(waveform)

        grouped_waveforms: dict[str, list[torch.Tensor]] = defaultdict(list)
        grouped_durations: dict[str, float] = defaultdict(float)
        grouped_segments: dict[str, int] = defaultdict(int)
        min_segment_seconds = float(self.settings.voiceprint_min_segment_seconds)

        for segment in diarization_segments:
            duration_seconds = max(float(segment.end) - float(segment.start), 0.0)
            if duration_seconds < min_segment_seconds:
                continue
            start_index = max(0, int(float(segment.start) * sample_rate))
            end_index = min(waveform.shape[1], int(float(segment.end) * sample_rate))
            if end_index <= start_index:
                continue
            grouped_waveforms[segment.speaker].append(waveform[:, start_index:end_index])
            grouped_durations[segment.speaker] += duration_seconds
            grouped_segments[segment.speaker] += 1

        embeddings: dict[str, dict[str, Any]] = {}
        for speaker_label, chunks in grouped_waveforms.items():
            total_seconds = round(grouped_durations[speaker_label], 3)
            if total_seconds < self.settings.voiceprint_min_total_seconds:
                continue
            merged_waveform = torch.cat(chunks, dim=1)
            embeddings[speaker_label] = {
                "embedding": self.encode_waveform(merged_waveform, sample_rate),
                "duration_seconds": total_seconds,
                "segment_count": grouped_segments[speaker_label],
            }
        return embeddings


class VoiceprintRegistry:
    def __init__(
        self,
        settings: MeetingAISettings | None = None,
        encoder: SpeechBrainVoiceEncoder | None = None,
        registry_path: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.encoder = encoder or SpeechBrainVoiceEncoder(self.settings)
        root = self.settings.ensure_voiceprint_dir()
        self.registry_path = registry_path or (root / "profiles.json")

    def _load_payload(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"profiles": []}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _save_payload(self, payload: dict[str, Any]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_profiles(self) -> list[dict[str, Any]]:
        payload = self._load_payload()
        profiles = payload.get("profiles") or []
        return sorted((dict(profile) for profile in profiles), key=lambda item: str(item.get("name", "")))

    def get_profile_count(self) -> int:
        return len(self.list_profiles())

    def save_profile(
        self,
        *,
        name: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Voiceprint profile name must not be empty.")

        payload = self._load_payload()
        profiles = payload.setdefault("profiles", [])
        existing_index = next(
            (index for index, profile in enumerate(profiles) if str(profile.get("name", "")).strip() == normalized_name),
            None,
        )
        if existing_index is not None and not overwrite:
            raise FileExistsError(f"Voiceprint profile already exists: {normalized_name}")

        profile = {
            "name": normalized_name,
            "embedding": embedding,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": dict(metadata or {}),
        }
        if existing_index is None:
            profiles.append(profile)
        else:
            profiles[existing_index] = profile
        self._save_payload(payload)
        return profile

    def enroll(
        self,
        *,
        name: str,
        audio_path: str | Path,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        embedding, metadata = self.encoder.encode_audio_file(
            audio_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        metadata.update(
            {
                "voiceprint_model": self.settings.voiceprint_model,
                "match_threshold": self.settings.voiceprint_match_threshold,
            }
        )
        return self.save_profile(
            name=name,
            embedding=embedding,
            metadata=metadata,
            overwrite=overwrite,
        )


class VoiceprintIdentifier:
    def __init__(
        self,
        settings: MeetingAISettings | None = None,
        registry: VoiceprintRegistry | None = None,
        encoder: SpeechBrainVoiceEncoder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.encoder = encoder or SpeechBrainVoiceEncoder(self.settings)
        self.registry = registry or VoiceprintRegistry(self.settings, encoder=self.encoder)

    def identify(
        self,
        *,
        audio_path: str | Path,
        diarization_segments: list[DiarizationSegment],
    ) -> dict[str, dict[str, Any]]:
        profiles = self.registry.list_profiles()
        if not profiles:
            return {}

        speaker_embeddings = self.encoder.encode_speaker_segments(
            audio_path=audio_path,
            diarization_segments=diarization_segments,
        )
        matches: dict[str, dict[str, Any]] = {}
        threshold = float(self.settings.voiceprint_match_threshold)

        for diarized_speaker, payload in speaker_embeddings.items():
            best_name: str | None = None
            best_score = -1.0
            for profile in profiles:
                score = cosine_similarity(payload["embedding"], profile["embedding"])
                if score > best_score:
                    best_score = score
                    best_name = str(profile["name"])

            matched_name = best_name if best_score >= threshold else None
            matches[diarized_speaker] = {
                "matched_name": matched_name,
                "score": round(best_score, 6),
                "threshold": threshold,
                "status": "matched" if matched_name else "unknown",
                "profile_count": len(profiles),
                "duration_seconds": payload["duration_seconds"],
                "segment_count": payload["segment_count"],
            }
        return matches


def apply_voiceprint_identities(
    transcript_segments: list[TranscriptSegment],
    speaker_matches: dict[str, dict[str, Any]],
) -> list[TranscriptSegment]:
    updated_segments: list[TranscriptSegment] = []
    for segment in transcript_segments:
        match = speaker_matches.get(segment.speaker)
        metadata = dict(segment.metadata)
        metadata["original_speaker_label"] = segment.speaker

        updated_speaker = segment.speaker
        if match:
            metadata["speaker_identity_status"] = match["status"]
            metadata["speaker_identity_score"] = match["score"]
            metadata["speaker_identity_threshold"] = match["threshold"]
            if match.get("matched_name"):
                metadata["speaker_identity_name"] = match["matched_name"]
                metadata["speaker_identity_source"] = "voiceprint_registry"
                updated_speaker = str(match["matched_name"])
            else:
                metadata["speaker_identity_name"] = None
        updated_segments.append(segment.model_copy(update={"speaker": updated_speaker, "metadata": metadata}))
    return updated_segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enroll a voiceprint profile from a reference audio clip.")
    parser.add_argument("--name", required=True, help="Speaker name to register.")
    parser.add_argument("--audio", required=True, help="Path to the enrollment audio clip.")
    parser.add_argument("--start-seconds", type=float, help="Optional start offset within the enrollment audio.")
    parser.add_argument("--end-seconds", type=float, help="Optional end offset within the enrollment audio.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the existing profile if it already exists.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    registry = VoiceprintRegistry(settings=get_settings())
    profile = registry.enroll(
        name=args.name,
        audio_path=args.audio,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        overwrite=args.overwrite,
    )
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
