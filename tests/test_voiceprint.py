from __future__ import annotations

from pathlib import Path

from meeting_ai.config import get_settings
from meeting_ai.schemas import DiarizationSegment, TranscriptSegment
from meeting_ai.voiceprint import VoiceprintIdentifier, VoiceprintRegistry, apply_voiceprint_identities


class FakeVoiceEncoder:
    def __init__(self) -> None:
        self.enrollment_requests: list[dict[str, object]] = []
        self.identification_requests: list[dict[str, object]] = []

    def encode_audio_file(self, audio_path, *, start_seconds=None, end_seconds=None):
        self.enrollment_requests.append(
            {
                "audio_path": str(audio_path),
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
            }
        )
        return [1.0, 0.0], {
            "audio_path": str(audio_path),
            "duration_seconds": 2.5,
            "sample_rate": 16000,
        }

    def encode_speaker_segments(self, *, audio_path, diarization_segments):
        self.identification_requests.append(
            {
                "audio_path": str(audio_path),
                "segment_count": len(diarization_segments),
            }
        )
        return {
            "SPEAKER_00": {"embedding": [0.9, 0.1], "duration_seconds": 3.1, "segment_count": 2},
            "SPEAKER_01": {"embedding": [0.0, 1.0], "duration_seconds": 2.2, "segment_count": 1},
        }


def build_settings(tmp_path: Path):
    return get_settings().model_copy(
        update={
            "use_gpu": False,
            "voiceprint_dir": tmp_path / "voiceprints",
            "voiceprint_model_cache_dir": tmp_path / "voiceprints" / "_model_cache",
            "voiceprint_match_threshold": 0.65,
        }
    )


def test_voiceprint_registry_enrolls_and_lists_profiles(tmp_path) -> None:
    settings = build_settings(tmp_path)
    encoder = FakeVoiceEncoder()
    registry = VoiceprintRegistry(settings=settings, encoder=encoder)

    profile = registry.enroll(name="Alice", audio_path="alice.wav")

    assert profile["name"] == "Alice"
    assert profile["metadata"]["audio_path"] == "alice.wav"
    assert registry.get_profile_count() == 1
    assert registry.list_profiles()[0]["name"] == "Alice"


def test_voiceprint_identifier_matches_best_profile_and_relabels_segments(tmp_path) -> None:
    settings = build_settings(tmp_path)
    encoder = FakeVoiceEncoder()
    registry = VoiceprintRegistry(settings=settings, encoder=encoder)
    registry.save_profile(name="Alice", embedding=[1.0, 0.0], overwrite=True)
    registry.save_profile(name="Bob", embedding=[0.0, 1.0], overwrite=True)

    identifier = VoiceprintIdentifier(settings=settings, registry=registry, encoder=encoder)
    matches = identifier.identify(
        audio_path="meeting.wav",
        diarization_segments=[
            DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.5),
            DiarizationSegment(speaker="SPEAKER_01", start=1.5, end=3.0),
        ],
    )

    updated = apply_voiceprint_identities(
        [
            TranscriptSegment(speaker="SPEAKER_00", text="Budget first.", start=0.0, end=1.0),
            TranscriptSegment(speaker="SPEAKER_01", text="Ship next week.", start=1.0, end=2.0),
        ],
        matches,
    )

    assert matches["SPEAKER_00"]["matched_name"] == "Alice"
    assert matches["SPEAKER_01"]["matched_name"] == "Bob"
    assert updated[0].speaker == "Alice"
    assert updated[0].metadata["original_speaker_label"] == "SPEAKER_00"
    assert updated[0].metadata["speaker_identity_source"] == "voiceprint_registry"
    assert updated[1].speaker == "Bob"
