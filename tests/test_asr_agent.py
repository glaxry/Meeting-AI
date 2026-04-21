from __future__ import annotations

from pathlib import Path

from meeting_ai.asr_agent import (
    MeetingASRAgent,
    _funasr_generate_kwargs,
    _funasr_model_kwargs,
    _sensevoice_language_hint,
    assign_speakers,
    normalize_sentence_info,
)
from meeting_ai.config import get_settings
from meeting_ai.schemas import DiarizationSegment, TranscriptSegment


def test_normalize_sentence_info_converts_milliseconds() -> None:
    sentence_info = [
        {"text": "你好", "start": 0, "end": 1500},
        {"text": "世界", "start": 1500, "end": 3200},
    ]

    segments = normalize_sentence_info(sentence_info, audio_duration=3.2)

    assert len(segments) == 2
    assert segments[0].start == 0.0
    assert segments[0].end == 1.5
    assert segments[1].start == 1.5
    assert segments[1].end == 3.2


def test_assign_speakers_prefers_overlap() -> None:
    transcript_segments = normalize_sentence_info(
        [
            {"text": "第一句", "start": 0, "end": 2000},
            {"text": "第二句", "start": 2000, "end": 4000},
        ],
        audio_duration=4.0,
    )
    diarization_segments = [
        DiarizationSegment(speaker="SPEAKER_A", start=0.0, end=1.8),
        DiarizationSegment(speaker="SPEAKER_B", start=1.8, end=4.0),
    ]

    assigned = assign_speakers(transcript_segments, diarization_segments)

    assert assigned[0].speaker == "SPEAKER_A"
    assert assigned[1].speaker == "SPEAKER_B"
    assert assigned[0].metadata["speaker_confidence"] == "high"
    assert assigned[1].metadata["speaker_confidence"] == "high"


def test_sensevoice_helpers_switch_remote_code_and_language() -> None:
    settings = get_settings().model_copy(update={"funasr_model": "iic/SenseVoiceSmall"})

    model_kwargs = _funasr_model_kwargs(settings)
    generate_kwargs = _funasr_generate_kwargs(settings, "zh")

    assert model_kwargs["trust_remote_code"] is True
    assert "punc_model" not in model_kwargs
    assert generate_kwargs["language"] == "zn"
    assert generate_kwargs["use_itn"] is False


def test_sensevoice_language_hint_defaults_to_auto() -> None:
    assert _sensevoice_language_hint("fr") == "auto"


def test_normalize_sentence_info_strips_sensevoice_control_tokens() -> None:
    segments = normalize_sentence_info(
        [
            {"text": "<|zh|><|NEUTRAL|><|Speech|><|woitn|>欢迎大家", "start": 0, "end": 1000},
        ],
        audio_duration=1.0,
    )

    assert segments[0].text == "欢迎大家"
    assert segments[0].emotion == "neutral"
    assert segments[0].event == "speech"


def test_assign_speakers_marks_low_confidence_when_overlap_is_too_small() -> None:
    transcript_segments = normalize_sentence_info(
        [
            {"text": "这句比较短", "start": 1000, "end": 1200},
        ],
        audio_duration=2.0,
    )
    diarization_segments = [
        DiarizationSegment(speaker="SPEAKER_A", start=0.0, end=0.9),
        DiarizationSegment(speaker="SPEAKER_B", start=1.3, end=2.0),
    ]

    assigned = assign_speakers(transcript_segments, diarization_segments, min_overlap_ratio=0.2)

    assert assigned[0].speaker == "SPEAKER_B"
    assert assigned[0].metadata["speaker_confidence"] == "low"
    assert assigned[0].metadata["assignment_strategy"] == "nearest_segment"


def test_normalize_sentence_info_reads_explicit_emotion_and_event_fields() -> None:
    segments = normalize_sentence_info(
        [
            {"text": "太好了", "start": 0, "end": 1000, "emotion": "happy", "event": "applause"},
        ],
        audio_duration=1.0,
    )

    assert segments[0].emotion == "happy"
    assert segments[0].event == "applause"


class FakeTranscriber:
    def transcribe(self, audio_path, language, audio_duration):
        return (
            [
                TranscriptSegment(speaker="SPEAKER_A", text="Budget first.", start=0.0, end=1.5),
                TranscriptSegment(speaker="SPEAKER_B", text="Ship next week.", start=1.5, end=3.0),
            ],
            {"asr_runtime_seconds": 1.2},
        )


class FakeDiarizer:
    enabled = True

    def diarize(self, path, num_speakers=None, min_speakers=None, max_speakers=None):
        return (
            [
                DiarizationSegment(speaker="SPEAKER_A", start=0.0, end=1.5),
                DiarizationSegment(speaker="SPEAKER_B", start=1.5, end=3.0),
            ],
            {"diarization_runtime_seconds": 0.5},
        )


class FakeRegistry:
    def get_profile_count(self):
        return 2


class FakeVoiceprintIdentifier:
    def __init__(self):
        self.registry = FakeRegistry()

    def identify(self, *, audio_path, diarization_segments):
        return {
            "SPEAKER_A": {
                "matched_name": "Alice",
                "score": 0.93,
                "threshold": 0.65,
                "status": "matched",
                "profile_count": 2,
                "duration_seconds": 1.5,
                "segment_count": 1,
            },
            "SPEAKER_B": {
                "matched_name": None,
                "score": 0.31,
                "threshold": 0.65,
                "status": "unknown",
                "profile_count": 2,
                "duration_seconds": 1.5,
                "segment_count": 1,
            },
        }


def test_meeting_asr_agent_applies_voiceprint_identity_matches() -> None:
    settings = get_settings().model_copy(update={"use_gpu": False})
    agent = MeetingASRAgent(
        settings=settings,
        transcriber=FakeTranscriber(),
        diarizer=FakeDiarizer(),
        voiceprint_identifier=FakeVoiceprintIdentifier(),
    )

    result = agent.transcribe(
        audio_path=Path("data/samples/asr_example_zh.wav"),
        language="zh",
        enable_voiceprint=True,
    )

    assert result.segments[0].speaker == "Alice"
    assert result.segments[0].metadata["original_speaker_label"] == "SPEAKER_A"
    assert result.segments[0].metadata["speaker_identity_name"] == "Alice"
    assert result.segments[1].speaker == "SPEAKER_B"
    assert result.metadata["voiceprint"]["matched_speakers"] == 1
    assert result.metadata["voiceprint"]["unknown_speakers"] == 1
