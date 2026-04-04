from __future__ import annotations

from meeting_ai.asr_agent import assign_speakers, normalize_sentence_info
from meeting_ai.schemas import DiarizationSegment


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

