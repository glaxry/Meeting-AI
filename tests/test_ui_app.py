from __future__ import annotations

import json

import numpy as np

from meeting_ai.schemas import (
    ActionItem,
    ActionItemPriority,
    ActionItemResult,
    MeetingWorkflowResult,
    RetrievalRecord,
    StreamingSessionInfo,
    StreamingTranscriptEvent,
    SentimentLabel,
    SentimentResult,
    SentimentSegment,
    SummaryResult,
    TranscriptResult,
    TranscriptSegment,
    TranslationResult,
)
from meeting_ai.streaming import StreamingSessionRegistry
import ui.app as app_module
from ui.app import (
    analyze_via_api,
    build_sentiment_chart,
    build_speaker_distribution_chart,
    format_action_items,
    format_history,
    format_sentiment,
    format_summary,
    format_transcript,
    format_translation,
    parse_glossary,
    process_streaming_audio,
    reset_streaming_demo,
    start_streaming_demo,
    stop_streaming_demo,
)


def test_format_action_items_returns_readable_lines() -> None:
    text = format_action_items(
        ActionItemResult(
            items=[
                ActionItem(
                    assignee="Alice",
                    task="Ship next Tuesday",
                    deadline="next Tuesday",
                    priority=ActionItemPriority.HIGH,
                    source_quote="Please ship next Tuesday.",
                )
            ]
        )
    )

    assert "Alice" in text
    assert "Ship next Tuesday" in text


def test_format_history_renders_scores() -> None:
    result = MeetingWorkflowResult(
        history=[RetrievalRecord(meeting_id="meeting-1", document="Previous decision", score=0.91)],
    )

    text = format_history(result)

    assert "meeting-1" in text
    assert "0.910" in text


def test_parse_glossary_returns_mapping() -> None:
    assert parse_glossary("语音识别=speech-recognition\n预算=budget") == {
        "语音识别": "speech-recognition",
        "预算": "budget",
    }


def test_analyze_via_api_posts_multipart_request(monkeypatch) -> None:
    audio_path = app_module.ROOT / "data" / "samples" / "asr_example_zh.wav"
    captured = {}

    class DummyResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return MeetingWorkflowResult(selected_agents=[]).model_dump(mode="json")

    def fake_post(url, data, files, timeout):
        captured["url"] = url
        captured["data"] = data
        captured["file_name"] = files["audio"][0]
        captured["file_open"] = not files["audio"][1].closed
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(app_module.requests, "post", fake_post)

    result = analyze_via_api(
        str(audio_path),
        language="zh",
        provider="deepseek",
        selected_agents=[],
        target_language="en",
        sentiment_route="llm",
        history_query="last decision",
        glossary={"语音识别": "speech-recognition"},
        use_diarization=True,
        num_speakers=None,
        enable_voiceprint=True,
    )

    assert captured["url"].endswith("/meetings/analyze")
    assert json.loads(captured["data"]["selected_agents"]) == []
    assert json.loads(captured["data"]["glossary"]) == {"语音识别": "speech-recognition"}
    assert captured["data"]["enable_voiceprint"] == "true"
    assert captured["file_name"] == "asr_example_zh.wav"
    assert captured["file_open"] is True
    assert captured["timeout"] is None
    assert result.selected_agents == []


def test_format_summary_renders_sections() -> None:
    result = MeetingWorkflowResult(
        selected_agents=["summary"],
        summary=SummaryResult(topics=["Launch"], decisions=["Ship"], follow_ups=["Send memo"]),
    )

    text = format_summary(result)

    assert "Topics:" in text
    assert "- Ship" in text


def test_format_summary_respects_unselected_agent() -> None:
    result = MeetingWorkflowResult(
        selected_agents=["translation"],
        summary=SummaryResult(topics=["Launch"], decisions=["Ship"], follow_ups=[]),
    )

    assert format_summary(result) == "Summary was not selected."


def test_format_translation_returns_readable_lines() -> None:
    result = TranslationResult(
        source_language="zh",
        target_language="en",
        segments=[TranscriptSegment(speaker="SPEAKER_00", text="Hello", start=0.0, end=1.0)],
        full_text="[SPEAKER_00] Hello",
    )

    text = format_translation(result, selected_agents=["translation"])

    assert "Source language: zh" in text
    assert "SPEAKER_00" in text
    assert "{" not in text


def test_format_sentiment_returns_readable_lines() -> None:
    result = SentimentResult(
        route="llm",
        overall_tone=SentimentLabel.AGREEMENT,
        segments=[
            SentimentSegment(
                text="Sounds good.",
                sentiment=SentimentLabel.AGREEMENT,
                confidence=0.9,
                speaker="SPEAKER_00",
                start=0.0,
                end=1.0,
            )
        ],
        timeline=[],
    )

    text = format_sentiment(result, selected_agents=["sentiment"])

    assert "Overall tone: agreement" in text
    assert "agreement (0.90)" in text
    assert "{" not in text


def test_format_transcript_shows_week2_annotations() -> None:
    result = MeetingWorkflowResult(
        transcript=TranscriptResult(
            audio_path="demo.wav",
            language="zh",
            asr_model="iic/SenseVoiceSmall",
            diarization_backend="mock",
            segments=[
                TranscriptSegment(
                    speaker="SPEAKER_00",
                    text="欢迎大家",
                    start=0.0,
                    end=1.0,
                    emotion="neutral",
                    event="speech",
                    metadata={
                        "speaker_confidence": "high",
                        "speaker_identity_name": "Alice",
                        "speaker_identity_score": 0.93,
                    },
                )
            ],
            full_text="[SPEAKER_00] 欢迎大家",
            metadata={},
        )
    )

    text = format_transcript(result)

    assert "emotion=neutral" in text
    assert "event=speech" in text
    assert "speaker_confidence=high" in text
    assert "identity=Alice (0.930)" in text


def test_format_sentiment_includes_timeline_snapshot_lines() -> None:
    result = SentimentResult(
        route="transformer",
        overall_tone=SentimentLabel.TENSION,
        segments=[
            SentimentSegment(
                text="This delay is a serious risk.",
                sentiment=SentimentLabel.TENSION,
                confidence=0.88,
                speaker="SPEAKER_01",
                start=2.0,
                end=3.0,
            )
        ],
        timeline=[
            {
                "window_start": 0.0,
                "window_end": 120.0,
                "dominant_label": "tension",
                "label_distribution": {"tension": 1.0},
                "speakers_involved": ["SPEAKER_01"],
            }
        ],
    )

    text = format_sentiment(result, selected_agents=["sentiment"])

    assert "Timeline snapshots:" in text
    assert "speakers=SPEAKER_01" in text


def test_format_action_items_respects_unselected_agent() -> None:
    assert format_action_items(ActionItemResult(), selected_agents=["summary"]) == "Action item extraction was not selected."


def test_build_sentiment_chart_prefers_timeline_snapshots() -> None:
    result = SentimentResult(
        route="transformer",
        overall_tone=SentimentLabel.TENSION,
        segments=[
            SentimentSegment(
                text="This delay is a serious risk.",
                sentiment=SentimentLabel.TENSION,
                confidence=0.88,
                speaker="SPEAKER_01",
                start=2.0,
                end=3.0,
            )
        ],
        timeline=[
            {
                "window_start": 0.0,
                "window_end": 120.0,
                "dominant_label": "tension",
                "label_distribution": {"tension": 1.0},
                "speakers_involved": ["SPEAKER_01"],
            }
        ],
    )

    chart = build_sentiment_chart(result)

    assert chart is not None
    assert chart.to_dict(orient="records") == [
        {
            "time_seconds": 60.0,
            "sentiment_score": -2.0,
            "label": "tension",
        }
    ]


def test_build_speaker_distribution_chart_aggregates_duration_per_speaker() -> None:
    result = MeetingWorkflowResult(
        transcript=TranscriptResult(
            audio_path="demo.wav",
            language="zh",
            asr_model="iic/SenseVoiceSmall",
            diarization_backend="mock",
            segments=[
                TranscriptSegment(speaker="SPEAKER_00", text="hello", start=0.0, end=1.5),
                TranscriptSegment(speaker="SPEAKER_01", text="world", start=1.5, end=3.0),
                TranscriptSegment(speaker="SPEAKER_00", text="again", start=3.0, end=5.0),
            ],
            full_text="[SPEAKER_00] hello",
            metadata={},
        )
    )

    chart = build_speaker_distribution_chart(result)

    assert chart is not None
    assert chart.to_dict(orient="records") == [
        {
            "speaker": "SPEAKER_00",
            "duration_seconds": 3.5,
            "segment_count": 2,
        },
        {
            "speaker": "SPEAKER_01",
            "duration_seconds": 1.5,
            "segment_count": 1,
        },
    ]


class DummyStreamingSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.language = "zh"
        self.sample_rate = 16000
        self.calls = 0

    def session_info(self):
        return StreamingSessionInfo(
            session_id=self.session_id,
            language=self.language,
            sample_rate=self.sample_rate,
            target_sample_rate=16000,
            asr_model="paraformer-zh-streaming",
            chunk_size=[0, 10, 5],
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=1,
        )

    def process_chunk(self, samples, *, sample_rate: int | None = None, is_final: bool = False):
        self.calls += 1
        return StreamingTranscriptEvent(
            session_id=self.session_id,
            chunk_index=self.calls - 1,
            delta_text="欢迎",
            cumulative_text="欢迎" if self.calls == 1 else "欢迎大家",
            is_final=is_final,
            received_seconds=0.5 * self.calls,
            sample_rate=sample_rate or self.sample_rate,
            target_sample_rate=16000,
        )

    def snapshot(self, *, is_final: bool = False):
        return StreamingTranscriptEvent(
            session_id=self.session_id,
            chunk_index=max(self.calls - 1, 0),
            delta_text="",
            cumulative_text="欢迎大家",
            is_final=is_final,
            received_seconds=1.0,
            sample_rate=self.sample_rate,
            target_sample_rate=16000,
        )


class DummyStreamingTranscriber:
    def __init__(self) -> None:
        self.sessions: list[DummyStreamingSession] = []

    def create_session(self, *, language: str = "zh", sample_rate: int | None = None, session_id: str | None = None):
        session = DummyStreamingSession(session_id or "stream-ui")
        self.sessions.append(session)
        return session


def test_streaming_demo_callbacks_manage_session_state(monkeypatch) -> None:
    fake_transcriber = DummyStreamingTranscriber()
    monkeypatch.setattr(app_module, "get_streaming_transcriber", lambda: fake_transcriber)
    monkeypatch.setattr(app_module, "STREAMING_SESSIONS", StreamingSessionRegistry())

    transcript, status, state = start_streaming_demo("zh")
    live_transcript, live_status, state = process_streaming_audio(
        (16000, np.array([0.0, 0.1, -0.1], dtype=np.float32)),
        "zh",
        state,
    )
    final_transcript, final_status, final_state = stop_streaming_demo(state)
    _, cleared_transcript, reset_status, reset_state = reset_streaming_demo(final_state)

    assert transcript == ""
    assert "Streaming session ready" in status
    assert live_transcript == "欢迎"
    assert "chunk_index=0" in live_status
    assert final_transcript == "欢迎大家"
    assert "Streaming finished" in final_status
    assert cleared_transcript == ""
    assert reset_status == "Streaming session reset."
    assert reset_state is None
