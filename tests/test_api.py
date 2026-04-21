from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from meeting_ai import api as api_module
from meeting_ai.schemas import (
    MeetingWorkflowResult,
    StreamingSessionInfo,
    StreamingTranscriptEvent,
    TranscriptResult,
    TranscriptSegment,
)
from meeting_ai.streaming import encode_pcm16_base64


class FakeOrchestrator:
    def __init__(self) -> None:
        self.kwargs = {}
        self.audio_path_existed = False

    def run(self, **kwargs) -> MeetingWorkflowResult:
        self.kwargs = kwargs
        self.audio_path_existed = Path(kwargs["audio_path"]).exists()
        transcript = TranscriptResult(
            audio_path=kwargs["audio_path"],
            language=kwargs["language"],
            asr_model="fake",
            diarization_backend="fake",
            segments=[TranscriptSegment(speaker="SPEAKER_00", text="Hello", start=0.0, end=1.0)],
            full_text="[SPEAKER_00] Hello",
        )
        return MeetingWorkflowResult(
            transcript=transcript,
            selected_agents=kwargs["selected_agents"],
            metadata={"provider": kwargs["provider"].value},
        )


class FakeStreamingSession:
    def __init__(self, language: str, sample_rate: int, session_id: str | None) -> None:
        self.language = language
        self.sample_rate = sample_rate
        self.session_id = session_id or "fake-stream"
        self.calls: list[dict[str, object]] = []

    def session_info(self) -> StreamingSessionInfo:
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

    def process_chunk(self, audio_chunk, *, sample_rate: int | None = None, is_final: bool = False) -> StreamingTranscriptEvent:
        self.calls.append(
            {
                "audio_chunk": audio_chunk,
                "sample_rate": sample_rate,
                "is_final": is_final,
            }
        )
        return StreamingTranscriptEvent(
            session_id=self.session_id,
            chunk_index=len(self.calls) - 1,
            delta_text="欢迎",
            cumulative_text="欢迎" if len(self.calls) == 1 else "欢迎大家",
            is_final=is_final,
            received_seconds=0.2 * len(self.calls),
            sample_rate=sample_rate or self.sample_rate,
            target_sample_rate=16000,
        )


class FakeStreamingTranscriber:
    def __init__(self) -> None:
        self.created_sessions: list[FakeStreamingSession] = []

    def create_session(self, *, language: str = "zh", sample_rate: int | None = None, session_id: str | None = None):
        session = FakeStreamingSession(language=language, sample_rate=sample_rate or 16000, session_id=session_id)
        self.created_sessions.append(session)
        return session


def test_analyze_meeting_endpoint_preserves_request_options(monkeypatch) -> None:
    fake = FakeOrchestrator()
    monkeypatch.setattr(api_module, "get_orchestrator", lambda: fake)
    client = TestClient(api_module.app)

    response = client.post(
        "/meetings/analyze",
        files={"audio": ("demo.wav", b"fake wav", "audio/wav")},
        data={
            "language": "en",
            "provider": "deepseek",
            "selected_agents": "[]",
            "target_language": "zh",
            "sentiment_route": "transformer",
            "history_query": "previous decision",
            "glossary": '{"budget": "budget-cn"}',
            "use_diarization": "false",
            "num_speakers": "2",
            "enable_voiceprint": "true",
        },
    )

    assert response.status_code == 200
    assert response.json()["selected_agents"] == []
    assert fake.audio_path_existed is True
    assert fake.kwargs["selected_agents"] == []
    assert fake.kwargs["target_language"] == "zh"
    assert fake.kwargs["glossary"] == {"budget": "budget-cn"}
    assert fake.kwargs["history_query"] == "previous decision"
    assert fake.kwargs["use_diarization"] is False
    assert fake.kwargs["num_speakers"] == 2
    assert fake.kwargs["enable_voiceprint"] is True


def test_analyze_meeting_rejects_invalid_selected_agents(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "get_orchestrator", lambda: FakeOrchestrator())
    client = TestClient(api_module.app)

    response = client.post(
        "/meetings/analyze",
        files={"audio": ("demo.wav", b"fake wav", "audio/wav")},
        data={"selected_agents": "{}"},
    )

    assert response.status_code == 400
    assert "selected_agents must be a list" in response.json()["detail"]


def test_stream_transcribe_websocket_returns_ready_and_final_events(monkeypatch) -> None:
    fake = FakeStreamingTranscriber()
    monkeypatch.setattr(api_module, "get_streaming_transcriber", lambda: fake)
    client = TestClient(api_module.app)

    with client.websocket_connect("/stream/transcribe") as websocket:
        websocket.send_json({"type": "start", "language": "zh", "sample_rate": 16000, "session_id": "demo"})
        ready = websocket.receive_json()

        websocket.send_json(
            {
                "type": "chunk",
                "audio_base64": encode_pcm16_base64(np.array([0.0, 0.2, -0.1], dtype=np.float32)),
                "sample_rate": 16000,
                "is_final": True,
            }
        )
        final = websocket.receive_json()

    assert ready["event"] == "ready"
    assert ready["session"]["session_id"] == "demo"
    assert final["event"] == "final"
    assert final["transcript"]["cumulative_text"] == "欢迎"
    assert fake.created_sessions[0].calls[0]["is_final"] is True


def test_stream_transcribe_rejects_chunk_before_start(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "get_streaming_transcriber", lambda: FakeStreamingTranscriber())
    client = TestClient(api_module.app)

    with client.websocket_connect("/stream/transcribe") as websocket:
        websocket.send_json(
            {
                "type": "chunk",
                "audio_base64": encode_pcm16_base64(np.array([0.0, 0.2], dtype=np.float32)),
            }
        )
        error = websocket.receive_json()

    assert error["event"] == "error"
    assert "start message" in error["detail"]
