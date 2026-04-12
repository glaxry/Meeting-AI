from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from meeting_ai import api as api_module
from meeting_ai.schemas import MeetingWorkflowResult, TranscriptResult, TranscriptSegment


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
