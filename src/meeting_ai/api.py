from __future__ import annotations

import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import get_settings
from .orchestrator import MeetingOrchestrator
from .runtime import find_ffmpeg
from .schemas import LLMProvider, MeetingWorkflowResult


app = FastAPI(title="Meeting AI API", version="0.1.0")


@lru_cache(maxsize=1)
def get_orchestrator() -> MeetingOrchestrator:
    return MeetingOrchestrator(settings=get_settings())


def _parse_json_field(raw: str, expected_type: type, field_name: str) -> Any:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON for {field_name}: {exc.msg}") from exc
    if not isinstance(value, expected_type):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a {expected_type.__name__}.")
    return value


@app.get("/health")
def health() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "device": settings.device,
        "ffmpeg_path": find_ffmpeg(),
    }


@app.post("/meetings/analyze", response_model=MeetingWorkflowResult)
async def analyze_meeting(
    audio: UploadFile = File(...),
    language: str = Form("zh"),
    provider: str = Form(LLMProvider.DEEPSEEK.value),
    selected_agents: str = Form('["summary", "translation", "action_items", "sentiment"]'),
    target_language: str = Form("en"),
    sentiment_route: str = Form("llm"),
    history_query: str = Form(""),
    glossary: str = Form("{}"),
    use_diarization: bool = Form(True),
    num_speakers: int | None = Form(None),
) -> MeetingWorkflowResult:
    agent_list = _parse_json_field(selected_agents, list, "selected_agents")
    glossary_dict = _parse_json_field(glossary, dict, "glossary")
    try:
        llm_provider = LLMProvider(provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}") from exc

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(audio.file, temp_file)
            temp_path = Path(temp_file.name)

        return get_orchestrator().run(
            audio_path=str(temp_path),
            language=language,
            provider=llm_provider,
            selected_agents=[str(agent) for agent in agent_list],
            target_language=target_language,
            glossary={str(source): str(target) for source, target in glossary_dict.items()},
            sentiment_route=sentiment_route,
            history_query=history_query.strip() or None,
            use_diarization=use_diarization,
            num_speakers=num_speakers,
        )
    finally:
        await audio.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
