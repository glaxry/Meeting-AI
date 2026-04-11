from __future__ import annotations

import time
from pathlib import Path

from .action_item_agent import ActionItemAgent
from .asr_agent import MeetingASRAgent
from .config import MeetingAISettings, get_settings
from .retrieval import MeetingVectorStore
from .schemas import LLMProvider, MeetingWorkflowResult, TranscriptResult
from .sentiment_agent import SentimentAgent
from .summary_agent import SummaryAgent
from .translation_agent import TranslationAgent


class SerialMeetingPipeline:
    def __init__(
        self,
        settings: MeetingAISettings | None = None,
        asr_agent: MeetingASRAgent | None = None,
        summary_agent: SummaryAgent | None = None,
        translation_agent: TranslationAgent | None = None,
        action_item_agent: ActionItemAgent | None = None,
        sentiment_agent: SentimentAgent | None = None,
        vector_store: MeetingVectorStore | None = None,
    ):
        self.settings = settings or get_settings()
        self.asr_agent = asr_agent or MeetingASRAgent(self.settings)
        self.summary_agent = summary_agent or SummaryAgent(self.settings)
        self.translation_agent = translation_agent or TranslationAgent(self.settings)
        self.action_item_agent = action_item_agent or ActionItemAgent(self.settings)
        self.sentiment_agent = sentiment_agent or SentimentAgent(self.settings)
        self.vector_store = vector_store or MeetingVectorStore(self.settings)

    def _configure_provider(self, provider: LLMProvider) -> None:
        for agent in [
            self.summary_agent,
            self.translation_agent,
            self.action_item_agent,
            self.sentiment_agent,
        ]:
            if hasattr(agent, "provider"):
                agent.provider = provider

    def run(
        self,
        audio_path: str | Path | None = None,
        transcript: TranscriptResult | None = None,
        language: str = "zh",
        provider: LLMProvider = LLMProvider.DEEPSEEK,
        selected_agents: list[str] | None = None,
        target_language: str = "en",
        glossary: dict[str, str] | None = None,
        sentiment_route: str = "llm",
        history_query: str | None = None,
        use_diarization: bool = True,
        num_speakers: int | None = None,
        persist_summary: bool = True,
        fail_fast: bool = True,
    ) -> MeetingWorkflowResult:
        if transcript is None and audio_path is None:
            raise ValueError("Either transcript or audio_path must be provided.")

        self._configure_provider(provider)
        started = time.perf_counter()
        selected = selected_agents if selected_agents is not None else ["summary", "translation", "action_items", "sentiment"]
        errors: dict[str, str] = {}

        resolved_transcript = transcript
        if resolved_transcript is None:
            resolved_transcript = self.asr_agent.transcribe(
                audio_path=str(Path(audio_path).expanduser().resolve()),
                language=language,
                use_diarization=use_diarization,
                num_speakers=num_speakers,
            )

        summary = None
        translation = None
        action_items = None
        sentiment = None
        history = []
        metadata: dict[str, object] = {"pipeline_mode": "serial"}

        def _should_stop() -> bool:
            return fail_fast and bool(errors)

        if "summary" in selected and not _should_stop():
            try:
                summary = self.summary_agent.summarize(transcript=resolved_transcript)
            except Exception as exc:
                errors["summary"] = str(exc)

        if "translation" in selected and not _should_stop():
            try:
                translation = self.translation_agent.translate(
                    source_language=language,
                    target_language=target_language,
                    transcript=resolved_transcript,
                    glossary=glossary or {},
                )
            except Exception as exc:
                errors["translation"] = str(exc)

        if "action_items" in selected and not _should_stop():
            try:
                action_items = self.action_item_agent.extract(transcript=resolved_transcript)
            except Exception as exc:
                errors["action_items"] = str(exc)

        if "sentiment" in selected and not _should_stop():
            try:
                sentiment = self.sentiment_agent.analyze(route=sentiment_route, transcript=resolved_transcript)
            except Exception as exc:
                errors["sentiment"] = str(exc)

        if history_query and not _should_stop():
            try:
                history = self.vector_store.query(history_query)
            except Exception as exc:
                errors["history"] = str(exc)

        if persist_summary and summary is not None:
            try:
                metadata["stored_meeting_id"] = self.vector_store.add_summary(
                    summary=summary,
                    transcript=resolved_transcript,
                    metadata={"source": "serial_pipeline"},
                )
            except Exception as exc:
                errors["storage"] = str(exc)

        latency = round(time.perf_counter() - started, 3)
        metadata["workflow_latency_seconds"] = latency

        return MeetingWorkflowResult(
            transcript=resolved_transcript,
            summary=summary,
            translation=translation,
            action_items=action_items,
            sentiment=sentiment,
            history=history,
            selected_agents=list(selected),
            errors=errors,
            metadata={"provider": provider.value, **metadata},
        )
