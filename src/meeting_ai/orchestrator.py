from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .action_item_agent import ActionItemAgent
from .asr_agent import MeetingASRAgent
from .config import MeetingAISettings, get_settings
from .retrieval import MeetingVectorStore
from .schemas import LLMProvider, MeetingWorkflowResult
from .sentiment_agent import SentimentAgent
from .summary_agent import SummaryAgent
from .translation_agent import TranslationAgent


def _merge_dicts(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    merged = dict(left)
    merged.update(right)
    return merged


class WorkflowState(TypedDict, total=False):
    audio_path: str
    language: str
    source_language: str
    target_language: str
    provider: str
    glossary: dict[str, str]
    selected_agents: list[str]
    sentiment_route: str
    history_query: str | None
    use_diarization: bool
    num_speakers: int | None
    enable_voiceprint: bool
    persist_summary: bool
    transcript: Any
    summary: Any
    translation: Any
    action_items: Any
    sentiment: Any
    history: list[Any]
    errors: Annotated[dict[str, str], _merge_dicts]
    metadata: dict[str, Any]


class MeetingOrchestrator:
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
        self.graph = self._build_graph().compile()

    def _configure_provider(self, provider: LLMProvider) -> None:
        for agent in [
            self.summary_agent,
            self.translation_agent,
            self.action_item_agent,
            self.sentiment_agent,
        ]:
            if hasattr(agent, "provider"):
                agent.provider = provider

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)
        workflow.add_node("asr", self._asr_node)
        workflow.add_node("summary", self._summary_node)
        workflow.add_node("translation", self._translation_node)
        workflow.add_node("action_items", self._action_items_node)
        workflow.add_node("sentiment", self._sentiment_node)
        workflow.add_node("history", self._history_node)
        workflow.add_node("aggregate", self._aggregate_node)

        workflow.add_edge(START, "asr")
        workflow.add_conditional_edges("asr", self._route_after_asr)
        workflow.add_edge("summary", "aggregate")
        workflow.add_edge("translation", "aggregate")
        workflow.add_edge("action_items", "aggregate")
        workflow.add_edge("sentiment", "aggregate")
        workflow.add_edge("history", "aggregate")
        workflow.add_edge("aggregate", END)
        return workflow

    def _route_after_asr(self, state: WorkflowState) -> list[str]:
        selected = set(state.get("selected_agents") or [])
        routes: list[str] = []
        if "summary" in selected:
            routes.append("summary")
        if "translation" in selected:
            routes.append("translation")
        if "action_items" in selected:
            routes.append("action_items")
        if "sentiment" in selected:
            routes.append("sentiment")
        if state.get("history_query"):
            routes.append("history")
        if not routes:
            routes.append("aggregate")
        return routes

    def _asr_node(self, state: WorkflowState) -> WorkflowState:
        transcript = self.asr_agent.transcribe(
            audio_path=state["audio_path"],
            language=state.get("language", "zh"),
            use_diarization=state.get("use_diarization", True),
            num_speakers=state.get("num_speakers"),
            enable_voiceprint=state.get("enable_voiceprint", False),
        )
        return {"transcript": transcript}

    def _summary_node(self, state: WorkflowState) -> WorkflowState:
        try:
            result = self.summary_agent.summarize(transcript=state["transcript"])
            return {"summary": result}
        except Exception as exc:
            return {"errors": {"summary": str(exc)}}

    def _translation_node(self, state: WorkflowState) -> WorkflowState:
        try:
            result = self.translation_agent.translate(
                source_language=state.get("source_language", state.get("language", "zh")),
                target_language=state.get("target_language", "en"),
                transcript=state["transcript"],
                glossary=state.get("glossary") or {},
            )
            return {"translation": result}
        except Exception as exc:
            return {"errors": {"translation": str(exc)}}

    def _action_items_node(self, state: WorkflowState) -> WorkflowState:
        try:
            result = self.action_item_agent.extract(transcript=state["transcript"])
            return {"action_items": result}
        except Exception as exc:
            return {"errors": {"action_items": str(exc)}}

    def _sentiment_node(self, state: WorkflowState) -> WorkflowState:
        try:
            result = self.sentiment_agent.analyze(
                route=state.get("sentiment_route", "llm"),
                transcript=state["transcript"],
            )
            return {"sentiment": result}
        except Exception as exc:
            return {"errors": {"sentiment": str(exc)}}

    def _history_node(self, state: WorkflowState) -> WorkflowState:
        try:
            question = state.get("history_query")
            if not question:
                return {"history": []}
            return {"history": self.vector_store.query(question)}
        except Exception as exc:
            return {"errors": {"history": str(exc)}, "history": []}

    def _aggregate_node(self, state: WorkflowState) -> WorkflowState:
        metadata = dict(state.get("metadata") or {})
        summary = state.get("summary")
        transcript = state.get("transcript")

        if state.get("persist_summary", True) and summary is not None:
            try:
                meeting_id = self.vector_store.add_summary(
                    summary=summary,
                    transcript=transcript,
                    metadata={"source": "orchestrator"},
                )
                metadata["stored_meeting_id"] = meeting_id
            except Exception as exc:
                return {"metadata": metadata, "errors": {"storage": str(exc)}}

        return {"metadata": metadata}

    def run(
        self,
        audio_path: str | Path,
        language: str = "zh",
        provider: LLMProvider = LLMProvider.DEEPSEEK,
        selected_agents: list[str] | None = None,
        target_language: str = "en",
        glossary: dict[str, str] | None = None,
        sentiment_route: str = "llm",
        history_query: str | None = None,
        use_diarization: bool = True,
        num_speakers: int | None = None,
        enable_voiceprint: bool = False,
        persist_summary: bool = True,
    ) -> MeetingWorkflowResult:
        started = time.perf_counter()
        self._configure_provider(provider)
        state: WorkflowState = {
            "audio_path": str(Path(audio_path).expanduser().resolve()),
            "language": language,
            "source_language": language,
            "target_language": target_language,
            "provider": provider.value,
            "glossary": glossary or {},
            "selected_agents": selected_agents if selected_agents is not None else ["summary", "translation", "action_items", "sentiment"],
            "sentiment_route": sentiment_route,
            "history_query": history_query,
            "use_diarization": use_diarization,
            "num_speakers": num_speakers,
            "enable_voiceprint": enable_voiceprint,
            "persist_summary": persist_summary,
            "errors": {},
            "metadata": {},
        }
        final_state = self.graph.invoke(state)
        latency = round(time.perf_counter() - started, 3)

        return MeetingWorkflowResult(
            transcript=final_state.get("transcript"),
            summary=final_state.get("summary"),
            translation=final_state.get("translation"),
            action_items=final_state.get("action_items"),
            sentiment=final_state.get("sentiment"),
            history=final_state.get("history") or [],
            selected_agents=final_state.get("selected_agents") or [],
            errors=final_state.get("errors") or {},
            metadata={
                "provider": provider.value,
                "workflow_latency_seconds": latency,
                **(final_state.get("metadata") or {}),
            },
        )


def _parse_glossary(entries: list[str] | None) -> dict[str, str]:
    glossary: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Invalid glossary entry: {entry}. Expected SOURCE=TARGET.")
        source, target = entry.split("=", 1)
        glossary[source.strip()] = target.strip()
    return glossary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Week 3 Meeting AI orchestrator.")
    parser.add_argument("--audio", required=True, help="Path to an input audio file.")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--provider", choices=[provider.value for provider in LLMProvider], default=LLMProvider.DEEPSEEK.value)
    parser.add_argument("--agent", action="append", dest="agents", help="Agent to run. Repeatable.")
    parser.add_argument("--target-language", default="en")
    parser.add_argument("--glossary", action="append", help="Glossary item in SOURCE=TARGET format.")
    parser.add_argument("--sentiment-route", choices=["llm", "transformer"], default="llm")
    parser.add_argument("--history-query", help="Optional query against stored meeting history.")
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--disable-diarization", action="store_true")
    parser.add_argument("--enable-voiceprint", action="store_true")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    orchestrator = MeetingOrchestrator(settings=get_settings())
    result = orchestrator.run(
        audio_path=args.audio,
        language=args.language,
        provider=LLMProvider(args.provider),
        selected_agents=args.agents,
        target_language=args.target_language,
        glossary=_parse_glossary(args.glossary),
        sentiment_route=args.sentiment_route,
        history_query=args.history_query,
        use_diarization=not args.disable_diarization,
        num_speakers=args.num_speakers,
        enable_voiceprint=args.enable_voiceprint,
    )
    payload = result.model_dump_json(indent=2)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
