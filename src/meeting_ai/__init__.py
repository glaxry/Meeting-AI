from __future__ import annotations

from typing import TYPE_CHECKING

from .runtime import ensure_runtime_paths


ensure_runtime_paths()


__all__ = [
    "ActionItemAgent",
    "MeetingASRAgent",
    "MeetingOrchestrator",
    "MeetingVectorStore",
    "SerialMeetingPipeline",
    "SentimentAgent",
    "SummaryAgent",
    "TranslationAgent",
    "UnifiedLLMClient",
    "Week5Artifacts",
    "export_week5_materials",
]


_EXPORTS = {
    "ActionItemAgent": (".action_item_agent", "ActionItemAgent"),
    "MeetingASRAgent": (".asr_agent", "MeetingASRAgent"),
    "MeetingOrchestrator": (".orchestrator", "MeetingOrchestrator"),
    "MeetingVectorStore": (".retrieval", "MeetingVectorStore"),
    "SerialMeetingPipeline": (".baseline", "SerialMeetingPipeline"),
    "SentimentAgent": (".sentiment_agent", "SentimentAgent"),
    "SummaryAgent": (".summary_agent", "SummaryAgent"),
    "TranslationAgent": (".translation_agent", "TranslationAgent"),
    "UnifiedLLMClient": (".llm_tools", "UnifiedLLMClient"),
    "Week5Artifacts": (".final_materials", "Week5Artifacts"),
    "export_week5_materials": (".final_materials", "export_week5_materials"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _EXPORTS[name]
    module = __import__(f"{__name__}{module_name}", fromlist=[attribute_name])
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from .action_item_agent import ActionItemAgent
    from .asr_agent import MeetingASRAgent
    from .baseline import SerialMeetingPipeline
    from .final_materials import Week5Artifacts, export_week5_materials
    from .llm_tools import UnifiedLLMClient
    from .orchestrator import MeetingOrchestrator
    from .retrieval import MeetingVectorStore
    from .sentiment_agent import SentimentAgent
    from .summary_agent import SummaryAgent
    from .translation_agent import TranslationAgent
