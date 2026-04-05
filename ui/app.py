from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meeting_ai.config import get_settings
from meeting_ai.orchestrator import MeetingOrchestrator
from meeting_ai.runtime import find_ffmpeg
from meeting_ai.schemas import ActionItemResult, LLMProvider, MeetingWorkflowResult


APP_CSS = """
:root {
  --app-bg: linear-gradient(135deg, #eef6ff 0%, #f9f4ea 48%, #eef8f1 100%);
  --panel-bg: rgba(255, 255, 255, 0.85);
  --accent: #115e59;
  --accent-2: #b45309;
  --text-main: #0f172a;
}
body, .gradio-container {
  background: var(--app-bg);
  color: var(--text-main);
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}
.app-shell {
  backdrop-filter: blur(12px);
  background: var(--panel-bg);
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 24px;
  box-shadow: 0 24px 80px rgba(15, 23, 42, 0.12);
}
.hero-title {
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.03em;
}
.hero-note {
  color: #334155;
}
"""


def format_action_items(result: ActionItemResult | None) -> str:
    if result is None or not result.items:
        return "No action items extracted."

    lines = []
    for index, item in enumerate(result.items, start=1):
        assignee = item.assignee or "Unassigned"
        deadline = item.deadline or "No deadline"
        lines.append(f"{index}. [{item.priority.value}] {item.task} | owner: {assignee} | deadline: {deadline}")
        lines.append(f'   quote: "{item.source_quote}"')
    return "\n".join(lines)


def format_history(result: MeetingWorkflowResult) -> str:
    if not result.history:
        return "No history results."
    lines = []
    for index, item in enumerate(result.history, start=1):
        score = f"{item.score:.3f}" if item.score is not None else "n/a"
        lines.append(f"{index}. meeting={item.meeting_id} score={score}")
        lines.append(item.document)
    return "\n\n".join(lines)


def format_summary(result: MeetingWorkflowResult) -> str:
    if result.summary is None:
        return "No summary generated."
    lines = ["Topics:"]
    lines.extend(f"- {item}" for item in result.summary.topics or ["None"])
    lines.append("")
    lines.append("Decisions:")
    lines.extend(f"- {item}" for item in result.summary.decisions or ["None"])
    lines.append("")
    lines.append("Follow-ups:")
    lines.extend(f"- {item}" for item in result.summary.follow_ups or ["None"])
    return "\n".join(lines)


def run_pipeline(
    audio_path: str | None,
    language: str,
    provider: str,
    selected_agents: list[str],
    target_language: str,
    sentiment_route: str,
    history_query: str,
    glossary_text: str,
    use_diarization: bool,
    num_speakers: float | None,
    progress: gr.Progress = gr.Progress(),
):
    if not audio_path:
        raise gr.Error("Upload an audio file first.")

    settings = get_settings()
    orchestrator = MeetingOrchestrator(settings=settings)
    glossary: dict[str, str] = {}
    for raw_line in glossary_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise gr.Error(f"Invalid glossary entry: {line}. Use SOURCE=TARGET.")
        source, target = line.split("=", 1)
        glossary[source.strip()] = target.strip()

    progress(0.05, desc="Starting workflow")
    result = orchestrator.run(
        audio_path=audio_path,
        language=language,
        provider=LLMProvider(provider),
        selected_agents=selected_agents,
        target_language=target_language,
        glossary=glossary,
        sentiment_route=sentiment_route,
        history_query=history_query.strip() or None,
        use_diarization=use_diarization,
        num_speakers=int(num_speakers) if num_speakers else None,
    )
    progress(1.0, desc="Completed")

    transcript_text = result.transcript.full_text if result.transcript is not None else "No transcript."
    translation_text = result.translation.full_text if result.translation is not None else "No translation generated."
    sentiment_json = json.dumps(
        result.sentiment.model_dump() if result.sentiment is not None else {},
        ensure_ascii=False,
        indent=2,
    )
    transcript_json = json.dumps(
        result.transcript.model_dump() if result.transcript is not None else {},
        ensure_ascii=False,
        indent=2,
    )
    metadata_json = json.dumps(result.metadata, ensure_ascii=False, indent=2)
    errors_json = json.dumps(result.errors, ensure_ascii=False, indent=2)

    status = (
        f"Workflow finished in {result.metadata.get('workflow_latency_seconds', 'n/a')}s | "
        f"ffmpeg={'yes' if find_ffmpeg() else 'no'} | "
        f"errors={len(result.errors)}"
    )
    return (
        status,
        transcript_text,
        transcript_json,
        format_summary(result),
        format_action_items(result.action_items),
        translation_text,
        sentiment_json,
        format_history(result),
        metadata_json,
        errors_json,
    )


def build_app() -> gr.Blocks:
    settings = get_settings()
    with gr.Blocks(css=APP_CSS, title="Meeting AI Week 3") as app:
        with gr.Column(elem_classes=["app-shell"]):
            gr.Markdown(
                """
                <div class="hero-title">Meeting AI</div>
                <div class="hero-note">Week 3 workflow: ASR -> parallel agents -> retrieval -> aggregated review.</div>
                """,
            )

            with gr.Row():
                audio = gr.Audio(label="Audio", type="filepath", sources=["upload", "microphone"])
                with gr.Column():
                    language = gr.Dropdown(choices=["zh", "en"], value="zh", label="Source Language")
                    provider = gr.Dropdown(
                        choices=[provider.value for provider in LLMProvider],
                        value=LLMProvider.DEEPSEEK.value,
                        label="LLM Provider",
                    )
                    target_language = gr.Dropdown(choices=["en", "zh"], value="en", label="Translation Target")
                    sentiment_route = gr.Dropdown(choices=["llm", "transformer"], value="llm", label="Sentiment Route")

            with gr.Row():
                selected_agents = gr.CheckboxGroup(
                    choices=["summary", "translation", "action_items", "sentiment"],
                    value=["summary", "translation", "action_items", "sentiment"],
                    label="Agents",
                )
                with gr.Column():
                    use_diarization = gr.Checkbox(value=True, label="Enable Diarization")
                    num_speakers = gr.Number(value=None, precision=0, label="Known Speaker Count")
                    history_query = gr.Textbox(label="History Query", placeholder="上次这个问题是怎么决定的？")
                    glossary_text = gr.Textbox(
                        label="Glossary",
                        lines=4,
                        placeholder="语音识别=speech-recognition\n预算=budget",
                    )

            run_button = gr.Button("Run Workflow", variant="primary")
            status = gr.Markdown(f"ffmpeg: {find_ffmpeg() or 'not found'} | server: {settings.gradio_server_name}:{settings.gradio_server_port}")

            with gr.Tabs():
                with gr.Tab("Transcript"):
                    transcript_text = gr.Textbox(lines=14, label="Transcript")
                    transcript_json = gr.Code(language="json", label="Transcript JSON")
                with gr.Tab("Summary"):
                    summary_text = gr.Textbox(lines=14, label="Summary")
                with gr.Tab("Action Items"):
                    action_items_text = gr.Textbox(lines=14, label="Action Items")
                with gr.Tab("Translation"):
                    translation_text = gr.Textbox(lines=14, label="Translation")
                with gr.Tab("Sentiment"):
                    sentiment_json = gr.Code(language="json", label="Sentiment JSON")
                with gr.Tab("History"):
                    history_text = gr.Textbox(lines=14, label="History Retrieval")
                with gr.Tab("Diagnostics"):
                    metadata_json = gr.Code(language="json", label="Metadata")
                    errors_json = gr.Code(language="json", label="Errors")

            run_button.click(
                fn=run_pipeline,
                inputs=[
                    audio,
                    language,
                    provider,
                    selected_agents,
                    target_language,
                    sentiment_route,
                    history_query,
                    glossary_text,
                    use_diarization,
                    num_speakers,
                ],
                outputs=[
                    status,
                    transcript_text,
                    transcript_json,
                    summary_text,
                    action_items_text,
                    translation_text,
                    sentiment_json,
                    history_text,
                    metadata_json,
                    errors_json,
                ],
            )
    return app


def main() -> None:
    settings = get_settings()
    app = build_app()
    app.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        share=False,
    )


if __name__ == "__main__":
    main()
