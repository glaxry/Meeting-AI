from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meeting_ai.config import get_settings
from meeting_ai.runtime import find_ffmpeg
from meeting_ai.schemas import ActionItemResult, LLMProvider, MeetingWorkflowResult, SentimentResult, TranslationResult
from meeting_ai.streaming import FunASRStreamingTranscriber, StreamingSessionRegistry


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


STREAMING_SESSIONS = StreamingSessionRegistry()


@lru_cache(maxsize=1)
def get_streaming_transcriber() -> FunASRStreamingTranscriber:
    return FunASRStreamingTranscriber(settings=get_settings())


def _is_selected(selected_agents: list[str] | None, agent_name: str) -> bool:
    return selected_agents is None or agent_name in selected_agents


def format_transcript(result: MeetingWorkflowResult) -> str:
    if result.transcript is None:
        return "No transcript generated."

    lines = []
    for segment in result.transcript.segments:
        annotations: list[str] = []
        if segment.emotion:
            annotations.append(f"emotion={segment.emotion}")
        if segment.event:
            annotations.append(f"event={segment.event}")
        speaker_confidence = segment.metadata.get("speaker_confidence")
        if speaker_confidence:
            annotations.append(f"speaker_confidence={speaker_confidence}")
        suffix = f" | {' | '.join(annotations)}" if annotations else ""
        lines.append(f"{segment.start:>7.2f}s - {segment.end:>7.2f}s | {segment.speaker} | {segment.text}{suffix}")
    return "\n".join(lines) or result.transcript.full_text or "No transcript text."


def format_action_items(result: ActionItemResult | None, selected_agents: list[str] | None = None) -> str:
    if not _is_selected(selected_agents, "action_items"):
        return "Action item extraction was not selected."
    if result is None or not result.items:
        return "No action items extracted."

    lines = []
    for index, item in enumerate(result.items, start=1):
        assignee = item.assignee or "Unassigned"
        deadline = item.deadline or "No deadline"
        lines.append(f"{index}. [{item.priority.value}] {item.task} | owner: {assignee} | deadline: {deadline}")
        lines.append(f'   quote: "{item.source_quote}"')
    return "\n".join(lines)


def format_translation(result: TranslationResult | None, selected_agents: list[str] | None = None) -> str:
    if not _is_selected(selected_agents, "translation"):
        return "Translation was not selected."
    if result is None:
        return "No translation generated."

    lines = [
        f"Source language: {result.source_language}",
        f"Target language: {result.target_language}",
        "",
    ]
    for segment in result.segments:
        lines.append(f"{segment.start:>7.2f}s - {segment.end:>7.2f}s | {segment.speaker} | {segment.text}")
    return "\n".join(lines).strip() or result.full_text or "No translation generated."


def format_sentiment(result: SentimentResult | None, selected_agents: list[str] | None = None) -> str:
    if not _is_selected(selected_agents, "sentiment"):
        return "Sentiment analysis was not selected."
    if result is None:
        return "No sentiment analysis generated."

    counts: dict[str, int] = {}
    for segment in result.segments:
        counts[segment.sentiment.value] = counts.get(segment.sentiment.value, 0) + 1

    lines = [
        f"Route: {result.route}",
        f"Overall tone: {result.overall_tone.value}",
        f"Segments analyzed: {len(result.segments)}",
        "",
        "Label distribution:",
    ]
    if counts:
        lines.extend(f"- {label}: {count}" for label, count in sorted(counts.items()))
    else:
        lines.append("- None")

    lines.append("")
    lines.append("Segment details:")
    for segment in result.segments:
        speaker = segment.speaker or "UNKNOWN"
        timing = ""
        if segment.start is not None and segment.end is not None:
            timing = f"{segment.start:>7.2f}s - {segment.end:>7.2f}s | "
        lines.append(
            f"{timing}{speaker} | {segment.sentiment.value} ({segment.confidence:.2f}) | {segment.text}"
        )
    if result.timeline:
        lines.append("")
        lines.append("Timeline snapshots:")
        for snapshot in result.timeline:
            distribution = ", ".join(
                f"{label}={value:.3f}"
                for label, value in snapshot.label_distribution.items()
            )
            speakers = ", ".join(snapshot.speakers_involved) or "UNKNOWN"
            lines.append(
                f"{snapshot.window_start:>7.2f}s - {snapshot.window_end:>7.2f}s | "
                f"{snapshot.dominant_label.value} | speakers={speakers} | {distribution}"
            )
    return "\n".join(lines)


def build_sentiment_chart(result: SentimentResult | None):
    if result is None:
        return None

    label_map = {
        "agreement": 1.0,
        "neutral": 0.0,
        "hesitation": -0.5,
        "disagreement": -1.0,
        "tension": -2.0,
    }
    rows: list[dict[str, object]] = []

    if result.timeline:
        for snapshot in result.timeline:
            midpoint = round((snapshot.window_start + snapshot.window_end) / 2, 3)
            rows.append(
                {
                    "time_seconds": midpoint,
                    "sentiment_score": label_map.get(snapshot.dominant_label.value, 0.0),
                    "label": snapshot.dominant_label.value,
                }
            )
    else:
        for index, segment in enumerate(result.segments):
            time_seconds = segment.start if segment.start is not None else float(index)
            rows.append(
                {
                    "time_seconds": round(float(time_seconds), 3),
                    "sentiment_score": label_map.get(segment.sentiment.value, 0.0),
                    "label": segment.sentiment.value,
                }
            )

    if not rows:
        return None
    return pd.DataFrame(rows)


def build_speaker_distribution_chart(result: MeetingWorkflowResult):
    transcript = result.transcript
    if transcript is None or not transcript.segments:
        return None

    distribution: dict[str, dict[str, float | int]] = {}
    for segment in transcript.segments:
        speaker = segment.speaker or "UNKNOWN"
        duration = max(segment.end - segment.start, 0.0)
        speaker_bucket = distribution.setdefault(
            speaker,
            {
                "speaker": speaker,
                "duration_seconds": 0.0,
                "segment_count": 0,
            },
        )
        speaker_bucket["duration_seconds"] = round(float(speaker_bucket["duration_seconds"]) + duration, 3)
        speaker_bucket["segment_count"] = int(speaker_bucket["segment_count"]) + 1

    if not distribution:
        return None
    rows = sorted(distribution.values(), key=lambda item: (-float(item["duration_seconds"]), str(item["speaker"])))
    return pd.DataFrame(rows)


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
    if not _is_selected(result.selected_agents, "summary"):
        return "Summary was not selected."
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


def format_diagnostics(result: MeetingWorkflowResult) -> str:
    lines = [
        f"Workflow latency: {result.metadata.get('workflow_latency_seconds', 'n/a')}s",
        f"Provider: {result.metadata.get('provider', 'n/a')}",
        f"Stored meeting id: {result.metadata.get('stored_meeting_id', 'not stored')}",
        f"Selected agents: {', '.join(result.selected_agents or []) or 'none'}",
        f"ffmpeg available: {'yes' if find_ffmpeg() else 'no'}",
        "",
        "Errors:",
    ]
    if result.errors:
        lines.extend(f"- {name}: {message}" for name, message in result.errors.items())
    else:
        lines.append("- None")

    if result.transcript is not None:
        lines.extend(
            [
                "",
                "Transcript metadata:",
                f"- audio duration: {result.transcript.metadata.get('audio_duration_seconds', 'n/a')}s",
                f"- ASR runtime: {result.transcript.metadata.get('asr_runtime_seconds', 'n/a')}s",
                f"- diarization runtime: {result.transcript.metadata.get('diarization_runtime_seconds', 'n/a')}s",
                f"- diarization backend: {result.transcript.diarization_backend}",
                f"- low-confidence speaker assignments: {result.transcript.metadata.get('speaker_confidence_low_count', 0)}",
            ]
        )
    return "\n".join(lines)


def parse_glossary(glossary_text: str) -> dict[str, str]:
    glossary: dict[str, str] = {}
    for raw_line in glossary_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise gr.Error(f"Invalid glossary entry: {line}. Use SOURCE=TARGET.")
        source, target = line.split("=", 1)
        glossary[source.strip()] = target.strip()
    return glossary


def analyze_via_api(
    audio_path: str,
    language: str,
    provider: str,
    selected_agents: list[str],
    target_language: str,
    sentiment_route: str,
    history_query: str,
    glossary: dict[str, str],
    use_diarization: bool,
    num_speakers: float | None,
) -> MeetingWorkflowResult:
    settings = get_settings()
    url = f"{settings.api_base_url.rstrip('/')}/meetings/analyze"
    data = {
        "language": language,
        "provider": provider,
        "selected_agents": json.dumps(selected_agents or [], ensure_ascii=False),
        "target_language": target_language,
        "sentiment_route": sentiment_route,
        "history_query": history_query.strip(),
        "glossary": json.dumps(glossary, ensure_ascii=False),
        "use_diarization": str(bool(use_diarization)).lower(),
    }
    if num_speakers is not None:
        data["num_speakers"] = str(int(num_speakers))
    try:
        with Path(audio_path).open("rb") as audio_file:
            response = requests.post(
                url,
                data=data,
                files={"audio": (Path(audio_path).name, audio_file, "application/octet-stream")},
                timeout=None,
            )
    except requests.RequestException as exc:
        raise gr.Error(f"Meeting AI API request failed. Start the FastAPI backend first: {exc}") from exc

    if response.status_code >= 400:
        raise gr.Error(f"Meeting AI API returned {response.status_code}: {response.text}")
    return MeetingWorkflowResult.model_validate(response.json())


def start_streaming_demo(language: str) -> tuple[str, str, dict[str, Any]]:
    session = STREAMING_SESSIONS.add(get_streaming_transcriber().create_session(language=language))
    info = session.session_info()
    status = (
        f"Streaming session ready | id={info.session_id} | model={info.asr_model} | "
        f"chunk_size={info.chunk_size} | target_sample_rate={info.target_sample_rate}"
    )
    return "", status, {"session_id": info.session_id, "language": language}


def process_streaming_audio(
    audio_chunk: tuple[int, Any] | None,
    language: str,
    state: dict[str, Any] | None,
) -> tuple[str, str, dict[str, Any] | None]:
    if audio_chunk is None:
        return "", "Waiting for microphone audio...", state
    if not isinstance(audio_chunk, tuple) or len(audio_chunk) != 2:
        raise gr.Error("Streaming audio chunk is invalid. Expected a (sample_rate, samples) tuple.")

    sample_rate, samples = audio_chunk
    state = state or {}
    session_id = str(state.get("session_id", "")).strip()
    session = STREAMING_SESSIONS.get(session_id) if session_id else None
    if session is None:
        session = STREAMING_SESSIONS.add(
            get_streaming_transcriber().create_session(language=language, sample_rate=int(sample_rate))
        )
        state = {"session_id": session.session_id, "language": language}

    event = session.process_chunk(samples, sample_rate=int(sample_rate), is_final=False)
    status = (
        f"Streaming live | id={event.session_id} | chunk_index={event.chunk_index} | "
        f"received={event.received_seconds:.2f}s | delta_chars={len(event.delta_text)}"
    )
    transcript = event.cumulative_text or "Listening..."
    return transcript, status, state


def stop_streaming_demo(state: dict[str, Any] | None) -> tuple[str, str, None]:
    if not state:
        return "", "Streaming session stopped.", None
    session_id = str(state.get("session_id", "")).strip()
    if not session_id:
        return "", "Streaming session stopped.", None

    session = STREAMING_SESSIONS.pop(session_id)
    if session is None:
        return "", "Streaming session already cleaned up.", None

    event = session.snapshot(is_final=True)
    status = (
        f"Streaming finished | id={event.session_id} | processed={event.received_seconds:.2f}s | "
        f"transcript_chars={len(event.cumulative_text)}"
    )
    return event.cumulative_text or "No transcript emitted.", status, None


def reset_streaming_demo(state: dict[str, Any] | None) -> tuple[None, str, str, None]:
    session_id = str((state or {}).get("session_id", "")).strip()
    if session_id:
        STREAMING_SESSIONS.pop(session_id)
    return None, "", "Streaming session reset.", None


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

    glossary = parse_glossary(glossary_text)

    progress(0.05, desc="Calling FastAPI backend")
    result = analyze_via_api(
        audio_path,
        language,
        provider,
        selected_agents,
        target_language,
        sentiment_route,
        history_query,
        glossary,
        use_diarization,
        num_speakers,
    )
    progress(1.0, desc="Completed")

    status = (
        f"Workflow finished in {result.metadata.get('workflow_latency_seconds', 'n/a')}s | "
        f"ffmpeg={'yes' if find_ffmpeg() else 'no'} | "
        f"errors={len(result.errors)}"
    )
    return (
        status,
        format_transcript(result),
        format_summary(result),
        format_action_items(result.action_items, result.selected_agents),
        format_translation(result.translation, result.selected_agents),
        format_sentiment(result.sentiment, result.selected_agents),
        build_sentiment_chart(result.sentiment),
        build_speaker_distribution_chart(result),
        format_history(result),
        format_diagnostics(result),
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
                    speaker_distribution_chart = gr.BarPlot(
                        x="speaker",
                        y="duration_seconds",
                        title="Speaker Participation",
                        x_title="Speaker",
                        y_title="Duration (s)",
                    )
                with gr.Tab("Summary"):
                    summary_text = gr.Textbox(lines=14, label="Summary")
                with gr.Tab("Action Items"):
                    action_items_text = gr.Textbox(lines=14, label="Action Items")
                with gr.Tab("Translation"):
                    translation_text = gr.Textbox(lines=14, label="Translation")
                with gr.Tab("Sentiment"):
                    sentiment_text = gr.Textbox(lines=14, label="Sentiment")
                    sentiment_chart = gr.LinePlot(
                        x="time_seconds",
                        y="sentiment_score",
                        title="Sentiment Timeline",
                        x_title="Time (s)",
                        y_title="Sentiment score",
                    )
                with gr.Tab("History"):
                    history_text = gr.Textbox(lines=14, label="History Retrieval")
                with gr.Tab("Diagnostics"):
                    diagnostics_text = gr.Textbox(lines=14, label="Diagnostics")

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
                    summary_text,
                    action_items_text,
                    translation_text,
                    sentiment_text,
                    sentiment_chart,
                    speaker_distribution_chart,
                    history_text,
                    diagnostics_text,
                ],
            )

            with gr.Accordion("Streaming MVP", open=False):
                gr.Markdown(
                    """
                    Live ASR MVP for microphone input. This path uses the FunASR streaming model locally and is separate from the
                    batch Week 3 workflow above. For application integrations, use the FastAPI WebSocket endpoint at
                    `/stream/transcribe`.
                    """
                )
                with gr.Row():
                    stream_language = gr.Dropdown(choices=["zh", "en"], value="zh", label="Streaming Language")
                    stream_reset = gr.Button("Reset Stream")
                stream_audio = gr.Audio(
                    label="Live Audio",
                    type="numpy",
                    sources=["microphone"],
                    streaming=True,
                )
                stream_status = gr.Markdown(
                    f"Ready for live ASR | model={settings.funasr_streaming_model} | "
                    f"stream_every={settings.streaming_gradio_chunk_seconds:.1f}s"
                )
                stream_transcript = gr.Textbox(lines=10, label="Live Transcript")
                stream_state = gr.State(value=None)

                stream_audio.start_recording(
                    fn=start_streaming_demo,
                    inputs=[stream_language],
                    outputs=[stream_transcript, stream_status, stream_state],
                    show_progress="hidden",
                )
                stream_audio.stream(
                    fn=process_streaming_audio,
                    inputs=[stream_audio, stream_language, stream_state],
                    outputs=[stream_transcript, stream_status, stream_state],
                    show_progress="hidden",
                    stream_every=settings.streaming_gradio_chunk_seconds,
                )
                stream_audio.stop_recording(
                    fn=stop_streaming_demo,
                    inputs=[stream_state],
                    outputs=[stream_transcript, stream_status, stream_state],
                    show_progress="hidden",
                )
                stream_reset.click(
                    fn=reset_streaming_demo,
                    inputs=[stream_state],
                    outputs=[stream_audio, stream_transcript, stream_status, stream_state],
                    show_progress="hidden",
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
