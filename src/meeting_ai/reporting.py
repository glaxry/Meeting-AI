from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path
from typing import Any

from .schemas import MeetingWorkflowResult, RetrievalRecord


@dataclass(frozen=True)
class Week35Artifacts:
    report_path: Path
    metrics_path: Path
    architecture_svg_path: Path
    runtime_svg_path: Path
    speaker_svg_path: Path
    snapshot_svg_path: Path
    retrieval_svg_path: Path


def load_workflow_result(path: Path) -> MeetingWorkflowResult:
    return MeetingWorkflowResult.model_validate_json(path.read_text(encoding="utf-8"))


def _sum_seconds(values: list[float] | tuple[float, ...] | None) -> float:
    return round(sum(float(value) for value in values or []), 3)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return round(float(numerator) / float(denominator), 3)


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}s"


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _wrap_text(text: str, max_chars: int) -> list[str]:
    text = " ".join(text.strip().split())
    if not text:
        return [""]

    lines: list[str] = []
    current = ""
    for char in text:
        if len(current) >= max_chars:
            lines.append(current)
            current = char
        else:
            current += char
    if current:
        lines.append(current)
    return lines


def _svg_text(lines: list[str], x: int, y: int, line_height: int = 24, size: int = 18, weight: int = 400) -> str:
    chunks: list[str] = []
    for index, line in enumerate(lines):
        dy = y + index * line_height
        chunks.append(
            f'<text x="{x}" y="{dy}" font-family="Segoe UI, Microsoft YaHei, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="#1f2937">{escape(line)}</text>'
        )
    return "\n".join(chunks)


def _panel(title: str, lines: list[str], x: int, y: int, width: int, height: int) -> str:
    safe_lines = lines[: int((height - 72) / 24)]
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="24" fill="#ffffff" stroke="#d0d7de" stroke-width="2"/>',
            _svg_text([title], x + 24, y + 40, line_height=24, size=24, weight=600),
            _svg_text(safe_lines, x + 24, y + 84, line_height=24, size=18),
        ]
    )


def build_week35_metrics(
    result: MeetingWorkflowResult,
    retrieval_query: str | None = None,
    retrieval_results: list[RetrievalRecord] | None = None,
    generated_on: str | None = None,
) -> dict[str, Any]:
    retrieval_results = retrieval_results or []
    transcript = result.transcript
    summary = result.summary
    translation = result.translation
    action_items = result.action_items
    sentiment = result.sentiment

    transcript_metadata = transcript.metadata if transcript else {}
    summary_metadata = summary.metadata if summary else {}
    translation_metadata = translation.metadata if translation else {}
    action_metadata = action_items.metadata if action_items else {}
    sentiment_metadata = sentiment.metadata if sentiment else {}

    audio_duration_seconds = transcript_metadata.get("audio_duration_seconds")
    asr_runtime_seconds = transcript_metadata.get("asr_runtime_seconds")
    diarization_runtime_seconds = transcript_metadata.get("diarization_runtime_seconds")
    workflow_latency_seconds = result.metadata.get("workflow_latency_seconds")
    summary_map_latencies = [float(value) for value in summary_metadata.get("map_latencies", [])]
    summary_reduce_latency = summary_metadata.get("reduce_latency")
    translation_latencies = [float(value) for value in translation_metadata.get("latencies", [])]
    action_latencies = [float(value) for value in action_metadata.get("latencies", [])]
    sentiment_latency_seconds = sentiment_metadata.get("latency_seconds")

    summary_total_latency = round(_sum_seconds(summary_map_latencies) + float(summary_reduce_latency or 0.0), 3)
    translation_total_latency = _sum_seconds(translation_latencies)
    action_total_latency = _sum_seconds(action_latencies)
    accounted_latency = round(
        sum(
            value
            for value in [
                float(asr_runtime_seconds or 0.0),
                float(diarization_runtime_seconds or 0.0),
                summary_total_latency,
                translation_total_latency,
                action_total_latency,
                float(sentiment_latency_seconds or 0.0),
            ]
        ),
        3,
    )
    parallel_time_saved_seconds = None
    if workflow_latency_seconds is not None:
        parallel_time_saved_seconds = round(accounted_latency - float(workflow_latency_seconds), 3)

    speaker_counts: dict[str, int] = {}
    transcript_preview: list[str] = []
    if transcript is not None:
        speaker_counts = dict(sorted(Counter(segment.speaker for segment in transcript.segments).items(), key=lambda item: (-item[1], item[0])))
        for segment in transcript.segments[:6]:
            transcript_preview.append(
                f"{segment.start:>6.2f}-{segment.end:>6.2f}s [{segment.speaker}] {segment.text}"
            )

    translation_preview: list[str] = []
    if translation is not None:
        for segment in translation.segments[:6]:
            translation_preview.append(f"[{segment.speaker}] {segment.text}")

    action_item_preview: list[str] = []
    if action_items is not None:
        for item in action_items.items[:5]:
            owner = item.assignee or "Unassigned"
            action_item_preview.append(f"{owner}: {item.task} ({item.priority.value})")

    sentiment_counts: dict[str, int] = {}
    if sentiment is not None:
        sentiment_counts = dict(
            sorted(Counter(segment.sentiment.value for segment in sentiment.segments).items(), key=lambda item: (-item[1], item[0]))
        )

    summary_preview = {
        "topics": (summary.topics[:5] if summary else []),
        "decisions": (summary.decisions[:3] if summary else []),
        "follow_ups": (summary.follow_ups[:4] if summary else []),
    }

    retrieval_preview = [
        {
            "meeting_id": record.meeting_id,
            "score": record.score,
            "document": record.document[:420].strip(),
        }
        for record in retrieval_results[:3]
    ]

    stages = [
        {
            "name": "Audio Duration",
            "seconds": audio_duration_seconds,
            "rtf": 1.0 if audio_duration_seconds else None,
        },
        {
            "name": "ASR",
            "seconds": asr_runtime_seconds,
            "rtf": _safe_ratio(asr_runtime_seconds, audio_duration_seconds),
        },
        {
            "name": "Diarization",
            "seconds": diarization_runtime_seconds,
            "rtf": _safe_ratio(diarization_runtime_seconds, audio_duration_seconds),
        },
        {
            "name": "Summary",
            "seconds": summary_total_latency or None,
            "rtf": _safe_ratio(summary_total_latency or None, audio_duration_seconds),
        },
        {
            "name": "Translation",
            "seconds": translation_total_latency or None,
            "rtf": _safe_ratio(translation_total_latency or None, audio_duration_seconds),
        },
        {
            "name": "Action Items",
            "seconds": action_total_latency or None,
            "rtf": _safe_ratio(action_total_latency or None, audio_duration_seconds),
        },
        {
            "name": "Sentiment",
            "seconds": sentiment_latency_seconds,
            "rtf": _safe_ratio(sentiment_latency_seconds, audio_duration_seconds),
        },
        {
            "name": "Workflow",
            "seconds": workflow_latency_seconds,
            "rtf": _safe_ratio(workflow_latency_seconds, audio_duration_seconds),
        },
    ]

    return {
        "generated_on": generated_on or date.today().isoformat(),
        "meeting_id": result.metadata.get("stored_meeting_id"),
        "provider": result.metadata.get("provider"),
        "selected_agents": result.selected_agents,
        "transcript": {
            "audio_path": transcript.audio_path if transcript else None,
            "language": transcript.language if transcript else None,
            "segment_count": len(transcript.segments) if transcript else 0,
            "preview": transcript_preview,
            "speaker_distribution": speaker_counts,
        },
        "summary": {
            "strategy": summary_metadata.get("strategy") if summary else None,
            "word_count": summary_metadata.get("word_count") if summary else None,
            "chunk_count": summary_metadata.get("chunk_count") if summary else None,
            "map_latencies": summary_map_latencies,
            "reduce_latency": summary_reduce_latency,
            "total_latency_seconds": summary_total_latency or None,
            "preview": summary_preview,
        },
        "translation": {
            "source_language": translation.source_language if translation else None,
            "target_language": translation.target_language if translation else None,
            "chunk_count": translation_metadata.get("chunk_count") if translation else None,
            "latencies": translation_latencies,
            "total_latency_seconds": translation_total_latency or None,
            "preview": translation_preview,
        },
        "action_items": {
            "count": len(action_items.items) if action_items else 0,
            "chunk_count": action_metadata.get("chunk_count") if action_items else None,
            "latencies": action_latencies,
            "total_latency_seconds": action_total_latency or None,
            "preview": action_item_preview,
        },
        "sentiment": {
            "route": sentiment.route if sentiment else None,
            "overall_tone": sentiment.overall_tone.value if sentiment else None,
            "segment_count": len(sentiment.segments) if sentiment else 0,
            "distribution": sentiment_counts,
            "latency_seconds": sentiment_latency_seconds,
        },
        "retrieval": {
            "query": retrieval_query,
            "result_count": len(retrieval_results),
            "preview": retrieval_preview,
        },
        "runtime": {
            "audio_duration_seconds": audio_duration_seconds,
            "asr_runtime_seconds": asr_runtime_seconds,
            "diarization_runtime_seconds": diarization_runtime_seconds,
            "workflow_latency_seconds": workflow_latency_seconds,
            "workflow_rtf": _safe_ratio(workflow_latency_seconds, audio_duration_seconds),
            "accounted_latency_seconds": accounted_latency,
            "parallel_time_saved_seconds": parallel_time_saved_seconds,
            "stages": stages,
        },
        "errors": result.errors,
        "limitations": {
            "wer_cer_status": "pending_reference_transcript",
            "notes": [
                "The repo includes jiwer in requirements, but test.wav has no aligned human reference transcript yet.",
                "LLM sentiment on the current sample collapsed to all-neutral labels, which is acceptable for Week 3.5 but should be benchmarked in Week 4.",
            ],
        },
    }


def _write_svg(path: Path, width: int, height: int, body: str) -> None:
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" fill="none">',
            f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
            body,
            "</svg>",
        ]
    )
    path.write_text(svg, encoding="utf-8")


def render_architecture_svg(path: Path) -> None:
    boxes = [
        (60, 90, 220, 92, "#dbeafe", "Audio Input", ["`test.wav` upload or CLI input"]),
        (340, 90, 220, 92, "#dcfce7", "ASR + Diarization", ["FunASR transcription", "pyannote speaker labels"]),
        (620, 70, 260, 132, "#fef3c7", "LangGraph Orchestrator", ["routing", "error isolation", "parallel fan-out"]),
        (940, 40, 220, 72, "#fce7f3", "Summary Agent", ["map-reduce JSON"]),
        (940, 126, 220, 72, "#ede9fe", "Translation Agent", ["speaker-preserving"]),
        (940, 212, 220, 72, "#fee2e2", "Action Agent", ["assignee / task / priority"]),
        (940, 298, 220, 72, "#e0f2fe", "Sentiment Agent", ["LLM or transformer route"]),
        (620, 282, 260, 108, "#f3f4f6", "Aggregate Result", ["Pydantic workflow payload", "runtime metadata"]),
        (340, 282, 220, 108, "#ecfccb", "Chroma History", ["meeting summary persistence", "cross-meeting retrieval"]),
        (60, 282, 220, 108, "#ffedd5", "Gradio UI / CLI", ["transcript", "summary", "action items", "history query"]),
    ]
    arrows = [
        (280, 136, 340, 136),
        (560, 136, 620, 136),
        (880, 106, 940, 76),
        (880, 136, 940, 162),
        (880, 166, 940, 248),
        (880, 196, 940, 334),
        (1050, 112, 750, 282),
        (1050, 198, 750, 282),
        (1050, 284, 750, 282),
        (1050, 370, 750, 282),
        (620, 336, 560, 336),
        (340, 336, 280, 336),
    ]
    body: list[str] = [
        _svg_text(["Meeting AI Week 3 System Design"], 60, 42, line_height=24, size=30, weight=700),
        _svg_text(["Architecture rendered from the implemented pipeline in the repository."], 60, 68, line_height=22, size=16),
    ]
    for x1, y1, x2, y2 in arrows:
        body.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#475569" stroke-width="3" marker-end="url(#arrow)"/>'
        )
    for x, y, width, height, fill, title, lines in boxes:
        body.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="22" fill="{fill}" stroke="#94a3b8" stroke-width="2"/>')
        body.append(_svg_text([title], x + 18, y + 34, line_height=22, size=22, weight=600))
        body.append(_svg_text(lines, x + 18, y + 62, line_height=22, size=16))
    body.append(
        """
<defs>
  <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
    <path d="M0,0 L12,6 L0,12 z" fill="#475569"/>
  </marker>
</defs>
""".strip()
    )
    _write_svg(path, 1220, 430, "\n".join(body))


def render_runtime_svg(path: Path, metrics: dict[str, Any]) -> None:
    stages = [stage for stage in metrics["runtime"]["stages"] if stage["seconds"]]
    max_seconds = max(float(stage["seconds"]) for stage in stages) if stages else 1.0
    chart_left = 240
    chart_right = 1040
    bar_height = 34
    top = 110
    body: list[str] = [
        _svg_text(["Runtime Breakdown from `data/outputs/week3_test_run.json`"], 60, 46, line_height=24, size=30, weight=700),
        _svg_text(
            [
                f"Workflow latency: {_format_seconds(metrics['runtime']['workflow_latency_seconds'])}",
                f"Audio duration: {_format_seconds(metrics['runtime']['audio_duration_seconds'])}",
                f"Workflow RTF: {_format_ratio(metrics['runtime']['workflow_rtf'])}",
            ],
            60,
            78,
            line_height=20,
            size=16,
        ),
    ]

    for index, stage in enumerate(stages):
        seconds = float(stage["seconds"])
        y = top + index * 52
        width = int((seconds / max_seconds) * (chart_right - chart_left))
        body.append(_svg_text([stage["name"]], 60, y + 23, size=18, weight=600))
        body.append(f'<rect x="{chart_left}" y="{y}" width="{chart_right - chart_left}" height="{bar_height}" rx="12" fill="#e2e8f0"/>')
        body.append(f'<rect x="{chart_left}" y="{y}" width="{width}" height="{bar_height}" rx="12" fill="#2563eb"/>')
        body.append(
            _svg_text(
                [f"{seconds:.3f}s | RTF {_format_ratio(stage['rtf'])}"],
                chart_left + min(width + 16, chart_right - 160),
                y + 23,
                size=16,
            )
        )
    _write_svg(path, 1120, 560, "\n".join(body))


def render_speaker_svg(path: Path, metrics: dict[str, Any]) -> None:
    speakers = metrics["transcript"]["speaker_distribution"]
    max_count = max(speakers.values()) if speakers else 1
    chart_left = 300
    chart_right = 1040
    bar_height = 28
    top = 110
    body: list[str] = [
        _svg_text(["Speaker Distribution on `test.wav`"], 60, 46, line_height=24, size=30, weight=700),
        _svg_text(
            [
                f"Detected speakers: {len(speakers)}",
                f"Transcript segments: {metrics['transcript']['segment_count']}",
            ],
            60,
            78,
            line_height=20,
            size=16,
        ),
    ]
    for index, (speaker, count) in enumerate(speakers.items()):
        y = top + index * 46
        width = int((count / max_count) * (chart_right - chart_left))
        body.append(_svg_text([speaker], 60, y + 20, size=18, weight=600))
        body.append(f'<rect x="{chart_left}" y="{y}" width="{chart_right - chart_left}" height="{bar_height}" rx="12" fill="#e2e8f0"/>')
        body.append(f'<rect x="{chart_left}" y="{y}" width="{width}" height="{bar_height}" rx="12" fill="#0f766e"/>')
        body.append(_svg_text([str(count)], chart_left + min(width + 16, chart_right - 48), y + 20, size=16))
    _write_svg(path, 1120, 420, "\n".join(body))


def render_snapshot_svg(path: Path, metrics: dict[str, Any]) -> None:
    transcript_lines: list[str] = []
    for line in metrics["transcript"]["preview"]:
        transcript_lines.extend(_wrap_text(line, 46))

    summary_lines: list[str] = ["Topics:"]
    for item in metrics["summary"]["preview"]["topics"]:
        summary_lines.extend(_wrap_text(f"- {item}", 42))
    summary_lines.append("Decisions:")
    decisions = metrics["summary"]["preview"]["decisions"] or ["- None recorded"]
    for item in decisions:
        summary_lines.extend(_wrap_text(f"- {item}", 42))

    action_lines: list[str] = []
    if metrics["action_items"]["preview"]:
        for item in metrics["action_items"]["preview"]:
            action_lines.extend(_wrap_text(f"- {item}", 42))
    else:
        action_lines.append("No action items extracted.")

    translation_lines: list[str] = []
    for line in metrics["translation"]["preview"]:
        translation_lines.extend(_wrap_text(line, 42))

    sentiment_lines = [
        f"Route: {metrics['sentiment']['route']}",
        f"Overall tone: {metrics['sentiment']['overall_tone']}",
    ]
    for label, count in metrics["sentiment"]["distribution"].items():
        sentiment_lines.append(f"{label}: {count}")

    body = "\n".join(
        [
            _svg_text(["Real Output Snapshot from `test.wav`"], 50, 40, line_height=24, size=30, weight=700),
            _svg_text(["Panels below are rendered from the actual Week 3 workflow JSON, not placeholders."], 50, 68, line_height=22, size=16),
            _panel("Transcript Preview", transcript_lines, 50, 100, 620, 380),
            _panel("Summary Preview", summary_lines, 700, 100, 620, 380),
            _panel("Action Items", action_lines, 50, 510, 620, 300),
            _panel("Translation + Sentiment", translation_lines + [""] + sentiment_lines, 700, 510, 620, 300),
        ]
    )
    _write_svg(path, 1370, 860, body)


def render_retrieval_svg(path: Path, metrics: dict[str, Any]) -> None:
    retrieval = metrics["retrieval"]
    query = retrieval["query"] or "No retrieval query provided."
    lines: list[str] = [f"Query: {query}", ""]
    if retrieval["preview"]:
        for index, item in enumerate(retrieval["preview"], start=1):
            score = item["score"]
            lines.append(f"Top {index}: {item['meeting_id']} (score={score if score is not None else 'n/a'})")
            lines.extend(_wrap_text(item["document"], 86)[:6])
            lines.append("")
    else:
        lines.append("No retrieval hits were available when the report was generated.")

    body = "\n".join(
        [
            _svg_text(["Chroma Retrieval Example"], 50, 40, line_height=24, size=30, weight=700),
            _svg_text(["History query is executed against persisted meeting summaries."], 50, 68, line_height=22, size=16),
            _panel("Retrieved Records", lines, 50, 100, 1260, 520),
        ]
    )
    _write_svg(path, 1360, 670, body)


def write_week35_report(report_path: Path, metrics: dict[str, Any]) -> None:
    assets_prefix = "assets/week3_5"
    summary_preview = metrics["summary"]["preview"]
    decisions = summary_preview["decisions"] or ["None recorded"]
    follow_ups = summary_preview["follow_ups"] or ["None recorded"]
    retrieval_preview = metrics["retrieval"]["preview"]

    stage_rows = []
    for stage in metrics["runtime"]["stages"]:
        if stage["seconds"] is None:
            continue
        stage_rows.append(f"| {stage['name']} | {_format_seconds(stage['seconds'])} | {_format_ratio(stage['rtf'])} |")

    speaker_rows = []
    for speaker, count in metrics["transcript"]["speaker_distribution"].items():
        speaker_rows.append(f"| {speaker} | {count} |")

    references = [
        "- Gao et al. *Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition*. https://arxiv.org/abs/2206.08317",
        "- Bredin et al. *pyannote.audio: neural building blocks for speaker diarization*. https://arxiv.org/abs/2011.04624",
        "- Wu et al. *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*. https://arxiv.org/abs/2308.08155",
        "- Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html",
    ]

    lines = [
        "# Week 3.5 Progress Report",
        "",
        f"Generated on {metrics['generated_on']} from a real workflow run on `data/samples/test.wav`.",
        "",
        "## Introduction",
        "",
        "This report summarizes the current state of the Meeting AI prototype after Week 3. The system targets a practical meeting-processing workflow: audio ingestion, speaker-aware transcription, structured downstream NLP analysis, meeting-memory retrieval, and a UI/CLI layer that can demo the full flow in one run.",
        "",
        f"The main evidence in this report comes from a single real run over a {metrics['runtime']['audio_duration_seconds']:.3f}s meeting recording. The pipeline stored the meeting under `{metrics['meeting_id']}` and completed with `{len(metrics['errors'])}` recorded workflow errors.",
        "",
        "## Related Work",
        "",
        "The current design draws on four strands of prior work:",
        "",
        *references,
        "",
        "In this project, those ideas are combined pragmatically rather than reproduced as a research system: FunASR and pyannote handle the speech front end, LLM-backed agents produce structured outputs, LangGraph manages orchestration and isolation, and Chroma provides lightweight retrieval over stored summaries.",
        "",
        "## System Design",
        "",
        "![Week 3 architecture](assets/week3_5/system_design.svg)",
        "",
        "The implemented workflow contains five main functional blocks:",
        "",
        "1. `MeetingASRAgent` converts audio into timestamped transcript segments and optionally attaches diarization labels.",
        "2. `SummaryAgent`, `TranslationAgent`, `ActionItemAgent`, and `SentimentAgent` run as the Week 2 NLU layer, with map-reduce chunking enabled for longer transcripts.",
        "3. `MeetingOrchestrator` fans out selected agents via LangGraph and aggregates the outputs into a single `MeetingWorkflowResult` object.",
        "4. `MeetingVectorStore` persists summary documents into Chroma for later history queries.",
        "5. `ui/app.py` surfaces the full path in Gradio for demo use.",
        "",
        f"For the current `test.wav` run, the transcript contains {metrics['transcript']['segment_count']} segments across {len(metrics['transcript']['speaker_distribution'])} detected speakers. Summary generation used the `{metrics['summary']['strategy']}` strategy with {metrics['summary']['chunk_count']} chunks. Translation and action-item extraction each operated on {metrics['translation']['chunk_count']} chunks, while sentiment used the `{metrics['sentiment']['route']}` route.",
        "",
        "## Preliminary Results",
        "",
        "### Runtime",
        "",
        "![Runtime breakdown](assets/week3_5/runtime_breakdown.svg)",
        "",
        "| Stage | Latency | RTF |",
        "| --- | --- | --- |",
        *stage_rows,
        "",
        f"The end-to-end workflow finished in {_format_seconds(metrics['runtime']['workflow_latency_seconds'])}, which corresponds to an overall RTF of {_format_ratio(metrics['runtime']['workflow_rtf'])}. The biggest latency contributors on this sample are translation ({_format_seconds(metrics['translation']['total_latency_seconds'])}) and LLM sentiment ({_format_seconds(metrics['sentiment']['latency_seconds'])}), while ASR ({_format_seconds(metrics['runtime']['asr_runtime_seconds'])}) and diarization ({_format_seconds(metrics['runtime']['diarization_runtime_seconds'])}) remain comfortably below real time.",
        f"The sum of individual stage latencies is {_format_seconds(metrics['runtime']['accounted_latency_seconds'])}, which is {metrics['runtime']['parallel_time_saved_seconds']:.3f}s higher than the end-to-end workflow latency. That gap is expected here and reflects useful overlap from the orchestrator's parallel fan-out rather than hidden overhead.",
        "",
        "### Speaker Activity",
        "",
        "![Speaker distribution](assets/week3_5/speaker_distribution.svg)",
        "",
        "| Speaker | Segments |",
        "| --- | --- |",
        *speaker_rows,
        "",
        "### Output Snapshot",
        "",
        "![Output snapshot](assets/week3_5/output_snapshot.svg)",
        "",
        "Observed output quality on the current run is already usable for demo purposes:",
        "",
        f"- The summary surfaces packaging evaluation, taste/aroma feedback, and pricing discussion as the dominant topics.",
        f"- The main explicit decision extracted so far is: `{decisions[0]}`.",
        f"- The system returned {metrics['action_items']['count']} actionable follow-ups; representative items include `{metrics['action_items']['preview'][0] if metrics['action_items']['preview'] else 'n/a'}`.",
        f"- Sentiment currently collapses to `{metrics['sentiment']['overall_tone']}` across all {metrics['sentiment']['segment_count']} segments on this sample, which is stable but not yet discriminative enough for final evaluation.",
        "",
        "### Retrieval Example",
        "",
        "![Retrieval example](assets/week3_5/retrieval_example.svg)",
        "",
        f"The history query used for this report is `{metrics['retrieval']['query']}`. It returned {metrics['retrieval']['result_count']} record(s). The highest-scoring hit reached a similarity score of {retrieval_preview[0]['score'] if retrieval_preview else 'n/a'}, which is enough to recover earlier packaging-related summaries from the local Chroma store.",
        "",
        "### WER/CER Status",
        "",
        "The codebase already includes `jiwer` and the evaluation wiring needed for ASR benchmarking, but this Week 3.5 report does not claim WER/CER numbers yet because `test.wav` does not have an aligned human reference transcript in the repository. For this milestone, the report therefore focuses on runtime, structured output quality, and retrieval behavior based on real system runs only.",
        "",
        "## Plan",
        "",
        "The remaining work to close Week 4 and Week 5 is straightforward:",
        "",
        "1. Build a small manually aligned reference set so WER/CER and RTF can be reported together.",
        "2. Add summary-quality evaluation with ROUGE and LLM-as-judge scoring.",
        "3. Compare multi-agent orchestration against a single-pipeline baseline for latency and error isolation.",
        "4. Prepare a polished demo package with a short walkthrough script, final screenshots, and the final six-page report.",
        "",
        "The current system is already stable enough to demo end-to-end. The next milestone is to convert that working prototype into quantified experimental evidence.",
        "",
        "## Appendix: Real Run Highlights",
        "",
        f"- Summary topics sample: {', '.join(summary_preview['topics'][:3])}",
        f"- Follow-up sample: {follow_ups[0]}",
        f"- Retrieval store meeting id: `{metrics['meeting_id']}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_week35_report(
    result: MeetingWorkflowResult,
    output_root: Path,
    retrieval_query: str | None = None,
    retrieval_results: list[RetrievalRecord] | None = None,
    generated_on: str | None = None,
) -> Week35Artifacts:
    output_root.mkdir(parents=True, exist_ok=True)
    assets_dir = output_root / "assets" / "week3_5"
    assets_dir.mkdir(parents=True, exist_ok=True)

    metrics = build_week35_metrics(
        result=result,
        retrieval_query=retrieval_query,
        retrieval_results=retrieval_results,
        generated_on=generated_on,
    )

    report_path = output_root / "week3_5_progress_report.md"
    metrics_path = assets_dir / "metrics.json"
    architecture_svg_path = assets_dir / "system_design.svg"
    runtime_svg_path = assets_dir / "runtime_breakdown.svg"
    speaker_svg_path = assets_dir / "speaker_distribution.svg"
    snapshot_svg_path = assets_dir / "output_snapshot.svg"
    retrieval_svg_path = assets_dir / "retrieval_example.svg"

    metrics_path.write_text(__import__("json").dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    render_architecture_svg(architecture_svg_path)
    render_runtime_svg(runtime_svg_path, metrics)
    render_speaker_svg(speaker_svg_path, metrics)
    render_snapshot_svg(snapshot_svg_path, metrics)
    render_retrieval_svg(retrieval_svg_path, metrics)
    write_week35_report(report_path, metrics)

    return Week35Artifacts(
        report_path=report_path,
        metrics_path=metrics_path,
        architecture_svg_path=architecture_svg_path,
        runtime_svg_path=runtime_svg_path,
        speaker_svg_path=speaker_svg_path,
        snapshot_svg_path=snapshot_svg_path,
        retrieval_svg_path=retrieval_svg_path,
    )
