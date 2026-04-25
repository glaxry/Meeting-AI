from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Week5Artifacts:
    report_path: Path
    asr_svg_path: Path
    summary_svg_path: Path
    architecture_svg_path: Path
    sentiment_svg_path: Path
    overview_svg_path: Path
    judge_quick_start_path: Path
    demo_script_path: Path
    presentation_outline_path: Path
    qna_bank_path: Path
    recording_runbook_path: Path
    highlight_demo_path: Path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _svg_text(lines: list[str], x: int, y: int, line_height: int = 24, size: int = 18, weight: int = 400) -> str:
    parts: list[str] = []
    for index, line in enumerate(lines):
        parts.append(
            f'<text x="{x}" y="{y + index * line_height}" '
            f'font-family="Segoe UI, Microsoft YaHei, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="#111827">{escape(line)}</text>'
        )
    return "\n".join(parts)


def _write_svg(path: Path, width: int, height: int, body: str) -> None:
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
            body,
            "</svg>",
        ]
    )
    path.write_text(svg, encoding="utf-8")


def _render_bar_chart(
    *,
    path: Path,
    title: str,
    subtitle: str,
    rows: list[tuple[str, float, str]],
    color: str,
    width: int = 1180,
    height: int = 480,
) -> None:
    chart_left = 300
    chart_right = width - 90
    top = 110
    bar_height = 34
    row_gap = 58
    max_value = max((value for _, value, _ in rows), default=1.0)
    body: list[str] = [
        _svg_text([title], 48, 44, size=30, weight=700),
        _svg_text([subtitle], 48, 74, size=16, line_height=20),
    ]
    for index, (label, value, suffix) in enumerate(rows):
        y = top + index * row_gap
        width_px = int((value / max_value) * (chart_right - chart_left)) if max_value else 0
        body.append(_svg_text([label], 48, y + 22, size=18, weight=600))
        body.append(f'<rect x="{chart_left}" y="{y}" width="{chart_right - chart_left}" height="{bar_height}" rx="12" fill="#e5e7eb"/>')
        body.append(f'<rect x="{chart_left}" y="{y}" width="{width_px}" height="{bar_height}" rx="12" fill="{color}"/>')
        body.append(_svg_text([f"{value:.3f}{suffix}"], chart_left + min(width_px + 16, chart_right - 180), y + 22, size=16))
    _write_svg(path, width, height, "\n".join(body))


def _render_two_metric_chart(
    *,
    path: Path,
    title: str,
    subtitle: str,
    rows: list[dict[str, Any]],
    left_key: str,
    right_key: str,
    left_label: str,
    right_label: str,
    left_color: str,
    right_color: str,
) -> None:
    width = 1320
    height = 520
    left_origin = 360
    gap = 140
    bar_width_max = 320
    top = 130
    row_gap = 70
    left_max = max((float(row[left_key]) for row in rows), default=1.0)
    right_max = max((float(row[right_key]) for row in rows), default=1.0)
    body: list[str] = [
        _svg_text([title], 48, 44, size=30, weight=700),
        _svg_text([subtitle], 48, 74, size=16, line_height=20),
        _svg_text([left_label], left_origin, 108, size=16, weight=700),
        _svg_text([right_label], left_origin + bar_width_max + gap, 108, size=16, weight=700),
    ]
    for index, row in enumerate(rows):
        y = top + index * row_gap
        left_width = int((float(row[left_key]) / left_max) * bar_width_max) if left_max else 0
        right_width = int((float(row[right_key]) / right_max) * bar_width_max) if right_max else 0
        body.append(_svg_text([str(row["label"])], 48, y + 22, size=18, weight=600))
        body.append(f'<rect x="{left_origin}" y="{y}" width="{bar_width_max}" height="32" rx="12" fill="#e5e7eb"/>')
        body.append(f'<rect x="{left_origin}" y="{y}" width="{left_width}" height="32" rx="12" fill="{left_color}"/>')
        body.append(_svg_text([f"{float(row[left_key]):.3f}"], left_origin + min(left_width + 16, bar_width_max - 70), y + 22, size=16))
        right_origin = left_origin + bar_width_max + gap
        body.append(f'<rect x="{right_origin}" y="{y}" width="{bar_width_max}" height="32" rx="12" fill="#e5e7eb"/>')
        body.append(f'<rect x="{right_origin}" y="{y}" width="{right_width}" height="32" rx="12" fill="{right_color}"/>')
        body.append(_svg_text([f"{float(row[right_key]):.3f}"], right_origin + min(right_width + 16, bar_width_max - 70), y + 22, size=16))
    _write_svg(path, width, height, "\n".join(body))


def _render_overview_svg(path: Path, week35_metrics: dict[str, Any], architecture_eval: dict[str, Any], sentiment_eval: dict[str, Any]) -> None:
    width = 1360
    height = 780
    body: list[str] = [
        _svg_text(["Meeting AI Final Snapshot"], 48, 44, size=30, weight=700),
        _svg_text(["This board condenses the core numbers used in the final report and demo."], 48, 74, size=16),
    ]
    sentiment_dataset = sentiment_eval.get("dataset", {})
    sentiment_sample_count = int(sentiment_dataset.get("sample_count", sentiment_eval["routes"]["transformer"].get("sample_count", 0)) or 0)
    llm_warnings = sentiment_eval["routes"].get("llm_deepseek", {}).get("warnings", [])
    llm_summary_line = (
        "LLM hit benchmark ceiling; keep as diagnostic only"
        if llm_warnings
        else f"LLM acc/F1: {sentiment_eval['routes']['llm_deepseek']['accuracy']:.3f} / {sentiment_eval['routes']['llm_deepseek']['macro_f1']:.3f}"
    )
    cards = [
        (48, 112, 390, 180, "#dbeafe", "End-to-End Runtime", [
            f"Audio duration: {week35_metrics['runtime']['audio_duration_seconds']:.3f}s",
            f"Workflow latency: {week35_metrics['runtime']['workflow_latency_seconds']:.3f}s",
            f"Workflow RTF: {week35_metrics['runtime']['workflow_rtf']:.3f}",
            f"Stored meeting id: {week35_metrics['meeting_id']}",
        ]),
        (484, 112, 390, 180, "#dcfce7", "Architecture Result", [
            f"Parallel latency: {architecture_eval['runtime_compare']['parallel_latency_seconds']:.3f}s",
            f"Serial latency: {architecture_eval['runtime_compare']['serial_latency_seconds']:.3f}s",
            f"Speedup: {architecture_eval['runtime_compare']['speedup']:.3f}x",
            f"Failure isolation: {architecture_eval['error_isolation_demo']['parallel_completed_agents']} vs {architecture_eval['error_isolation_demo']['serial_completed_agents']} agents",
        ]),
        (920, 112, 390, 180, "#fee2e2", "Sentiment Trade-off", [
            f"Benchmark size: {sentiment_sample_count} items",
            f"Transformer acc/F1: {sentiment_eval['routes']['transformer']['accuracy']:.3f} / {sentiment_eval['routes']['transformer']['macro_f1']:.3f}",
            f"Transformer latency: {sentiment_eval['routes']['transformer']['latency_seconds']:.3f}s",
            llm_summary_line,
            f"LLM latency: {sentiment_eval['routes']['llm_deepseek']['latency_seconds']:.3f}s",
        ]),
        (48, 330, 1262, 380, "#ffffff", "Demo Narrative", [
            "1. Upload `test.wav` in Gradio and show the transcript tab first.",
            "2. Highlight summary / translation / action-item tabs while the audience already sees structured outputs.",
            "3. Explain why LangGraph matters with the Week 4 architecture comparison: faster runtime plus partial results on failure.",
            "4. Close with the Week 4 sentiment trade-off: transformer is fast, LLM is more accurate.",
            "5. Keep `reports/final_project_report.md` open as backup evidence if the UI or API is slow.",
        ]),
    ]
    for x, y, w, h, fill, title, lines in cards:
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" fill="{fill}" stroke="#cbd5e1" stroke-width="2"/>')
        body.append(_svg_text([title], x + 22, y + 38, size=24, weight=700))
        body.append(_svg_text(lines, x + 22, y + 78, size=18, line_height=28))
    _write_svg(path, width, height, "\n".join(body))


def write_final_report(
    path: Path,
    week35_metrics: dict[str, Any],
    asr_eval: dict[str, Any],
    summary_eval: dict[str, Any],
    architecture_eval: dict[str, Any],
    sentiment_eval: dict[str, Any],
) -> None:
    asr_paraformer = asr_eval["models"]["paraformer-zh"]
    asr_sensevoice = asr_eval["models"]["iic/SenseVoiceSmall"]
    summary_default = summary_eval["strategies"]["default"]
    summary_single = summary_eval["strategies"]["single_pass"]
    architecture_runtime = architecture_eval["runtime_compare"]
    architecture_failure = architecture_eval["error_isolation_demo"]
    sentiment_transformer = sentiment_eval["routes"]["transformer"]
    sentiment_llm = sentiment_eval["routes"]["llm_deepseek"]
    sentiment_dataset = sentiment_eval.get("dataset", {})
    sentiment_sample_count = int(sentiment_dataset.get("sample_count", sentiment_transformer.get("sample_count", 0)) or 0)
    sentiment_warnings = sentiment_eval.get("warnings", [])
    llm_warnings = sentiment_llm.get("warnings", [])

    lines = [
        "# Smart Meeting Assistant: Final Project Report",
        "",
        "## Abstract",
        "",
        "This project implements a practical multi-agent meeting assistant for transcription, summarization, translation, action-item extraction, sentiment analysis, and meeting-memory retrieval. The final system combines FunASR, pyannote, DeepSeek-backed structured LLM agents, LangGraph orchestration, Chroma retrieval, and a Gradio demo interface. The final report is backed by real workflow runs and reproducible Week 4 evaluations rather than design intent alone.",
        "",
        "## 1. Introduction",
        "",
        "Meetings generate large amounts of semi-structured information, but the useful outcomes are rarely limited to raw transcripts. A practical assistant needs to identify who spoke, summarize the main topics, preserve bilingual usability, extract follow-up tasks, characterize discussion tone, and recover related prior decisions.",
        "",
        f"The current end-to-end run on `data/samples/test.wav` processed {week35_metrics['transcript']['segment_count']} transcript segments from {len(week35_metrics['transcript']['speaker_distribution'])} detected speakers and finished in {week35_metrics['runtime']['workflow_latency_seconds']:.3f}s on a {week35_metrics['runtime']['audio_duration_seconds']:.3f}s recording, giving workflow RTF {week35_metrics['runtime']['workflow_rtf']:.3f}.",
        "",
        "## 2. Related Work",
        "",
        "- Gao et al. *Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition*. https://arxiv.org/abs/2206.08317",
        "- Bredin et al. *pyannote.audio: neural building blocks for speaker diarization*. https://arxiv.org/abs/2011.04624",
        "- Wu et al. *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*. https://arxiv.org/abs/2308.08155",
        "- Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html",
        "",
        "## 3. Method and System Design",
        "",
        "![System architecture](assets/week3_5/system_design.svg)",
        "",
        "The system is organized as a speech front end, a set of structured NLU agents, a LangGraph orchestrator, a Chroma retrieval layer, and a Gradio/CLI interface. A serial baseline is also implemented so the architecture claim can be benchmarked rather than asserted.",
        "",
        "## 4. Experimental Results",
        "",
        "### 4.1 ASR Comparison",
        "",
        "![ASR comparison](assets/week5/asr_compare.svg)",
        "",
        f"`paraformer-zh` reached WER/CER {asr_paraformer['mean_wer']:.3f}/{asr_paraformer['mean_cer']:.3f} with mean RTF {asr_paraformer['mean_rtf']:.3f}. `iic/SenseVoiceSmall` reached WER/CER {asr_sensevoice['mean_wer']:.3f}/{asr_sensevoice['mean_cer']:.3f} with mean RTF {asr_sensevoice['mean_rtf']:.3f}. `SenseVoiceSmall` is much faster but still drops punctuation on the current short Chinese sample.",
        "",
        "### 4.2 Summary Quality and Ablation",
        "",
        "![Summary evaluation](assets/week5/summary_compare.svg)",
        "",
        f"On the current three-sample summary set, `single_pass` achieved ROUGE-1/2/L {summary_single['mean_rouge1']:.3f}/{summary_single['mean_rouge2']:.3f}/{summary_single['mean_rougeL']:.3f} with mean judge score {summary_single['mean_judge_overall']:.3f}, outperforming the default strategy ({summary_default['mean_rouge1']:.3f}/{summary_default['mean_rouge2']:.3f}/{summary_default['mean_rougeL']:.3f}, judge {summary_default['mean_judge_overall']:.3f}). The current reduce prompt over-expands long summaries and should be tuned further.",
        "",
        "### 4.3 Architecture Comparison",
        "",
        "![Architecture evaluation](assets/week5/architecture_compare.svg)",
        "",
        f"Using the first {architecture_runtime['transcript_segment_count']} segments from the real `test.wav` transcript, the parallel orchestrator completed in {architecture_runtime['parallel_latency_seconds']:.3f}s versus {architecture_runtime['serial_latency_seconds']:.3f}s for the serial baseline, a {architecture_runtime['speedup']:.3f}x speedup. In the injected translation-failure case, the parallel orchestrator still completed {architecture_failure['parallel_completed_agents']} downstream agents, while the serial fail-fast baseline completed only {architecture_failure['serial_completed_agents']}.",
        "",
        "### 4.4 Sentiment Trade-off",
        "",
        "![Sentiment evaluation](assets/week5/sentiment_compare.svg)",
        "",
        f"On the current {sentiment_sample_count}-item sentiment benchmark, the transformer route achieved accuracy {sentiment_transformer['accuracy']:.3f}, macro F1 {sentiment_transformer['macro_f1']:.3f}, and latency {sentiment_transformer['latency_seconds']:.3f}s.",
        (
            f"The DeepSeek LLM route returned accuracy {sentiment_llm['accuracy']:.3f}, macro F1 {sentiment_llm['macro_f1']:.3f}, and latency {sentiment_llm['latency_seconds']:.3f}s, "
            "but this route saturated the current benchmark and is treated as a ceiling-effect diagnostic rather than a headline production claim."
            if llm_warnings
            else f"The DeepSeek LLM route achieved accuracy {sentiment_llm['accuracy']:.3f}, macro F1 {sentiment_llm['macro_f1']:.3f}, and latency {sentiment_llm['latency_seconds']:.3f}s."
        ),
        *[f"Benchmark note: {warning}" for warning in sentiment_warnings],
        "",
        "### 4.5 End-to-End Demo Evidence",
        "",
        "![Week 5 overview](assets/week5/final_overview.svg)",
        "",
        f"The final workflow run on `test.wav` used `{week35_metrics['summary']['strategy']}` summarization over {week35_metrics['summary']['chunk_count']} chunks, produced {week35_metrics['action_items']['count']} action items, and persisted the meeting under `{week35_metrics['meeting_id']}` for retrieval.",
        "",
        "## 5. Demo and Deployment Materials",
        "",
        "The repository now includes a Week 5 demo package under `demo/`, covering judge quick start, live-demo script, presentation outline, Q&A bank, a recording runbook, and a reusable highlight-demo transcript.",
        "",
        "## 6. Limitations and Future Work",
        "",
        "The benchmark sets are still small, there is no DER benchmark yet, and the summary reduce prompt needs more tuning on long transcripts. These limitations are now clearly isolated because the repo already contains the measurement harness needed to improve them.",
        "",
        "## 7. Conclusion",
        "",
        "The project now ships as a working multi-agent meeting assistant with measured Week 4 results and a complete Week 5 report/demo package. It is ready for submission, live demo, and incremental benchmarking.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_demo_documents(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)

    judge_quick_start = root / "judge_quick_start.md"
    judge_quick_start.write_text(
        "\n".join(
            [
                "# Judge Quick Start",
                "",
                "1. Activate the environment: `conda activate meeting-ai-w1`",
                "2. Verify the runtime: `python scripts/check_env.py`",
                "3. Launch the UI: `python ui/app.py`",
                "4. Open `http://127.0.0.1:7860` and upload `data/samples/test.wav`.",
                "5. If the UI path is inconvenient, run:",
                "   `python scripts/week3_demo.py --audio .\\data\\samples\\test.wav --language zh --provider deepseek --target-language en --sentiment-route llm --output .\\data\\outputs\\week3_test_run.json`",
                "",
                "Backup evidence:",
                "",
                "- `reports/final_project_report.md`",
                "- `reports/week4_experiments.md`",
                "- `reports/week4/*.json`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    demo_script = root / "demo_script.md"
    demo_script.write_text(
        "\n".join(
            [
                "# Demo Script",
                "",
                "## Upload and Transcription",
                "- Upload `test.wav` in Gradio.",
                "- Start from the transcript tab so the audience sees raw evidence first.",
                "",
                "## Architecture",
                "- Show `reports/assets/week3_5/system_design.svg`.",
                "- Explain that ASR happens once and the downstream agents fan out in parallel.",
                "",
                "## Agent Walkthrough",
                "- Summary: topics, decisions, follow-ups.",
                "- Translation: speaker labels preserved.",
                "- Action items: structured tasks for follow-up.",
                "- Sentiment: explain both LLM and transformer routes.",
                "- History: explain retrieval over stored meeting summaries.",
                "",
                "## Quantitative Close",
                "- Show `reports/assets/week5/architecture_compare.svg` and mention the 1.34x speedup.",
                "- Show `reports/assets/week5/sentiment_compare.svg` and mention the speed/accuracy trade-off.",
                "- Close by pointing to `reports/final_project_report.md` as the written evidence package.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    presentation_outline = root / "presentation_outline.md"
    presentation_outline.write_text(
        "\n".join(
            [
                "# Presentation Outline",
                "",
                "1. Problem and project goal",
                "2. System architecture",
                "3. Agent layer and orchestration",
                "4. End-to-end demo on `test.wav`",
                "5. Week 4 experiments",
                "6. Takeaways and next steps",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    qna_bank = root / "qna_bank.md"
    qna_bank.write_text(
        "\n".join(
            [
                "# Q&A Bank",
                "",
                "## Why LangGraph instead of a simple function pipeline?",
                "Week 4 measured 50.803s versus 68.093s on the shared transcript slice, and the parallel version preserved more partial outputs under failure.",
                "",
                "## Why keep both transformer and LLM sentiment routes?",
                "The transformer route is fast and local; the LLM route is slower but more accurate on the current labeled set.",
                "",
                "## Why is single-pass summary better than map-reduce right now?",
                "The current reduce prompt over-expands long summaries, which is visible in the Week 4 ablation.",
                "",
                "## What are the biggest remaining limitations?",
                "Small evaluation sets, no DER benchmark yet, and summary tuning on long meetings.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    recording_runbook = root / "recording_runbook.md"
    recording_runbook.write_text(
        "\n".join(
            [
                "# Recording Runbook",
                "",
                "- Activate `meeting-ai-w1`.",
                "- Run `python scripts/check_env.py`.",
                "- Start `python ui/app.py` before recording.",
                "- Keep `reports/final_project_report.md` open as fallback evidence.",
                "- If the API is slow, switch sentiment to `transformer` or narrate the existing Week 4 result files.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    highlight_demo = root / "highlight_demo_transcript.md"
    highlight_demo.write_text(
        "\n".join(
            [
                "# Highlight Demo Transcript",
                "",
                "Use this as a stronger showcase script for a later recording. It includes disagreement, risk, task assignment, and a final decision.",
                "",
                "[PM] We need to decide today whether the partner portal launches on Friday.",
                "[ENG] The backend is stable, but the export retry bug is still open and I am not comfortable releasing unless that is fixed.",
                "[QA] I agree with engineering. The main regression suite passed, but I still see a mobile navigation issue that could confuse first-time users.",
                "[SALES] Delaying to next week will hurt two customer demos that are already scheduled.",
                "[OPS] The deployment runbook is ready, but we still need the vendor to confirm webhook rate limits.",
                "[PM] Decision: we keep the Friday launch date only if engineering closes the export issue, QA signs off tomorrow morning, and operations gets vendor confirmation today.",
                "[PM] Action items: engineering updates the launch channel at 3 p.m., QA sends final sign-off tomorrow morning, operations confirms rate limits today, and sales updates the launch brief this afternoon.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "judge_quick_start": judge_quick_start,
        "demo_script": demo_script,
        "presentation_outline": presentation_outline,
        "qna_bank": qna_bank,
        "recording_runbook": recording_runbook,
        "highlight_demo": highlight_demo,
    }


def export_week5_materials(
    *,
    project_root: Path,
    report_root: Path,
    demo_root: Path,
) -> Week5Artifacts:
    week35_metrics = _load_json(report_root / "assets" / "week3_5" / "metrics.json")
    asr_eval = _load_json(report_root / "week4" / "asr_eval.json")
    summary_eval = _load_json(report_root / "week4" / "summary_eval.json")
    architecture_eval = _load_json(report_root / "week4" / "architecture_eval.json")
    sentiment_eval = _load_json(report_root / "week4" / "sentiment_eval.json")

    assets_dir = report_root / "assets" / "week5"
    assets_dir.mkdir(parents=True, exist_ok=True)

    asr_svg_path = assets_dir / "asr_compare.svg"
    _render_two_metric_chart(
        path=asr_svg_path,
        title="Week 4 ASR Comparison",
        subtitle="Measured on the committed smoke benchmark manifest.",
        rows=[
            {"label": "paraformer-zh", "wer": float(asr_eval["models"]["paraformer-zh"]["mean_wer"]), "rtf": float(asr_eval["models"]["paraformer-zh"]["mean_rtf"])},
            {"label": "iic/SenseVoiceSmall", "wer": float(asr_eval["models"]["iic/SenseVoiceSmall"]["mean_wer"]), "rtf": float(asr_eval["models"]["iic/SenseVoiceSmall"]["mean_rtf"])},
        ],
        left_key="wer",
        right_key="rtf",
        left_label="WER",
        right_label="RTF",
        left_color="#ef4444",
        right_color="#2563eb",
    )

    summary_svg_path = assets_dir / "summary_compare.svg"
    _render_two_metric_chart(
        path=summary_svg_path,
        title="Summary Evaluation and Ablation",
        subtitle="Current long-sample behavior favors the single-pass prompt over the default strategy.",
        rows=[
            {"label": "default", "rouge1": float(summary_eval["strategies"]["default"]["mean_rouge1"]), "judge": float(summary_eval["strategies"]["default"]["mean_judge_overall"])},
            {"label": "single_pass", "rouge1": float(summary_eval["strategies"]["single_pass"]["mean_rouge1"]), "judge": float(summary_eval["strategies"]["single_pass"]["mean_judge_overall"])},
        ],
        left_key="rouge1",
        right_key="judge",
        left_label="ROUGE-1",
        right_label="Judge Score",
        left_color="#0f766e",
        right_color="#7c3aed",
    )

    architecture_svg_path = assets_dir / "architecture_compare.svg"
    _render_bar_chart(
        path=architecture_svg_path,
        title="Architecture Comparison",
        subtitle="Parallel orchestrator versus serial baseline on the shared transcript slice.",
        rows=[
            ("Parallel LangGraph", float(architecture_eval["runtime_compare"]["parallel_latency_seconds"]), "s"),
            ("Serial Baseline", float(architecture_eval["runtime_compare"]["serial_latency_seconds"]), "s"),
            ("Failure-Isolation Agents Saved", float(architecture_eval["error_isolation_demo"]["parallel_completed_agents"] - architecture_eval["error_isolation_demo"]["serial_completed_agents"]), ""),
        ],
        color="#2563eb",
    )

    sentiment_svg_path = assets_dir / "sentiment_compare.svg"
    _render_two_metric_chart(
        path=sentiment_svg_path,
        title="Sentiment Route Trade-off",
        subtitle="Accuracy/F1 versus latency on the manually labeled Week 4 set.",
        rows=[
            {"label": "transformer", "metric": float(sentiment_eval["routes"]["transformer"]["macro_f1"]), "latency": float(sentiment_eval["routes"]["transformer"]["latency_seconds"])},
            {"label": "llm_deepseek", "metric": float(sentiment_eval["routes"]["llm_deepseek"]["macro_f1"]), "latency": float(sentiment_eval["routes"]["llm_deepseek"]["latency_seconds"])},
        ],
        left_key="metric",
        right_key="latency",
        left_label="Macro F1",
        right_label="Latency (s)",
        left_color="#16a34a",
        right_color="#dc2626",
    )

    overview_svg_path = assets_dir / "final_overview.svg"
    _render_overview_svg(overview_svg_path, week35_metrics, architecture_eval, sentiment_eval)

    report_path = report_root / "final_project_report.md"
    write_final_report(report_path, week35_metrics, asr_eval, summary_eval, architecture_eval, sentiment_eval)

    demo_paths = write_demo_documents(demo_root)

    return Week5Artifacts(
        report_path=report_path,
        asr_svg_path=asr_svg_path,
        summary_svg_path=summary_svg_path,
        architecture_svg_path=architecture_svg_path,
        sentiment_svg_path=sentiment_svg_path,
        overview_svg_path=overview_svg_path,
        judge_quick_start_path=demo_paths["judge_quick_start"],
        demo_script_path=demo_paths["demo_script"],
        presentation_outline_path=demo_paths["presentation_outline"],
        qna_bank_path=demo_paths["qna_bank"],
        recording_runbook_path=demo_paths["recording_runbook"],
        highlight_demo_path=demo_paths["highlight_demo"],
    )
