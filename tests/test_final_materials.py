from __future__ import annotations

import json
from pathlib import Path

from meeting_ai.final_materials import export_week5_materials


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_export_week5_materials_writes_report_and_demo_pack(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    demo_root = tmp_path / "demo"
    _write_json(
        report_root / "assets" / "week3_5" / "metrics.json",
        {
            "meeting_id": "meeting-1",
            "transcript": {"segment_count": 100, "speaker_distribution": {"SPEAKER_00": 60, "SPEAKER_01": 40}},
            "summary": {"strategy": "map_reduce", "chunk_count": 3},
            "translation": {"chunk_count": 3},
            "action_items": {"count": 4},
            "runtime": {"audio_duration_seconds": 120.0, "workflow_latency_seconds": 50.0, "workflow_rtf": 0.417},
        },
    )
    _write_json(
        report_root / "week4" / "asr_eval.json",
        {
            "models": {
                "paraformer-zh": {"mean_wer": 0.0, "mean_cer": 0.0, "mean_rtf": 0.4},
                "iic/SenseVoiceSmall": {"mean_wer": 0.05, "mean_cer": 0.05, "mean_rtf": 0.03},
            }
        },
    )
    _write_json(
        report_root / "week4" / "summary_eval.json",
        {
            "strategies": {
                "default": {"mean_rouge1": 0.5, "mean_rouge2": 0.3, "mean_rougeL": 0.4, "mean_judge_overall": 3.5},
                "single_pass": {"mean_rouge1": 0.6, "mean_rouge2": 0.35, "mean_rougeL": 0.5, "mean_judge_overall": 4.5},
            }
        },
    )
    _write_json(
        report_root / "week4" / "architecture_eval.json",
        {
            "runtime_compare": {"transcript_segment_count": 80, "parallel_latency_seconds": 50.0, "serial_latency_seconds": 68.0, "speedup": 1.36},
            "error_isolation_demo": {"parallel_completed_agents": 3, "serial_completed_agents": 1},
        },
    )
    _write_json(
        report_root / "week4" / "sentiment_eval.json",
        {
            "routes": {
                "transformer": {"accuracy": 0.75, "macro_f1": 0.67, "latency_seconds": 0.4},
                "llm_deepseek": {"accuracy": 1.0, "macro_f1": 1.0, "latency_seconds": 11.2},
            }
        },
    )

    artifacts = export_week5_materials(project_root=tmp_path, report_root=report_root, demo_root=demo_root)

    assert artifacts.report_path.exists()
    assert artifacts.asr_svg_path.exists()
    assert artifacts.summary_svg_path.exists()
    assert artifacts.architecture_svg_path.exists()
    assert artifacts.sentiment_svg_path.exists()
    assert artifacts.overview_svg_path.exists()
    assert artifacts.judge_quick_start_path.exists()
    assert artifacts.demo_script_path.exists()
    assert artifacts.presentation_outline_path.exists()
    assert artifacts.qna_bank_path.exists()
    assert artifacts.recording_runbook_path.exists()
    assert artifacts.highlight_demo_path.exists()
    assert "Final Project Report" in artifacts.report_path.read_text(encoding="utf-8")
