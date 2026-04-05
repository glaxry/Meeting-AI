from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from rouge_score import rouge_scorer

from .schemas import SummaryResult


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def resolve_manifest_path(manifest_path: str | Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(manifest_path).expanduser().resolve().parent.joinpath(path).resolve()


def strip_speaker_labels(text: str) -> str:
    normalized_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and "]" in stripped:
            stripped = stripped.split("]", 1)[1].strip()
        normalized_lines.append(stripped)
    return "\n".join(normalized_lines).strip()


def normalize_metric_text(text: str) -> str:
    return " ".join(strip_speaker_labels(text).replace("\u3000", " ").split())


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _prepare_wer_text(text: str) -> str:
    normalized = normalize_metric_text(text)
    if " " not in normalized and _contains_cjk(normalized):
        return " ".join(char for char in normalized if not char.isspace())
    return normalized


def compute_error_rates(reference_text: str, hypothesis_text: str) -> dict[str, float]:
    from jiwer import cer, wer

    normalized_reference = normalize_metric_text(reference_text)
    normalized_hypothesis = normalize_metric_text(hypothesis_text)
    return {
        "wer": round(float(wer(_prepare_wer_text(reference_text), _prepare_wer_text(hypothesis_text))), 6),
        "cer": round(float(cer(normalized_reference, normalized_hypothesis)), 6),
    }


def summary_to_eval_text(summary: SummaryResult | dict[str, Any]) -> str:
    payload = summary if isinstance(summary, dict) else summary.model_dump(exclude={"metadata"})
    lines: list[str] = []
    for key in ["topics", "decisions", "follow_ups"]:
        lines.append(f"{key}:")
        for item in payload.get(key, []) or []:
            lines.append(f"- {item}")
    return "\n".join(lines)


def compute_rouge(reference_text: str, hypothesis_text: str) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_text, hypothesis_text)
    return {
        name: round(float(score.fmeasure), 6)
        for name, score in scores.items()
    }


def accuracy_score(labels: Sequence[str], predictions: Sequence[str]) -> float:
    if not labels:
        return 0.0
    correct = sum(1 for gold, pred in zip(labels, predictions) if gold == pred)
    return round(correct / len(labels), 6)


def precision_recall_f1(labels: Sequence[str], predictions: Sequence[str], label_space: Sequence[str]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for label in label_space:
        true_positive = sum(1 for gold, pred in zip(labels, predictions) if gold == label and pred == label)
        false_positive = sum(1 for gold, pred in zip(labels, predictions) if gold != label and pred == label)
        false_negative = sum(1 for gold, pred in zip(labels, predictions) if gold == label and pred != label)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": int(sum(1 for gold in labels if gold == label)),
        }
    return rows


def macro_f1_score(labels: Sequence[str], predictions: Sequence[str], label_space: Sequence[str]) -> float:
    rows = precision_recall_f1(labels, predictions, label_space)
    if not rows:
        return 0.0
    return round(mean(row["f1"] for row in rows.values()), 6)


def confusion_matrix(labels: Sequence[str], predictions: Sequence[str], label_space: Sequence[str]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        gold: {pred: 0 for pred in label_space}
        for gold in label_space
    }
    for gold, pred in zip(labels, predictions):
        matrix[gold][pred] += 1
    return matrix


def mean_or_none(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return round(mean(items), 6)
