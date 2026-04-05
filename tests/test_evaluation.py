from __future__ import annotations

from meeting_ai.evaluation import (
    accuracy_score,
    compute_error_rates,
    compute_rouge,
    confusion_matrix,
    macro_f1_score,
    precision_recall_f1,
    strip_speaker_labels,
    summary_to_eval_text,
)
from meeting_ai.schemas import SummaryResult


def test_strip_speaker_labels_removes_prefixes() -> None:
    text = "[SPEAKER_00] hello\n[SPEAKER_01] world"

    assert strip_speaker_labels(text) == "hello\nworld"


def test_compute_error_rates_matches_identical_text() -> None:
    scores = compute_error_rates("hello world", "hello world")

    assert scores["wer"] == 0.0
    assert scores["cer"] == 0.0


def test_compute_error_rates_tokenizes_cjk_for_wer() -> None:
    scores = compute_error_rates("欢迎大家。", "欢迎大家")

    assert scores["wer"] == 0.2
    assert scores["cer"] == 0.2


def test_compute_rouge_returns_full_match_for_identical_text() -> None:
    scores = compute_rouge("launch on friday", "launch on friday")

    assert scores["rouge1"] == 1.0
    assert scores["rougeL"] == 1.0


def test_summary_to_eval_text_lists_sections() -> None:
    text = summary_to_eval_text(
        SummaryResult(topics=["Launch"], decisions=["Ship"], follow_ups=["Send memo"])
    )

    assert "topics:" in text
    assert "- Send memo" in text


def test_classification_metrics_cover_macro_scores() -> None:
    labels = ["agreement", "neutral", "neutral"]
    predictions = ["agreement", "disagreement", "neutral"]
    label_space = ["agreement", "disagreement", "neutral"]

    assert accuracy_score(labels, predictions) == 0.666667
    per_label = precision_recall_f1(labels, predictions, label_space)
    assert per_label["agreement"]["f1"] == 1.0
    assert macro_f1_score(labels, predictions, label_space) == 0.555556
    matrix = confusion_matrix(labels, predictions, label_space)
    assert matrix["neutral"]["disagreement"] == 1
