import json

import numpy as np
import pytest

from classifier.metrics import classification_metrics


def test_metrics_match_hand_calculated_binary_example():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([[0.9, 0.1], [0.6, 0.4], [0.7, 0.3], [0.2, 0.8]])
    metrics = classification_metrics(labels, probabilities, ["cat", "dog"])

    expected = {
        "accuracy": 0.75,
        "balanced_accuracy": 0.75,
        "macro_precision": 5 / 6,
        "macro_recall": 0.75,
        "macro_f1": 11 / 15,
        "weighted_f1": 11 / 15,
        "top3_accuracy": 1,
        "top5_accuracy": 1,
        "negative_log_likelihood": -np.log([0.9, 0.6, 0.3, 0.8]).mean(),
        "brier_score": 0.35,
        "ece": 0.35,
        "mce": 0.7,
        "mean_confidence": 0.75,
        "confidence_correct": 23 / 30,
        "confidence_incorrect": 0.7,
        "multiclass_roc_auc_ovr_macro": 0.75,
        "average_precision_macro": 5 / 6,
    }
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value), key
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert metrics["normalized_confusion_matrix"] == [[1, 0], [0.5, 0.5]]
    assert metrics["per_class"][0]["f1"] == pytest.approx(0.8)
    assert metrics["per_class"][1]["recall"] == pytest.approx(0.5)
    assert [item["support"] for item in metrics["per_class"]] == [2, 2]
    # Published Wilson formula gives this interval for 3 successes in 4 trials.
    assert metrics["accuracy_ci95"]["lower"] == pytest.approx(0.30064184258240184)
    assert metrics["accuracy_ci95"]["upper"] == pytest.approx(0.9544127391902995)
    assert json.loads(json.dumps(metrics, allow_nan=False)) == metrics


def test_missing_classes_are_explicit_and_never_nan():
    metrics = classification_metrics(
        np.array([0, 0]), np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]]), ["a", "b", "c"]
    )

    assert metrics["balanced_accuracy"] == 1
    assert metrics["macro_recall"] == pytest.approx(1 / 3)
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["confidence_incorrect"] is None
    assert metrics["multiclass_roc_auc_ovr_macro"] is None
    assert metrics["average_precision_macro"] is None
    assert metrics["per_class"][0]["roc_auc"] is None
    assert metrics["per_class"][0]["average_precision"] == 1
    assert metrics["per_class"][1]["roc_auc"] is None
    assert metrics["per_class"][1]["average_precision"] is None
    assert metrics["normalized_confusion_matrix"][1] == [0, 0, 0]
    assert metrics["macro_f1_ci95"]["lower"] == pytest.approx(1 / 3)
    assert metrics["macro_f1_ci95"]["upper"] == pytest.approx(1 / 3)
    json.dumps(metrics, allow_nan=False)


def test_calibration_includes_confidence_one_and_leaves_empty_bins_null():
    metrics = classification_metrics(
        np.array([0, 1, 1]), np.array([[1, 0], [0, 1], [0.5, 0.5]]), ["a", "b"]
    )
    bins = metrics["calibration_bins"]

    assert len(bins) == 15
    assert sum(item["count"] for item in bins) == 3
    assert bins[-1]["count"] == 2
    assert bins[-1]["accuracy"] == bins[-1]["confidence"] == 1
    assert bins[-1]["upper"] == 1
    assert bins[7]["count"] == 1
    assert bins[7]["accuracy"] == 0
    assert bins[7]["confidence"] == 0.5
    assert metrics["ece"] == pytest.approx(1 / 6)
    for item in bins:
        if not item["count"]:
            assert item["accuracy"] is None
            assert item["confidence"] is None
            assert item["gap"] is None


def test_top_k_uses_stable_class_index_ties_and_clips_k_to_class_count():
    metrics = classification_metrics(np.array([0, 2, 4, 5]), np.full((4, 6), 1 / 6), list("abcdef"))
    assert metrics["accuracy"] == 0.25
    assert metrics["top3_accuracy"] == 0.5
    assert metrics["top5_accuracy"] == 0.75


def test_zero_true_class_probability_has_finite_nll_and_no_correct_confidence():
    metrics = classification_metrics(np.array([0, 1]), np.array([[0, 1], [1, 0]]), ["a", "b"])
    assert metrics["accuracy"] == 0
    assert metrics["brier_score"] == 2
    assert metrics["confidence_correct"] is None
    assert metrics["negative_log_likelihood"] == pytest.approx(-np.log(np.finfo(float).eps))
    json.dumps(metrics, allow_nan=False)


def test_bootstrap_is_reproducible_and_records_its_protocol():
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    probabilities = np.tile([0.6, 0.25, 0.15], (60, 1))
    first = classification_metrics(labels, probabilities, ["a", "b", "c"], seed=17)
    repeat = classification_metrics(labels, probabilities, ["a", "b", "c"], seed=17)
    interval = first["macro_f1_ci95"]

    assert interval == repeat["macro_f1_ci95"]
    assert interval["lower"] < first["macro_f1"] < interval["upper"]
    assert interval["resamples"] == 1000
    assert interval["seed"] == 17
    assert interval["confidence_level"] == 0.95


@pytest.mark.parametrize(
    ("labels", "probabilities", "class_names"),
    [
        (np.array([], dtype=int), np.empty((0, 2)), ["a", "b"]),
        (np.array([[0]]), np.array([[0.5, 0.5]]), ["a", "b"]),
        (np.array([0.0]), np.array([[0.5, 0.5]]), ["a", "b"]),
        (np.array([False]), np.array([[0.5, 0.5]]), ["a", "b"]),
        (np.array([-1]), np.array([[0.5, 0.5]]), ["a", "b"]),
        (np.array([2]), np.array([[0.5, 0.5]]), ["a", "b"]),
        (np.array([0]), np.array([0.5, 0.5]), ["a", "b"]),
        (np.array([0]), np.array([[0.4, 0.5]]), ["a", "b"]),
        (np.array([0]), np.array([[np.nan, 0.5]]), ["a", "b"]),
        (np.array([0]), np.array([[np.inf, 0.5]]), ["a", "b"]),
        (np.array([0]), np.array([[-0.1, 1.1]]), ["a", "b"]),
        (np.array([0]), np.array([["0.5", "0.5"]]), ["a", "b"]),
        (np.array([0]), np.array([[0.5, 0.5]]), ["same", "same"]),
    ],
)
def test_invalid_metrics_inputs_raise_value_error(labels, probabilities, class_names):
    with pytest.raises(ValueError):
        classification_metrics(labels, probabilities, class_names)
