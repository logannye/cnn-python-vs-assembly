"""Reproducible classification, calibration, and uncertainty measurements.

All probabilities and metrics use float64. Confidence intervals describe finite-test-set
sampling uncertainty for a fixed fitted model; they do not measure training-seed variation.
Macro scores average over every supplied class, assigning zero precision/recall/F1 where
undefined. Balanced accuracy averages recall over classes present in the labeled sample.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

CALIBRATION_BINS = 15
BOOTSTRAP_RESAMPLES = 1000


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide elementwise, defining an unsupported classification score as zero."""
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def _validate_inputs(
    labels: np.ndarray, probabilities: np.ndarray, class_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if (
        not isinstance(class_names, list)
        or len(class_names) < 2
        or not all(isinstance(name, str) and name for name in class_names)
        or len(set(class_names)) != len(class_names)
    ):
        raise ValueError("class_names must contain at least two distinct, nonempty strings")
    if labels.ndim != 1 or labels.size == 0 or labels.dtype.kind not in "iu":
        raise ValueError("labels must be a nonempty one-dimensional integer array")
    if probabilities.shape != (labels.size, len(class_names)):
        raise ValueError("probabilities must have shape (number of labels, number of classes)")
    if probabilities.dtype.kind not in "fiu":
        raise ValueError("probabilities must contain real numbers")
    probabilities = probabilities.astype(np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("probabilities must be between zero and one")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1, rtol=0, atol=1e-6):
        raise ValueError("each probability row must sum to one within 1e-6")
    if np.any(labels < 0) or np.any(labels >= len(class_names)):
        raise ValueError("labels must index the supplied class_names")
    # Remove accepted softmax rounding error before calculating probability scores.
    return labels.astype(np.int64), probabilities / row_sums


def _accuracy_interval(correct: int, count: int) -> dict:
    """Two-sided 95% Wilson score interval, including all-success/failure cases."""
    z = 1.959963984540054
    observed = correct / count
    denominator = 1 + z * z / count
    center = (observed + z * z / (2 * count)) / denominator
    radius = z * np.sqrt(observed * (1 - observed) / count + z * z / (4 * count**2))
    radius /= denominator
    return {
        "lower": float(max(0, center - radius)),
        "upper": float(min(1, center + radius)),
        "method": "wilson",
        "confidence_level": 0.95,
    }


def _macro_f1_interval(confusion: np.ndarray, seed: int) -> dict:
    """Percentile iid image bootstrap, sampled efficiently via confusion-cell counts."""
    count = int(confusion.sum())
    rng = np.random.default_rng(seed)
    samples = rng.multinomial(count, confusion.ravel() / count, size=BOOTSTRAP_RESAMPLES).reshape(
        BOOTSTRAP_RESAMPLES, *confusion.shape
    )
    true_positive = samples.diagonal(axis1=1, axis2=2)
    actual = samples.sum(axis=2)
    predicted = samples.sum(axis=1)
    macro_f1 = _divide(2 * true_positive, actual + predicted).mean(axis=1)
    lower, upper = np.quantile(macro_f1, [0.025, 0.975])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "method": "percentile_multinomial_confusion_bootstrap",
        "confidence_level": 0.95,
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": int(seed),
    }


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    seed: int = 42,
) -> dict:
    """Return JSON-safe metrics for single-label classification.

    Rows of ``probabilities`` correspond to labels; columns follow ``class_names``.
    Exact score ties favor the lower class index, including top-k scores. Top-k uses
    min(k, number of classes). Confusion matrices have actual rows and predicted columns.
    Missing actual-class rows in the normalized matrix are zero.

    NLL clips true-class probabilities to float64 machine epsilon before taking logs.
    Brier score sums squared error over classes, then averages over examples. ECE/MCE
    compare accuracy with the maximum predicted probability in 15 equal-width bins
    [lower, upper), with probability 1 included in the final bin. Empty-bin statistics
    and undefined confidence/AUC/AP scores are null. Macro AUC/AP are null unless every
    class occurs. Per-class AP is defined whenever that class has a positive example;
    per-class AUC additionally requires a negative example.
    """
    labels, probabilities = _validate_inputs(labels, probabilities, class_names)
    count, num_classes = probabilities.shape
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    confidence = probabilities.max(axis=1)
    confusion = np.bincount(labels * num_classes + predictions, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )
    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    true_positive = confusion.diagonal()
    precision = _divide(true_positive, predicted_support)
    recall = _divide(true_positive, support)
    f1 = _divide(2 * true_positive, support + predicted_support)
    normalized_confusion = _divide(confusion, support[:, None])
    ranked_classes = np.argsort(-probabilities, axis=1, kind="stable")

    per_class = []
    for class_index, name in enumerate(class_names):
        binary_labels = labels == class_index
        roc_auc = None
        average_precision = None
        if 0 < support[class_index] < count:
            roc_auc = float(roc_auc_score(binary_labels, probabilities[:, class_index]))
        if support[class_index] > 0:
            average_precision = float(
                average_precision_score(binary_labels, probabilities[:, class_index])
            )
        per_class.append(
            {
                "class": name,
                "support": int(support[class_index]),
                "precision": float(precision[class_index]),
                "recall": float(recall[class_index]),
                "f1": float(f1[class_index]),
                "roc_auc": roc_auc,
                "average_precision": average_precision,
            }
        )

    calibration_bins = []
    bin_indices = np.minimum((confidence * CALIBRATION_BINS).astype(int), CALIBRATION_BINS - 1)
    ece = 0.0
    mce = 0.0
    for bin_index in range(CALIBRATION_BINS):
        selected = bin_indices == bin_index
        bin_count = int(selected.sum())
        bin_accuracy = float(correct[selected].mean()) if bin_count else None
        bin_confidence = float(confidence[selected].mean()) if bin_count else None
        gap = abs(bin_accuracy - bin_confidence) if bin_count else None
        if bin_count:
            ece += bin_count / count * gap
            mce = max(mce, gap)
        calibration_bins.append(
            {
                "lower": bin_index / CALIBRATION_BINS,
                "upper": (bin_index + 1) / CALIBRATION_BINS,
                "count": bin_count,
                "accuracy": bin_accuracy,
                "confidence": bin_confidence,
                "gap": gap,
            }
        )

    true_probabilities = probabilities[np.arange(count), labels]
    # The identity avoids allocating a dense one-hot target matrix.
    brier = (np.square(probabilities).sum(axis=1) - 2 * true_probabilities + 1).mean()
    all_classes_present = bool(np.all(support > 0))
    return {
        "num_examples": int(count),
        "num_classes": int(num_classes),
        "class_names": list(class_names),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / count),
        "top3_accuracy": float(np.any(ranked_classes[:, :3] == labels[:, None], axis=1).mean()),
        "top5_accuracy": float(np.any(ranked_classes[:, :5] == labels[:, None], axis=1).mean()),
        "negative_log_likelihood": float(
            -np.log(np.maximum(true_probabilities, np.finfo(np.float64).eps)).mean()
        ),
        "brier_score": float(max(0.0, brier)),
        "ece": float(ece),
        "mce": float(mce),
        "mean_confidence": float(confidence.mean()),
        "confidence_correct": float(confidence[correct].mean()) if correct.any() else None,
        "confidence_incorrect": float(confidence[~correct].mean()) if (~correct).any() else None,
        "multiclass_roc_auc_ovr_macro": (
            float(np.mean([item["roc_auc"] for item in per_class])) if all_classes_present else None
        ),
        "average_precision_macro": (
            float(np.mean([item["average_precision"] for item in per_class]))
            if all_classes_present
            else None
        ),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "normalized_confusion_matrix": normalized_confusion.tolist(),
        "calibration_bins": calibration_bins,
        "accuracy_ci95": _accuracy_interval(int(correct.sum()), count),
        "macro_f1_ci95": _macro_f1_interval(confusion, seed),
    }
