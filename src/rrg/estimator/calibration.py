"""AUROC calibration utilities for uncertainty estimators."""
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class AUROCMetrics:
    auroc: float
    precision: float
    recall: float
    f1: float
    optimal_threshold: float


def find_optimal_threshold(
    uncertainty_scores: list[float] | np.ndarray,
    is_error_labels: list[bool] | np.ndarray,
) -> float:
    """Find optimal threshold using Youden's J statistic (maximizing recall - false_positive_rate)."""
    scores_arr = np.asarray(uncertainty_scores)
    labels_arr = np.asarray(is_error_labels, dtype=int)

    # Edge case: empty arrays
    if len(scores_arr) == 0 or len(labels_arr) == 0:
        return 0.5

    # Edge case: all same label
    if len(np.unique(labels_arr)) < 2:
        return float(np.mean(scores_arr))

    # Edge case: all same score
    if scores_arr.min() == scores_arr.max():
        return float(scores_arr.min())

    # Use actual score values as thresholds (not linspace) to match test expectations
    thresholds = np.unique(scores_arr)
    best_j = -1.0
    best_t = float(scores_arr.mean())

    for t in thresholds:
        preds = (scores_arr >= t).astype(int)
        tp = int(np.sum((preds == 1) & (labels_arr == 1)))
        fp = int(np.sum((preds == 1) & (labels_arr == 0)))
        tn = int(np.sum((preds == 0) & (labels_arr == 0)))
        fn = int(np.sum((preds == 0) & (labels_arr == 1)))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = float(t)

    return best_t


def compute_auroc(
    reasoning_pairs: list[tuple[Any, bool]],
    uncertainty_estimates: list[Any],  # list of UncertaintyEstimate
) -> AUROCMetrics:
    """
    Compute AUROC and related metrics for uncertainty estimates against error labels.
    """
    if len(reasoning_pairs) != len(uncertainty_estimates):
        raise ValueError(
            f"Length mismatch: reasoning_pairs has {len(reasoning_pairs)} items, "
            f"uncertainty_estimates has {len(uncertainty_estimates)} items"
        )

    scores = np.array([e.score for e in uncertainty_estimates])
    labels = np.array([int(is_err) for _, is_err in reasoning_pairs])

    if len(np.unique(labels)) < 2:
        return AUROCMetrics(auroc=0.5, precision=0.0, recall=0.0, f1=0.0, optimal_threshold=0.5)

    auroc = roc_auc_score(labels, scores)

    optimal_t = find_optimal_threshold(scores.tolist(), labels.tolist())
    preds = (scores >= optimal_t).astype(int)

    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    return AUROCMetrics(
        auroc=float(auroc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        optimal_threshold=float(optimal_t),
    )


class CalibrationDataset:
    """Dataset of (reasoning_step, is_error) pairs for calibration."""
    def __init__(self, pairs: list[tuple[Any, bool]] | None = None, name: str = "calibration"):
        self.pairs: list[tuple[Any, bool]] = pairs if pairs is not None else []
        self.name = name

    def add(self, reasoning_step: Any, is_error: bool) -> None:
        self.pairs.append((reasoning_step, is_error))

    def __len__(self) -> int:
        return len(self.pairs)

    def split(self, test_ratio: float = 0.2):
        """Split into train/test as CalibrationDataset objects."""
        import random
        pairs = self.pairs.copy()
        random.shuffle(pairs)
        split_idx = int(len(pairs) * (1 - test_ratio))
        base_name = self.name
        train_ds = CalibrationDataset(pairs=pairs[:split_idx], name=f"{base_name}_train")
        val_ds = CalibrationDataset(pairs=pairs[split_idx:], name=f"{base_name}_val")
        return train_ds, val_ds
