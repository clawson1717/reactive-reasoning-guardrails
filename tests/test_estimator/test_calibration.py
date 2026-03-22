"""Tests for rrg.estimator.calibration module."""

from __future__ import annotations

import numpy as np
import pytest

from rrg.estimator.calibration import (
    AUROCMetrics,
    CalibrationDataset,
    compute_auroc,
    find_optimal_threshold,
)
from rrg.estimator.hybrid_estimator import UncertaintyEstimate
from rrg.patterns import ReasoningStep


# ---------------------------------------------------------------------------
# Tests: CalibrationDataset
# ---------------------------------------------------------------------------

class TestCalibrationDataset:
    def test_empty_dataset(self) -> None:
        ds = CalibrationDataset(pairs=[], name="test")
        assert len(ds) == 0

    def test_add_pair(self) -> None:
        ds = CalibrationDataset(pairs=[], name="test")
        step = ReasoningStep(1, "sample reasoning")
        ds.add(step, is_error=True)
        assert len(ds) == 1
        assert ds.pairs[0] == (step, True)

    def test_split_fraction(self) -> None:
        pairs = [(ReasoningStep(i, f"step {i}"), i % 2 == 1) for i in range(10)]
        ds = CalibrationDataset(pairs=pairs, name="split_test")
        first, second = ds.split(0.7)
        assert len(first) == 7
        assert len(second) == 3
        assert first.name == "split_test_train"
        assert second.name == "split_test_val"

    def test_split_fraction_min_size(self) -> None:
        pairs = [(ReasoningStep(i, f"step {i}"), False) for i in range(2)]
        ds = CalibrationDataset(pairs=pairs, name="tiny")
        first, second = ds.split(0.5)
        assert len(first) == 1
        assert len(second) == 1

    def test_len(self) -> None:
        ds = CalibrationDataset(
            pairs=[(ReasoningStep(i, f"s{i}"), False) for i in range(7)],
            name="len_test",
        )
        assert len(ds) == 7


# ---------------------------------------------------------------------------
# Tests: find_optimal_threshold
# ---------------------------------------------------------------------------

class TestFindOptimalThreshold:
    """Tests for Youden's J threshold optimization."""

    def test_empty_returns_05(self) -> None:
        tau = find_optimal_threshold(np.array([]), np.array([]))
        assert tau == 0.5

    def test_single_unique_score(self) -> None:
        tau = find_optimal_threshold(np.array([0.7]), np.array([1]))
        assert tau == 0.7

    def test_perfect_separation(self) -> None:
        # All errors score high (0.8-0.9), all corrects score low (0.1-0.2).
        scores = np.array([0.9, 0.85, 0.8, 0.88, 0.82] * 4)  # 20 errors
        labels = np.array([1] * 20 + [0] * 20)
        # Add correct scores in 0.1-0.2 range.
        correct_scores = np.linspace(0.1, 0.2, 20)
        all_scores = np.concatenate([scores, correct_scores])

        tau = find_optimal_threshold(all_scores, labels)
        # Threshold should be between 0.2 and 0.8 (inclusive).
        assert 0.2 <= tau <= 0.8

    def test_all_same_label(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        labels = np.array([1, 1, 1, 1, 1])
        tau = find_optimal_threshold(scores, labels)
        # With only one class, the returned tau is the mean score.
        assert tau == pytest.approx(0.3, rel=0.1)

    def test_reversed_signal(self) -> None:
        # Errors score LOW, correct scores HIGH — the method should still
        # find a threshold that gives J > 0.
        scores = np.array([0.1, 0.15, 0.2] * 10 + [0.8, 0.85, 0.9] * 10)
        labels = np.array([1] * 30 + [0] * 30)
        tau = find_optimal_threshold(scores, labels)
        # With reversed signal, optimal tau should still be somewhere sensible.
        assert 0.0 <= tau <= 1.0

    def test_youden_j_equivalence(self) -> None:
        # Verify that threshold found by Youden's J gives the max J.
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
        tau = find_optimal_threshold(scores, labels)

        best_j = -1.0
        best_tau = tau
        for t in scores:
            predicted = (scores >= float(t)).astype(int)
            tp = ((predicted == 1) & (labels == 1)).sum()
            fp = ((predicted == 1) & (labels == 0)).sum()
            tn = ((predicted == 0) & (labels == 0)).sum()
            fn = ((predicted == 0) & (labels == 1)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            j = tpr - fpr
            if j > best_j:
                best_j = j
                best_tau = float(t)

        assert tau == pytest.approx(best_tau, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: compute_auroc
# ---------------------------------------------------------------------------

class TestComputeAuroc:
    def test_empty_input(self) -> None:
        metrics = compute_auroc([], [])
        assert metrics.auroc == 0.5
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.optimal_threshold == 0.5

    def test_length_mismatch_raises(self) -> None:
        steps = [ReasoningStep(1, "a"), ReasoningStep(2, "b")]
        estimates = [UncertaintyEstimate(0.5, 0.5, 0.5, False)]
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_auroc(list(zip(steps, [False, True])), estimates)

    def test_perfect_auroc(self) -> None:
        # All errors have high uncertainty scores, all corrects have low.
        steps_errors = [ReasoningStep(i, f"error {i}") for i in range(10)]
        steps_correct = [ReasoningStep(i, f"correct {i}") for i in range(10)]
        pairs = list(zip(steps_errors, [True] * 10)) + list(zip(steps_correct, [False] * 10))

        # error estimates have score ~0.9, correct have score ~0.1
        estimates = (
            [UncertaintyEstimate(0.9, 0.1, 0.1, True)] * 10
            + [UncertaintyEstimate(0.1, 0.9, 0.9, False)] * 10
        )

        metrics = compute_auroc(pairs, estimates)
        assert metrics.auroc >= 0.99
        assert metrics.optimal_threshold == pytest.approx(0.5, abs=0.2)

    def test_random_auroc_approx_05(self) -> None:
        # Random predictions should give AUROC ≈ 0.5.
        rng = np.random.default_rng(42)
        steps = [ReasoningStep(i, f"step {i}") for i in range(100)]
        pairs = [(steps[i], bool(rng.integers(0, 2))) for i in range(100)]
        estimates = [
            UncertaintyEstimate(float(rng.random()), 0.5, 0.5, False)
            for _ in range(100)
        ]
        metrics = compute_auroc(pairs, estimates)
        assert 0.4 <= metrics.auroc <= 0.6

    def test_precision_recall_at_optimal(self) -> None:
        # Build a dataset where threshold cleanly separates.
        steps = [ReasoningStep(i, f"s{i}") for i in range(50)]
        # First 25 are errors with high scores, next 25 correct with low scores.
        pairs = [(steps[i], True) for i in range(25)] + [(steps[i], False) for i in range(25, 50)]
        estimates = (
            [UncertaintyEstimate(0.85, 0.2, 0.2, True)] * 25
            + [UncertaintyEstimate(0.15, 0.8, 0.8, False)] * 25
        )

        metrics = compute_auroc(pairs, estimates)
        assert metrics.precision > 0.8
        assert metrics.recall > 0.8
        assert metrics.f1 > 0.8

    def test_all_same_score(self) -> None:
        # When all scores are the same, AUROC = 0.5 and threshold = that score.
        steps = [ReasoningStep(i, f"s{i}") for i in range(10)]
        pairs = [(steps[i], i < 5) for i in range(10)]
        estimates = [UncertaintyEstimate(0.5, 0.5, 0.5, False)] * 10
        metrics = compute_auroc(pairs, estimates)
        assert metrics.auroc == 0.5

    def test_auroc_reversed_signal(self) -> None:
        # If errors get LOW scores and correct get HIGH, AUROC should be low.
        steps = [ReasoningStep(i, f"s{i}") for i in range(20)]
        pairs = [(steps[i], i < 10) for i in range(20)]
        # Errors have low score (0.1), correct have high score (0.9).
        estimates = (
            [UncertaintyEstimate(0.1, 0.9, 0.9, False)] * 10
            + [UncertaintyEstimate(0.9, 0.1, 0.1, True)] * 10
        )
        metrics = compute_auroc(pairs, estimates)
        # AUROC should be low (near 0) for reversed signal.
        assert metrics.auroc <= 0.1


# ---------------------------------------------------------------------------
# Tests: AUROCMetrics dataclass
# ---------------------------------------------------------------------------

class TestAUROCMetrics:
    def test_frozen_dataclass(self) -> None:
        metrics = AUROCMetrics(
            auroc=0.9, precision=0.8, recall=0.85, f1=0.825, optimal_threshold=0.5
        )
        with pytest.raises(AttributeError):
            metrics.auroc = 0.5  # type: ignore[attr-defined]

    def test_fields(self) -> None:
        metrics = AUROCMetrics(
            auroc=0.95, precision=0.9, recall=0.88, f1=0.89, optimal_threshold=0.45
        )
        assert metrics.auroc == 0.95
        assert metrics.precision == 0.9
        assert metrics.recall == 0.88
        assert metrics.f1 == 0.89
        assert metrics.optimal_threshold == 0.45
