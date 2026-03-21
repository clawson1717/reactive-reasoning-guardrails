"""Tests for rrg.estimator module — UncertaintyEstimator."""

from __future__ import annotations

import numpy as np
import pytest

from rrg.estimator import EmbeddingModel, UncertaintyEstimator, UncertaintyScore
from tests.conftest import MockEmbeddingModel


class TestUncertaintyScore:
    def test_score_bounds(self) -> None:
        score = UncertaintyScore(score=0.5, auroc=0.5, mean_agreement=0.3, n_samples=10)
        assert 0.0 <= score.score <= 1.0

    def test_is_uncertain_threshold(self) -> None:
        score_uncertain = UncertaintyScore(score=0.6, auroc=0.4, mean_agreement=0.1, n_samples=5)
        score_certain = UncertaintyScore(score=0.3, auroc=0.7, mean_agreement=0.9, n_samples=5)
        assert score_uncertain.is_uncertain is True
        assert score_certain.is_uncertain is False

    def test_metadata_default(self) -> None:
        score = UncertaintyScore(score=0.5, auroc=0.5, mean_agreement=0.3, n_samples=4)
        assert score.metadata == {}


class TestMockEmbeddingModel:
    def test_embed_shape(self, mock_embedding: MockEmbeddingModel) -> None:
        vec = mock_embedding.embed("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (128,)

    def test_embed_deterministic(self, mock_embedding: MockEmbeddingModel) -> None:
        v1 = mock_embedding.embed("same text")
        v2 = mock_embedding.embed("same text")
        np.testing.assert_array_almost_equal(v1, v2)

    def test_embed_different_texts_differ(self, mock_embedding: MockEmbeddingModel) -> None:
        v1 = mock_embedding.embed("text a")
        v2 = mock_embedding.embed("text b")
        assert not np.allclose(v1, v2)

    def test_similarity_bounds(self, mock_embedding: MockEmbeddingModel) -> None:
        v1 = mock_embedding.embed("test")
        v2 = mock_embedding.embed("test")
        s = mock_embedding.similarity(v1, v2)
        assert -1.0 <= s <= 1.0


class TestUncertaintyEstimator:
    def test_estimate_insufficient_primary(self, mock_uncertainty_estimator: UncertaintyEstimator) -> None:
        score = mock_uncertainty_estimator.estimate([], ["a", "b"])
        assert score.score == 1.0
        assert score.n_samples == 0

    def test_estimate_insufficient_reference(self, mock_uncertainty_estimator: UncertaintyEstimator) -> None:
        score = mock_uncertainty_estimator.estimate(["a"], [])
        assert score.score == 1.0

    def test_estimate_single_sample_uncertain(self, mock_uncertainty_estimator: UncertaintyEstimator) -> None:
        score = mock_uncertainty_estimator.estimate_from_single(["a"])
        assert score.score == 1.0
        assert score.n_samples == 1

    def test_estimate_from_single_odd_number(
        self, mock_uncertainty_estimator: UncertaintyEstimator
    ) -> None:
        samples = ["apple", "banana", "cherry"]
        score = mock_uncertainty_estimator.estimate_from_single(samples)
        assert score.n_samples == 3

    def test_estimate_identical_samples(
        self, mock_uncertainty_estimator: UncertaintyEstimator
    ) -> None:
        # With deterministic mock embeddings, identical texts get identical
        # embeddings, so within-group and cross-group sims are the same,
        # yielding AUROC=0.5 and score=0.5.
        samples = ["the sky is blue"] * 4
        score = mock_uncertainty_estimator.estimate(samples[:2], samples[2:])
        assert 0.0 <= score.score <= 1.0
        assert 0.0 <= score.auroc <= 1.0
        assert score.mean_agreement == pytest.approx(1.0)

    def test_estimate_auroc_in_range(self, mock_uncertainty_estimator: UncertaintyEstimator) -> None:
        score = mock_uncertainty_estimator.estimate(
            ["apple", "fruit", "red"], ["car", "vehicle", "red"]
        )
        assert 0.0 <= score.auroc <= 1.0

    def test_estimate_two_groups(self, mock_uncertainty_estimator: UncertaintyEstimator) -> None:
        score = mock_uncertainty_estimator.estimate(
            ["a", "b"], ["c", "d"]
        )
        assert score.n_samples == 4
        assert 0.0 <= score.score <= 1.0
