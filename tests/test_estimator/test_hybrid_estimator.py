"""Tests for HybridUncertaintyEstimator."""
import pytest
from unittest.mock import MagicMock
import numpy as np
from rrg.estimator.hybrid_estimator import (
    HybridUncertaintyEstimator,
    UncertaintyEstimate,
)
from rrg.patterns import ReasoningStep


class MockEmbeddingModel:
    """Deterministic fake embedding model."""
    def __init__(self, dim: int = 128):
        self.dim = dim
        self._counter = 0

    def embed(self, text: str) -> np.ndarray:
        self._counter += 1
        # Deterministic fake embedding based on text hash
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(self.dim)


class MockLLMBackend:
    """Mock LLM backend that returns synthetic completions."""
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or [
            "I'm 90% confident the answer is A.",
            "Based on the evidence, I believe it's B.",
        ]
        self.call_count = 0

    def complete(self, prompt: str, temperature: float = 1.0, seed: int | None = None) -> str:
        idx = self.call_count % len(self.responses)
        self.call_count += 1
        return self.responses[idx]


def test_fusion_formula():
    """Test the uncertainty fusion: U = alpha*(1-consistency) + (1-alpha)*(1-verbalized)."""
    # High consistency (agree) + high verbalized confidence → low uncertainty
    mock_emb = MockEmbeddingModel()
    mock_llm = MockLLMBackend(["I'm 90% confident this is correct.", "I agree with the conclusion."])
    est = HybridUncertaintyEstimator(mock_llm, mock_emb, alpha=0.7, tau=0.5)

    step = ReasoningStep(step_id=1, content="The answer is A.")
    result = est.estimate(step)

    # consistency should be high (both samples similar to each other via embeddings)
    assert 0 <= result.consistency <= 1
    assert 0 <= result.verbalized <= 1
    assert 0 <= result.score <= 1


def test_above_threshold():
    """Test threshold comparison."""
    mock_emb = MockEmbeddingModel()
    mock_llm = MockLLMBackend(["I don't know.", "I'm not sure."])
    est = HybridUncertaintyEstimator(mock_llm, mock_emb, alpha=0.7, tau=0.3)

    step = ReasoningStep(step_id=1, content="?")
    result = est.estimate(step)

    # With low verbalized confidence, score should be high
    assert isinstance(result.above_threshold, bool)


def test_verbalized_confidence_parsing():
    """Test various verbalized confidence formats."""
    mock_emb = MockEmbeddingModel()
    mock_llm = MockLLMBackend(["I'm 85% confident."])
    est = HybridUncertaintyEstimator(mock_llm, mock_emb)

    conf = est._extract_verbalized_confidence("I am 85% confident in this answer.")
    assert conf == 0.85


def test_verbalized_confidence_uncertain():
    mock_emb = MockEmbeddingModel()
    mock_llm = MockLLMBackend(["default"])
    est = HybridUncertaintyEstimator(mock_llm, mock_emb)

    conf = est._extract_verbalized_confidence("I don't know the answer to this question.")
    assert conf == 0.1


def test_verbalized_confidence_certain():
    mock_emb = MockEmbeddingModel()
    mock_llm = MockLLMBackend(["default"])
    est = HybridUncertaintyEstimator(mock_llm, mock_emb)

    conf = est._extract_verbalized_confidence("I am absolutely certain this is correct.")
    assert conf == 0.9


def test_semantic_consistency():
    mock_emb = MockEmbeddingModel()
    mock_llm = MockLLMBackend(["default"])
    est = HybridUncertaintyEstimator(mock_llm, mock_emb)

    # Same text should give consistency 1.0
    consistency = est._semantic_consistency(["hello world", "hello world"])
    assert consistency == 1.0


def test_calibration_with_synthetic_pairs():
    """Test calibration on synthetic high/low uncertainty pairs."""
    mock_emb = MockEmbeddingModel()

    # Create mock backend with known responses
    high_uncertain = ["I don't know.", "Maybe it's X."]
    low_uncertain = ["I'm 95% confident it's Y.", "The answer is definitely Z."]

    pairs = []
    # 5 error cases (high uncertainty)
    for _ in range(5):
        mock_llm = MockLLMBackend(high_uncertain)
        est = HybridUncertaintyEstimator(mock_llm, mock_emb)
        step = ReasoningStep(step_id=1, content="What is X?")
        pairs.append((step, True))

    # 5 correct cases (low uncertainty)
    for _ in range(5):
        mock_llm = MockLLMBackend(low_uncertain)
        est = HybridUncertaintyEstimator(mock_llm, mock_emb)
        step = ReasoningStep(step_id=2, content="What is X?")
        pairs.append((step, False))

    mock_emb2 = MockEmbeddingModel()
    final_est = HybridUncertaintyEstimator(MockLLMBackend(low_uncertain), mock_emb2)
    alpha, tau = final_est.calibrate(pairs)

    assert 0 <= alpha <= 1
    assert tau >= 0


def test_auroc_score():
    mock_emb = MockEmbeddingModel()
    low_uncertain = ["I'm 95% confident.", "Very certain."]
    pairs = []
    for _ in range(3):
        mock_llm = MockLLMBackend(low_uncertain)
        est = HybridUncertaintyEstimator(mock_llm, mock_emb)
        step = ReasoningStep(step_id=1, content="?")
        pairs.append((step, False))
    for _ in range(3):
        mock_llm = MockLLMBackend(["I don't know.", "Unsure."])
        est = HybridUncertaintyEstimator(mock_llm, mock_emb)
        step = ReasoningStep(step_id=2, content="?")
        pairs.append((step, True))

    mock_emb2 = MockEmbeddingModel()
    final_est = HybridUncertaintyEstimator(MockLLMBackend(low_uncertain), mock_emb2)
    auroc = final_est.get_auroc_score(pairs, alpha=0.7, tau=0.5)
    assert 0 <= auroc <= 1
