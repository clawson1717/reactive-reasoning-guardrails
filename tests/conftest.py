"""Shared pytest fixtures for reactive-reasoning-guardrails tests."""

from __future__ import annotations

import pytest
import numpy as np

from rrg.core import ReasoningAgent, ReasoningAgentConfig, ReasoningResult
from rrg.core import ReasoningStep, ReasoningTrace
from rrg.patterns import PatternMatch, PatternType, PatternDetector
from rrg.estimator import EmbeddingModel, UncertaintyEstimator, UncertaintyScore
from rrg.corrector import (
    CorrectionEngine,
    CorrectionResult,
    CorrectionStrategy,
    CorrectionStrategyHandler,
    CorrectionAction,
)
from rrg.monitor import GuardrailMonitor, GuardrailConfig


# ---------------------------------------------------------------------------
# Synthetic Reasoning Trace Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_trace() -> ReasoningTrace:
    """A clean reasoning trace with no obvious issues."""
    return ReasoningTrace(
        steps=(
            ReasoningStep(1, "The user asks about the capital of France."),
            ReasoningStep(2, "I recall that Paris is the capital of France."),
            ReasoningStep(3, "Paris has a population of approximately 2.1 million."),
            ReasoningStep(4, "France is located in Western Europe."),
            ReasoningStep(5, "Therefore, the capital of France is Paris."),
        ),
        final_answer="Paris",
        metadata={"topic": "geography", "source": "fixture"},
    )


@pytest.fixture
def circular_trace() -> ReasoningTrace:
    """A trace exhibiting circular reasoning."""
    return ReasoningTrace(
        steps=(
            ReasoningStep(1, "We need to determine if X is true."),
            ReasoningStep(2, "Assuming X is true, then Y follows."),
            ReasoningStep(3, "Since Y follows, X must be true."),
            ReasoningStep(4, "Therefore X is true because X implies Y and Y implies X."),
        ),
        final_answer="X is true",
        metadata={"issue": "circular", "source": "fixture"},
    )


@pytest.fixture
def self_contradiction_trace() -> ReasoningTrace:
    """A trace that contradicts itself mid-reasoning."""
    return ReasoningTrace(
        steps=(
            ReasoningStep(1, "The sky appears blue during the day."),
            ReasoningStep(2, "However, the sky is actually not blue."),
            ReasoningStep(3, "Wait, I just said it's not blue but also blue earlier."),
            ReasoningStep(4, "Let me reconsider the evidence."),
            ReasoningStep(5, "The sky is blue because of Rayleigh scattering."),
        ),
        final_answer="The sky is blue",
        metadata={"issue": "self-contradiction", "source": "fixture"},
    )


@pytest.fixture
def hallucination_trace() -> ReasoningTrace:
    """A trace with potential hallucination indicators."""
    return ReasoningTrace(
        steps=(
            ReasoningStep(1, "A study published in 1923 showed X."),
            ReasoningStep(2, "According to Dr. Fake Name's 2024 paper, Y is true."),
            ReasoningStep(3, "This is widely known fact Z with no citation."),
            ReasoningStep(4, "Therefore we conclude Z."),
        ),
        final_answer="Z is true",
        metadata={"issue": "hallucination", "source": "fixture"},
    )


@pytest.fixture
def incomplete_trace() -> ReasoningTrace:
    """A trace with incomplete reasoning (jumps to conclusion)."""
    return ReasoningTrace(
        steps=(
            ReasoningStep(1, "This complex problem has many variables."),
            ReasoningStep(2, "It's obviously A."),
            ReasoningStep(3, "Done."),
        ),
        final_answer="A",
        metadata={"issue": "incomplete", "source": "fixture"},
    )


@pytest.fixture
def multi_step_trace() -> ReasoningTrace:
    """A longer, more complex reasoning trace."""
    return ReasoningTrace(
        steps=(
            ReasoningStep(1, "Problem: What is 17 × 23?"),
            ReasoningStep(2, "Break it down: 17 × 20 = 340"),
            ReasoningStep(3, "Then 17 × 3 = 51"),
            ReasoningStep(4, "Add them: 340 + 51 = 391"),
            ReasoningStep(5, "Verification: 391 ÷ 23 = 17 ✓"),
        ),
        final_answer="391",
        metadata={"topic": "math", "source": "fixture"},
    )


# ---------------------------------------------------------------------------
# Mock LLM Backend
# ---------------------------------------------------------------------------

class MockLLMBackend:
    """Deterministic mock LLM backend for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["Mock response"]
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> str:
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return self.responses[idx]

    def generate_with_reasoning(self, prompt: str, **kwargs):
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        steps = [
            ReasoningStep(1, f"Thinking about: {prompt[:50]}"),
            ReasoningStep(2, f"Elaborating on key points"),
        ]
        return self.responses[idx], steps


# ---------------------------------------------------------------------------
# Mock Embedding Model
# ---------------------------------------------------------------------------

class MockEmbeddingModel(EmbeddingModel):
    """Deterministic mock embedding model using random vectors."""

    def __init__(self, dim: int = 128, seed: int = 42) -> None:
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def embed(self, text: str) -> np.ndarray:
        # Deterministically map text to a vector
        h = hash(text) % (2**31)
        rng = np.random.default_rng(h)
        return rng.standard_normal(self.dim)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        # Cosine similarity
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Mock Reasoning Agent
# ---------------------------------------------------------------------------

class MockReasoningAgent(ReasoningAgent):
    """Mock reasoning agent using MockLLMBackend."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.backend = MockLLMBackend()
        self.llm_backend = self.backend

    def run(self, prompt: str, **kwargs) -> ReasoningResult:
        response, steps = self.backend.generate_with_reasoning(prompt, **kwargs)
        trace = self._build_trace(steps, response)
        return ReasoningResult(
            trace=trace,
            response=response,
            accepted=True,
        )


# ---------------------------------------------------------------------------
# Mock Pattern Detector
# ---------------------------------------------------------------------------

class MockPatternDetector(PatternDetector):
    """Mock detector that returns configurable matches."""

    def __init__(self, pattern_type: PatternType, matches: list[PatternMatch]) -> None:
        self._pattern_type = pattern_type
        self._matches = matches

    @property
    def pattern_type(self) -> PatternType:
        return self._pattern_type

    def detect(self, trace: ReasoningTrace) -> list[PatternMatch]:
        return self._matches


# ---------------------------------------------------------------------------
# Mock Correction Handler
# ---------------------------------------------------------------------------

class MockCorrectionHandler(CorrectionStrategyHandler):
    """Mock correction handler that always succeeds with a fixed response."""

    def __init__(self, strategy: CorrectionStrategy, corrected: str = "corrected") -> None:
        self._strategy = strategy
        self._corrected = corrected

    @property
    def strategy(self) -> CorrectionStrategy:
        return self._strategy

    def apply(
        self,
        response: str,
        trace: ReasoningTrace,
        match: PatternMatch | None,
        uncertainty: UncertaintyScore | None,
    ) -> CorrectionResult:
        return CorrectionResult(
            original_response=response,
            corrected_response=self._corrected,
            action=CorrectionAction(strategy=self._strategy, reason="mock handler"),
            accepted=True,
            attempts=1,
        )


# ---------------------------------------------------------------------------
# Agent Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_agent() -> MockReasoningAgent:
    return MockReasoningAgent()


@pytest.fixture
def mock_embedding() -> MockEmbeddingModel:
    return MockEmbeddingModel()


@pytest.fixture
def mock_uncertainty_estimator(mock_embedding: MockEmbeddingModel) -> UncertaintyEstimator:
    return UncertaintyEstimator(embedding_model=mock_embedding, sample_size=4)


@pytest.fixture
def empty_correction_engine() -> CorrectionEngine:
    return CorrectionEngine()


@pytest.fixture
def simple_monitor(mock_agent: MockReasoningAgent) -> GuardrailMonitor:
    return GuardrailMonitor(agent=mock_agent)
