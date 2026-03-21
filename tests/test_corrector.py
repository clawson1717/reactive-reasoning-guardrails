"""Tests for rrg.corrector module — CorrectionEngine."""

from __future__ import annotations

import pytest

from rrg.corrector import (
    CorrectionAction,
    CorrectionEngine,
    CorrectionResult,
    CorrectionStrategy,
    CorrectionStrategyHandler,
)
from rrg.patterns import PatternMatch, PatternType
from rrg.estimator import UncertaintyScore


class MockCorrectionHandler(CorrectionStrategyHandler):
    """Concrete mock handler for testing."""

    def __init__(self, strategy: CorrectionStrategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> CorrectionStrategy:
        return self._strategy

    def apply(
        self,
        response: str,
        trace,  # ReasoningTrace
        match: PatternMatch | None,
        uncertainty: UncertaintyScore | None,
    ) -> CorrectionResult:
        return CorrectionResult(
            original_response=response,
            corrected_response="corrected: " + response,
            action=CorrectionAction(strategy=self._strategy, reason="mock"),
            accepted=True,
            attempts=1,
        )


class TestCorrectionStrategy:
    def test_all_strategies_exist(self) -> None:
        assert CorrectionStrategy.REASK is not None
        assert CorrectionStrategy.DECOMPOSE is not None
        assert CorrectionStrategy.SELF_VERIFY is not None
        assert CorrectionStrategy.ENSEMBLE is not None
        assert CorrectionStrategy.ESCALATE is not None
        assert CorrectionStrategy.NONE is not None


class TestCorrectionAction:
    def test_creation(self) -> None:
        action = CorrectionAction(
            strategy=CorrectionStrategy.REASK,
            reason="Low confidence",
            triggered_by=PatternType.CIRCULAR_REASONING,
            confidence=0.8,
        )
        assert action.strategy == CorrectionStrategy.REASK
        assert action.triggered_by == PatternType.CIRCULAR_REASONING
        assert action.confidence == 0.8

    def test_default_confidence(self) -> None:
        action = CorrectionAction(strategy=CorrectionStrategy.NONE, reason="no issues")
        assert action.confidence == 1.0


class TestCorrectionResult:
    def test_creation(self) -> None:
        result = CorrectionResult(
            original_response="orig",
            corrected_response="corrected",
            action=CorrectionAction(strategy=CorrectionStrategy.REASK, reason="test"),
            accepted=True,
            attempts=1,
        )
        assert result.original_response == "orig"
        assert result.corrected_response == "corrected"
        assert result.accepted is True


class TestCorrectionEngine:
    def test_empty_engine_no_handlers(self, empty_correction_engine: CorrectionEngine) -> None:
        result = empty_correction_engine.correct(
            response="test",
            trace=None,
            pattern_matches=(),
            uncertainty=None,
        )
        assert result.corrected_response is None
        assert result.accepted is False

    def test_register_handler(self, empty_correction_engine: CorrectionEngine) -> None:
        handler = MockCorrectionHandler(CorrectionStrategy.REASK)
        empty_correction_engine.register_handler(handler)
        assert CorrectionStrategy.REASK in empty_correction_engine.handlers

    def test_correct_with_matching_handler(
        self, empty_correction_engine: CorrectionEngine
    ) -> None:
        handler = MockCorrectionHandler(CorrectionStrategy.DECOMPOSE)
        empty_correction_engine.register_handler(handler)

        # Need a pattern match so a strategy is selected
        match = PatternMatch(
            pattern_type=PatternType.CIRCULAR_REASONING,
            confidence=0.9,
            evidence="circular",
        )
        result = empty_correction_engine.correct(
            response="test response",
            trace=None,
            pattern_matches=(match,),
            uncertainty=None,
        )
        assert result.accepted is True
        assert result.corrected_response == "corrected: test response"

    def test_selects_none_when_no_issues(
        self, empty_correction_engine: CorrectionEngine
    ) -> None:
        handler = MockCorrectionHandler(CorrectionStrategy.REASK)
        empty_correction_engine.register_handler(handler)

        result = empty_correction_engine.correct(
            response="test",
            trace=None,
            pattern_matches=(),
            uncertainty=None,
        )
        assert result.action.strategy == CorrectionStrategy.NONE

    def test_selects_by_pattern_type(
        self, empty_correction_engine: CorrectionEngine
    ) -> None:
        handler = MockCorrectionHandler(CorrectionStrategy.DECOMPOSE)
        empty_correction_engine.register_handler(handler)

        match = PatternMatch(
            pattern_type=PatternType.CIRCULAR_REASONING,
            confidence=0.9,
            evidence="circular argument",
        )
        result = empty_correction_engine.correct(
            response="test",
            trace=None,
            pattern_matches=(match,),
            uncertainty=None,
        )
        assert result.action.strategy == CorrectionStrategy.DECOMPOSE

    def test_selects_self_verify_on_self_contradiction(
        self, empty_correction_engine: CorrectionEngine
    ) -> None:
        handler = MockCorrectionHandler(CorrectionStrategy.SELF_VERIFY)
        empty_correction_engine.register_handler(handler)

        match = PatternMatch(
            pattern_type=PatternType.SELF_CONTRADICTION,
            confidence=0.7,
            evidence="contradiction",
        )
        result = empty_correction_engine.correct(
            response="test",
            trace=None,
            pattern_matches=(match,),
            uncertainty=None,
        )
        assert result.action.strategy == CorrectionStrategy.SELF_VERIFY

    def test_selects_self_verify_on_uncertainty(
        self, empty_correction_engine: CorrectionEngine
    ) -> None:
        handler = MockCorrectionHandler(CorrectionStrategy.SELF_VERIFY)
        empty_correction_engine.register_handler(handler)

        uncertainty = UncertaintyScore(score=0.7, auroc=0.3, mean_agreement=0.1, n_samples=5)
        result = empty_correction_engine.correct(
            response="test",
            trace=None,
            pattern_matches=(),
            uncertainty=uncertainty,
        )
        assert result.action.strategy == CorrectionStrategy.SELF_VERIFY

    def test_repr(self, empty_correction_engine: CorrectionEngine) -> None:
        r = repr(empty_correction_engine)
        assert "CorrectionEngine" in r
