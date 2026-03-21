"""Correction engine: applies remediation strategies to flagged reasoning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import structlog

from rrg.estimator import UncertaintyScore
from rrg.patterns import PatternMatch, PatternType

logger = structlog.get_logger()


class CorrectionStrategy(Enum):
    """Available correction strategies."""

    REASK = auto()  # Re-prompt with the same question
    DECOMPOSE = auto()  # Break the problem into sub-parts
    SELF_VERIFY = auto()  # Ask the model to verify its own answer
    ENSEMBLE = auto()  # Aggregate multiple responses
    ESCALATE = auto()  # Flag for human review
    NONE = auto()  # No correction needed


@dataclass(frozen=True)
class CorrectionAction:
    """A planned or taken correction step."""

    strategy: CorrectionStrategy
    reason: str
    triggered_by: PatternType | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrectionResult:
    """Outcome of applying corrections."""

    original_response: str
    corrected_response: str | None  # None if correction failed
    action: CorrectionAction
    accepted: bool  # whether to use the corrected response
    attempts: int = 1


class CorrectionStrategyHandler(ABC):
    """ABC for implementing a single correction strategy."""

    @property
    @abstractmethod
    def strategy(self) -> CorrectionStrategy:
        """Identifier for this handler's strategy."""

    @abstractmethod
    def apply(
        self,
        response: str,
        trace: Any,  # ReasoningTrace
        match: PatternMatch | None,
        uncertainty: UncertaintyScore | None,
    ) -> CorrectionResult:
        """Apply this correction strategy and return the result."""


class CorrectionEngine:
    """Orchestrates correction strategies based on detected patterns.

    The engine is configured with a set of handlers and selects among them
    based on the pattern type and uncertainty score.
    """

    def __init__(
        self,
        handlers: list[CorrectionStrategyHandler] | None = None,
        max_attempts: int = 2,
    ) -> None:
        self.handlers = {h.strategy: h for h in (handlers or [])}
        self.max_attempts = max_attempts
        self._logger = logger.bind(component="CorrectionEngine")

    def correct(
        self,
        response: str,
        trace: Any,  # ReasoningTrace
        pattern_matches: tuple[PatternMatch, ...],
        uncertainty: UncertaintyScore | None,
    ) -> CorrectionResult:
        """Select and apply the best correction strategy for the given issues."""
        strategy = self._select_strategy(pattern_matches, uncertainty)

        handler = self.handlers.get(strategy)
        if handler is None:
            return CorrectionResult(
                original_response=response,
                corrected_response=None,
                action=CorrectionAction(
                    strategy=strategy,
                    reason="No handler registered for selected strategy",
                ),
                accepted=False,
            )

        # Use the highest-confidence pattern match for context
        match = max(pattern_matches, key=lambda m: m.confidence) if pattern_matches else None

        result = handler.apply(response, trace, match, uncertainty)
        self._logger.info(
            "correction_applied",
            strategy=strategy.name,
            accepted=result.accepted,
            attempts=result.attempts,
        )
        return result

    def _select_strategy(
        self,
        pattern_matches: tuple[PatternMatch, ...],
        uncertainty: UncertaintyScore | None,
    ) -> CorrectionStrategy:
        """Select the most appropriate correction strategy."""
        if not pattern_matches:
            if uncertainty and uncertainty.is_uncertain:
                return CorrectionStrategy.SELF_VERIFY
            return CorrectionStrategy.NONE

        dominant = max(pattern_matches, key=lambda m: m.confidence)
        type_strategy_map: dict[PatternType, CorrectionStrategy] = {
            PatternType.CIRCULAR_REASONING: CorrectionStrategy.DECOMPOSE,
            PatternType.CONTRADICTION: CorrectionStrategy.SELF_VERIFY,
            PatternType.HALLUCINATION_INDICATOR: CorrectionStrategy.ENSEMBLE,
            PatternType.SELF_CONTRADICTION: CorrectionStrategy.SELF_VERIFY,
            PatternType.UNGROUNDED_ASSUMPTION: CorrectionStrategy.DECOMPOSE,
            PatternType.INCOMPLETE_REASONING: CorrectionStrategy.DECOMPOSE,
            PatternType.UNCERTAINTY_MISMATCH: CorrectionStrategy.REASK,
        }

        return type_strategy_map.get(
            dominant.pattern_type,
            CorrectionStrategy.ESCALATE if dominant.confidence > 0.8 else CorrectionStrategy.SELF_VERIFY,
        )

    def register_handler(self, handler: CorrectionStrategyHandler) -> None:
        """Register a correction strategy handler."""
        self.handlers[handler.strategy] = handler

    def __repr__(self) -> str:
        return f"<CorrectionEngine handlers={list(self.handlers.keys())}>"


__all__ = [
    "CorrectionAction",
    "CorrectionEngine",
    "CorrectionResult",
    "CorrectionStrategy",
    "CorrectionStrategyHandler",
]
