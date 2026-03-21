"""GuardrailMonitor: orchestrates pattern detection and uncertainty estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from rrg.core import ReasoningAgent, ReasoningResult, ReasoningTrace
from rrg.corrector import CorrectionEngine, CorrectionResult, CorrectionStrategy
from rrg.estimator import UncertaintyEstimator, UncertaintyScore
from rrg.patterns import PatternDetector, PatternMatch, ReasoningStep

logger = structlog.get_logger()


@dataclass(frozen=True)
class GuardrailDecision:
    """Final decision from the guardrail monitor."""

    response: str
    accepted: bool
    uncertainty_score: UncertaintyScore | None
    triggered_patterns: tuple[PatternMatch, ...] = field(default_factory=tuple)
    correction_result: CorrectionResult | None = None
    bypass_allowed: bool = False  # True to force-accept despite flags


@dataclass
class GuardrailConfig:
    """Configuration for the guardrail monitor."""

    uncertainty_threshold: float = 0.5
    pattern_confidence_threshold: float = 0.6
    max_correction_attempts: int = 2
    enable_corrections: bool = True
    bypass_on_uncertainty: bool = False  # If True, bypass rather than correct


class GuardrailMonitor:
    """Orchestrates pattern detection and uncertainty estimation around a ReasoningAgent.

    The monitor wraps agent runs and:
    1. Executes the reasoning agent
    2. Runs pattern detectors on the trace
    3. Estimates uncertainty via the UncertaintyEstimator
    4. Applies corrections if issues are found
    5. Returns a GuardrailDecision with the final response
    """

    def __init__(
        self,
        agent: ReasoningAgent,
        pattern_detectors: list[PatternDetector] | None = None,
        uncertainty_estimator: UncertaintyEstimator | None = None,
        correction_engine: CorrectionEngine | None = None,
        config: GuardrailConfig | None = None,
    ) -> None:
        self.agent = agent
        self.pattern_detectors: list[PatternDetector] = pattern_detectors or []
        self.uncertainty_estimator = uncertainty_estimator
        self.correction_engine = correction_engine or CorrectionEngine()
        self.config = config or GuardrailConfig()
        self._logger = logger.bind(component="GuardrailMonitor")

    def run(self, prompt: str, **kwargs: Any) -> GuardrailDecision:
        """Run the agent and apply guardrails.

        Args:
            prompt: The user prompt to send to the reasoning agent.

        Returns:
            GuardrailDecision with the (possibly corrected) response.
        """
        self._logger.info("guardrail_run_start", prompt=prompt[:100])

        # 1. Run the reasoning agent
        result: ReasoningResult = self.agent.run(prompt, **kwargs)
        trace = result.trace

        # 2. Run pattern detectors
        pattern_matches = self._detect_patterns(trace)

        # 3. Estimate uncertainty
        uncertainty = self._estimate_uncertainty(trace)

        # 4. Decide whether to accept or correct
        issues_found = bool(pattern_matches) or (uncertainty and uncertainty.is_uncertain)

        if not issues_found or not self.config.enable_corrections:
            decision = GuardrailDecision(
                response=result.response,
                accepted=True,
                uncertainty_score=uncertainty,
                triggered_patterns=pattern_matches,
            )
            self._logger.info(
                "guardrail_accepted",
                patterns=len(pattern_matches),
                uncertainty=uncertainty.score if uncertainty else None,
            )
            return decision

        # 5. Attempt corrections
        correction_result = self.correction_engine.correct(
            response=result.response,
            trace=trace,
            pattern_matches=pattern_matches,
            uncertainty=uncertainty,
        )

        if correction_result.accepted and correction_result.corrected_response:
            final_response = correction_result.corrected_response
            accepted = True
        else:
            final_response = result.response
            accepted = not issues_found

        decision = GuardrailDecision(
            response=final_response,
            accepted=accepted,
            uncertainty_score=uncertainty,
            triggered_patterns=pattern_matches,
            correction_result=correction_result,
        )

        self._logger.info(
            "guardrail_complete",
            accepted=accepted,
            strategy=correction_result.action.strategy.name if correction_result else "none",
        )
        return decision

    def _detect_patterns(self, trace: ReasoningTrace) -> tuple[PatternMatch, ...]:
        """Run all registered pattern detectors."""
        all_matches: list[PatternMatch] = []
        for detector in self.pattern_detectors:
            try:
                matches = detector.detect(trace)
                all_matches.extend(matches)
            except Exception as exc:
                self._logger.warning("pattern_detector_error", detector=type(detector).__name__, error=str(exc))

        # Filter by confidence threshold
        threshold = self.config.pattern_confidence_threshold
        filtered = tuple(m for m in all_matches if m.confidence >= threshold)
        return filtered

    def _estimate_uncertainty(self, trace: ReasoningTrace) -> UncertaintyScore | None:
        """Estimate uncertainty from reasoning steps."""
        if self.uncertainty_estimator is None:
            return None

        try:
            # Treat consecutive steps as primary/reference samples
            steps = [s.content for s in trace.steps]
            return self.uncertainty_estimator.estimate_from_single(steps)
        except Exception as exc:
            self._logger.warning("uncertainty_estimation_error", error=str(exc))
            return None

    def register_detector(self, detector: PatternDetector) -> None:
        """Register a pattern detector."""
        self.pattern_detectors.append(detector)

    def __repr__(self) -> str:
        return (
            f"<GuardrailMonitor detectors={len(self.pattern_detectors)} "
            f"estimator={'yes' if self.uncertainty_estimator else 'no'}>"
        )


__all__ = [
    "GuardrailConfig",
    "GuardrailDecision",
    "GuardrailMonitor",
]
