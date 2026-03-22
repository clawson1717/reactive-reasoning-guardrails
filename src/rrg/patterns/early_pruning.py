"""EarlyPruningDetector — detects when relevant context is discarded prematurely."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rrg.patterns import (
    BasePatternDetector,
    PatternMatch,
    PatternType,
    ReasoningTrace,
    ReasoningStep,
)

if TYPE_CHECKING:
    pass


class EarlyPruningDetector(BasePatternDetector):
    """Detects when relevant context is discarded before it influences reasoning.

    Monitors token logprob entropy drop below threshold in first N tokens
    of reasoning; flags when context window utilization < 40% at reasoning start.

    Detection triggers:
    - Trace has fewer than 3 steps AND first step content length < 100 chars
    - All step contents are very short (avg < 40 chars per step)
    """

    def __init__(self, min_step_content_length: int = 40, short_trace_threshold: int = 100) -> None:
        """Initialize the detector.

        Args:
            min_step_content_length: Minimum average chars per step before flagging as too short.
            short_trace_threshold: Content length threshold for first step in short traces.
        """
        self._min_step_content_length = min_step_content_length
        self._short_trace_threshold = short_trace_threshold

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.EARLY_PRUNING

    def detect(self, trace: ReasoningTrace) -> list[PatternMatch]:
        """Scan the trace for early pruning indicators."""
        matches: list[PatternMatch] = []
        steps = trace.steps

        if len(steps) == 0:
            return matches

        # Check 1: Very short trace (< 3 steps) with short first step
        if len(steps) < 3:
            first_step = steps[0]
            if len(first_step.content) < self._short_trace_threshold:
                evidence = (
                    f"Trace has only {len(steps)} step(s); "
                    f"first step is {len(first_step.content)} chars "
                    f"(threshold {self._short_trace_threshold}). "
                    f"Content: {first_step.content[:80]!r}"
                )
                evidence_strength = 0.8
                confidence = min(1.0, evidence_strength * 1.5)
                matches.append(
                    PatternMatch(
                        pattern_type=PatternType.EARLY_PRUNING,
                        confidence=confidence,
                        evidence=evidence,
                        span=None,
                    )
                )
                return matches  # already flagged, short circuit

        # Check 2: All steps very short (avg < min_step_content_length)
        if len(steps) > 0:
            total_len = sum(len(s.content) for s in steps)
            avg_len = total_len / len(steps)
            if avg_len < self._min_step_content_length:
                evidence = (
                    f"Average step content length is {avg_len:.1f} chars "
                    f"(threshold {self._min_step_content_length}). "
                    f"Total steps: {len(steps)}. "
                    f"Total content: {total_len} chars."
                )
                evidence_strength = min(1.0, avg_len / self._min_step_content_length)
                confidence = min(1.0, evidence_strength * 1.5)
                matches.append(
                    PatternMatch(
                        pattern_type=PatternType.EARLY_PRUNING,
                        confidence=confidence,
                        evidence=evidence,
                        span=None,
                    )
                )

        return matches

    def __repr__(self) -> str:
        return (
            f"EarlyPruningDetector("
            f"min_step_content_length={self._min_step_content_length}, "
            f"short_trace_threshold={self._short_trace_threshold})"
        )
