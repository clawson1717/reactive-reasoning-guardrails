"""Tests for rrg.patterns module."""

from __future__ import annotations

import pytest

from rrg.patterns import (
    BasePatternDetector,
    PatternDetector,
    PatternMatch,
    PatternType,
    ReasoningStep,
    ReasoningTrace,
)


class TestPatternType:
    def test_all_have_values(self) -> None:
        assert PatternType.CIRCULAR_REASONING is not None
        assert PatternType.SELF_CONTRADICTION is not None
        assert PatternType.HALLUCINATION_INDICATOR is not None


class TestPatternMatch:
    def test_creation(self) -> None:
        match = PatternMatch(
            pattern_type=PatternType.CIRCULAR_REASONING,
            confidence=0.85,
            evidence="The argument loops back to itself",
            span=(10, 50),
        )
        assert match.pattern_type == PatternType.CIRCULAR_REASONING
        assert match.confidence == 0.85
        assert match.span == (10, 50)

    def test_default_span(self) -> None:
        match = PatternMatch(
            pattern_type=PatternType.CONTRADICTION,
            confidence=0.7,
            evidence="Statement contradicts earlier claim",
        )
        assert match.span is None

    def test_immutable(self) -> None:
        match = PatternMatch(
            pattern_type=PatternType.HALLUCINATION_INDICATOR,
            confidence=0.9,
            evidence="x",
        )
        with pytest.raises(Exception):  # frozen dataclass
            match.confidence = 0.5  # type: ignore


class TestReasoningTraceFixtures:
    def test_valid_trace(self, valid_trace: ReasoningTrace) -> None:
        assert len(valid_trace.steps) == 5
        assert valid_trace.final_answer == "Paris"

    def test_circular_trace(self, circular_trace: ReasoningTrace) -> None:
        assert circular_trace.metadata["issue"] == "circular"

    def test_self_contradiction_trace(
        self, self_contradiction_trace: ReasoningTrace
    ) -> None:
        assert "contradict" in str(self_contradiction_trace.metadata)

    def test_multi_step_trace(self, multi_step_trace: ReasoningTrace) -> None:
        assert len(multi_step_trace.steps) == 5


class TestPatternDetectorProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        # A class that satisfies the protocol
        class GoodDetector:
            @property
            def pattern_type(self) -> PatternType:
                return PatternType.CIRCULAR_REASONING

            def detect(self, trace: ReasoningTrace) -> list[PatternMatch]:
                return []

        assert isinstance(GoodDetector(), PatternDetector)
