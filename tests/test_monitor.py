"""Tests for rrg.monitor module — GuardrailMonitor."""

from __future__ import annotations

import pytest

from rrg.monitor import GuardrailConfig, GuardrailDecision, GuardrailMonitor
from rrg.patterns import PatternMatch, PatternType
from tests.conftest import MockReasoningAgent, MockPatternDetector


class TestGuardrailConfig:
    def test_defaults(self) -> None:
        cfg = GuardrailConfig()
        assert cfg.uncertainty_threshold == 0.5
        assert cfg.pattern_confidence_threshold == 0.6
        assert cfg.enable_corrections is True
        assert cfg.max_correction_attempts == 2

    def test_custom_config(self) -> None:
        cfg = GuardrailConfig(uncertainty_threshold=0.3, enable_corrections=False)
        assert cfg.uncertainty_threshold == 0.3
        assert cfg.enable_corrections is False


class TestGuardrailDecision:
    def test_creation(self) -> None:
        decision = GuardrailDecision(
            response="the answer is 42",
            accepted=True,
            uncertainty_score=None,
            triggered_patterns=(),
        )
        assert decision.response == "the answer is 42"
        assert decision.accepted is True

    def test_bypass_default(self) -> None:
        decision = GuardrailDecision(response="x", accepted=True, uncertainty_score=None)
        assert decision.bypass_allowed is False


class TestGuardrailMonitor:
    def test_run_accepts_clean_response(self, simple_monitor: GuardrailMonitor) -> None:
        decision = simple_monitor.run("What is 2+2?")
        assert decision.accepted is True
        assert isinstance(decision.response, str)

    def test_run_returns_trace(self, simple_monitor: GuardrailMonitor) -> None:
        decision = simple_monitor.run("What is 2+2?")
        assert decision.triggered_patterns is not None

    def test_run_with_no_detectors(self, mock_agent: MockReasoningAgent) -> None:
        monitor = GuardrailMonitor(agent=mock_agent, pattern_detectors=[])
        decision = monitor.run("test prompt")
        assert decision.accepted is True
        assert len(decision.triggered_patterns) == 0

    def test_run_with_detector_triggered(
        self, mock_agent: MockReasoningAgent, valid_trace: ReasoningTrace
    ) -> None:
        match = PatternMatch(
            pattern_type=PatternType.CIRCULAR_REASONING,
            confidence=0.9,
            evidence="circular",
        )
        detector = MockPatternDetector(PatternType.CIRCULAR_REASONING, [match])
        monitor = GuardrailMonitor(
            agent=mock_agent,
            pattern_detectors=[detector],
            config=GuardrailConfig(enable_corrections=False),
        )
        decision = monitor.run("test")
        # With no corrections, even flagged responses accepted
        assert isinstance(decision.response, str)

    def test_run_with_corrections_enabled_but_no_handler(
        self, mock_agent: MockReasoningAgent
    ) -> None:
        """When corrections enabled but no handler, falls back gracefully."""
        match = PatternMatch(
            pattern_type=PatternType.CONTRADICTION,
            confidence=0.9,
            evidence="contradiction",
        )
        detector = MockPatternDetector(PatternType.CONTRADICTION, [match])
        monitor = GuardrailMonitor(
            agent=mock_agent,
            pattern_detectors=[detector],
            correction_engine=None,
            config=GuardrailConfig(enable_corrections=True),
        )
        decision = monitor.run("test")
        # Should not crash — falls back to no correction
        assert isinstance(decision.response, str)

    def test_detect_patterns_filters_by_threshold(self, mock_agent: MockReasoningAgent) -> None:
        """Patterns below confidence threshold should be filtered."""
        low_conf = PatternMatch(
            pattern_type=PatternType.CIRCULAR_REASONING,
            confidence=0.3,  # below default 0.6 threshold
            evidence="maybe circular",
        )
        high_conf = PatternMatch(
            pattern_type=PatternType.SELF_CONTRADICTION,
            confidence=0.8,  # above threshold
            evidence="definitely contradictory",
        )
        detector = MockPatternDetector(PatternType.CIRCULAR_REASONING, [low_conf, high_conf])
        monitor = GuardrailMonitor(
            agent=mock_agent,
            pattern_detectors=[detector],
            config=GuardrailConfig(enable_corrections=False),
        )
        decision = monitor.run("test")
        # Only high_conf should be included
        assert len(decision.triggered_patterns) == 1
        assert decision.triggered_patterns[0].confidence >= 0.6

    def test_register_detector(self, simple_monitor: GuardrailMonitor) -> None:
        detector = MockPatternDetector(
            PatternType.CIRCULAR_REASONING,
            [PatternMatch(PatternType.CIRCULAR_REASONING, 0.9, "circular")],
        )
        simple_monitor.register_detector(detector)
        assert len(simple_monitor.pattern_detectors) == 1

    def test_repr(self, simple_monitor: GuardrailMonitor) -> None:
        r = repr(simple_monitor)
        assert "GuardrailMonitor" in r
