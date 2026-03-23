"""Tests for rrg.core.orchestrator module — ReactiveReasoningLoop."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rrg.core import ReasoningAgent, ReasoningResult, ReasoningStep, ReasoningTrace
from rrg.core.orchestrator import (
    PatternDetectorRegistry,
    ReactiveReasoningLoop,
    ReactiveReasoningResult,
    ReasoningAuditLog,
    ReasoningAuditLogEntry,
)
from rrg.corrector import CorrectionAction, CorrectionResult, CorrectionStrategy
from rrg.monitor import GuardrailDecision, GuardrailMonitor
from rrg.patterns import PatternMatch, PatternType
from rrg.estimator import UncertaintyScore


class TestReasoningAuditLog:
    def test_record_appends_json_line(self) -> None:
        """ReasoningAuditLog.record() appends one JSON line to the log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.jsonl"
            audit_log = ReasoningAuditLog(log_path=log_path)

            entry = ReasoningAuditLogEntry(
                run_id="test-run-123",
                task="What is 2+2?",
                timestamp="2024-01-01T00:00:00Z",
                num_attempts=1,
                final_response="4",
                accepted=True,
                triggered_patterns=[],
                total_uncertainty_score=0.1,
                convergence_achieved=True,
                correction_actions=[],
                trace_summaries=["step 1: thinking"],
            )
            audit_log.record(entry)

            # Verify file contents
            assert log_path.exists()
            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 1

            parsed = json.loads(lines[0])
            assert parsed["run_id"] == "test-run-123"
            assert parsed["task"] == "What is 2+2?"
            assert parsed["accepted"] is True

    def test_record_multiple_entries(self) -> None:
        """Multiple record() calls append multiple JSON lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.jsonl"
            audit_log = ReasoningAuditLog(log_path=log_path)

            for i in range(3):
                entry = ReasoningAuditLogEntry(
                    run_id=f"run-{i}",
                    task=f"task {i}",
                    timestamp="2024-01-01T00:00:00Z",
                    num_attempts=1,
                    final_response=f"response {i}",
                    accepted=True,
                    correction_actions=[],
                    trace_summaries=[],
                )
                audit_log.record(entry)

            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 3


class TestReactiveReasoningLoop:
    def test_run_completes_without_error(self) -> None:
        """ReactiveReasoningLoop.run() completes without raising."""
        # Build minimal mocks
        mock_agent = MagicMock(spec=ReasoningAgent)
        mock_agent.run.return_value = ReasoningResult(
            trace=ReasoningTrace(
                steps=(ReasoningStep(1, "Thinking..."),),
                final_answer="answer",
                metadata={},
            ),
            response="final answer",
            accepted=True,
        )

        mock_monitor = MagicMock(spec=GuardrailMonitor)
        mock_monitor.run.return_value = GuardrailDecision(
            response="final answer",
            accepted=True,
            uncertainty_score=None,
            triggered_patterns=(),
        )

        loop = ReactiveReasoningLoop(
            agent=mock_agent,
            monitor=mock_monitor,
            max_attempts=2,
        )

        result = loop.run("What is 2+2?")

        assert isinstance(result, ReactiveReasoningResult)
        assert result.response == "final answer"
        assert result.num_attempts == 1
        mock_monitor.run.assert_called_once_with("What is 2+2?")

    def test_max_attempts_limit_respected(self) -> None:
        """Loop respects max_attempts and stops after that many calls."""
        mock_agent = MagicMock(spec=ReasoningAgent)
        mock_agent.run.return_value = ReasoningResult(
            trace=ReasoningTrace(
                steps=(ReasoningStep(1, "Thinking..."),),
                final_answer="answer",
                metadata={},
            ),
            response="response",
            accepted=False,
        )

        # Monitor always rejects, so no convergence
        mock_monitor = MagicMock(spec=GuardrailMonitor)
        mock_monitor.run.return_value = GuardrailDecision(
            response="response",
            accepted=False,
            uncertainty_score=UncertaintyScore(score=0.9, auroc=0.3, mean_agreement=0.2, n_samples=4),
            triggered_patterns=(
                PatternMatch(
                    pattern_type=PatternType.CIRCULAR_REASONING,
                    confidence=0.7,
                    evidence="evidence",
                ),
            ),
            correction_result=CorrectionResult(
                original_response="response",
                corrected_response=None,
                action=CorrectionAction(strategy=CorrectionStrategy.DECOMPOSE, reason="test"),
                accepted=False,
                attempts=1,
            ),
        )

        loop = ReactiveReasoningLoop(
            agent=mock_agent,
            monitor=mock_monitor,
            max_attempts=3,
        )

        result = loop.run("Test task")

        # Should run exactly max_attempts times
        assert mock_monitor.run.call_count == 3
        assert result.num_attempts == 3
        assert result.convergence_achieved is False

    def test_convergence_on_accepted_response(self) -> None:
        """Loop stops early when a response is accepted (convergence achieved)."""
        mock_agent = MagicMock(spec=ReasoningAgent)
        mock_agent.run.return_value = ReasoningResult(
            trace=ReasoningTrace(
                steps=(ReasoningStep(1, "Thinking..."),),
                final_answer="answer",
                metadata={},
            ),
            response="good response",
            accepted=True,
        )

        mock_monitor = MagicMock(spec=GuardrailMonitor)
        # First attempt accepted → should stop immediately
        mock_monitor.run.return_value = GuardrailDecision(
            response="good response",
            accepted=True,
            uncertainty_score=None,
            triggered_patterns=(),
        )

        loop = ReactiveReasoningLoop(
            agent=mock_agent,
            monitor=mock_monitor,
            max_attempts=5,
        )

        result = loop.run("Test")

        assert mock_monitor.run.call_count == 1
        assert result.convergence_achieved is True
        assert result.accepted is True

    def test_convergence_with_correction_acceptance_rate(self) -> None:
        """Convergence is detected when acceptance rate >= threshold."""
        mock_agent = MagicMock(spec=ReasoningAgent)
        mock_agent.run.return_value = ReasoningResult(
            trace=ReasoningTrace(
                steps=(ReasoningStep(1, "Thinking..."),),
                final_answer="answer",
                metadata={},
            ),
            response="response",
            accepted=False,
        )

        call_count = 0

        def mock_run(task: str) -> GuardrailDecision:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # First 2: corrections rejected
                return GuardrailDecision(
                    response="rejected",
                    accepted=False,
                    uncertainty_score=UncertaintyScore(score=0.8, auroc=0.2, mean_agreement=0.1, n_samples=4),
                    triggered_patterns=(
                        PatternMatch(pattern_type=PatternType.CIRCULAR_REASONING, confidence=0.7, evidence="x"),
                    ),
                    correction_result=CorrectionResult(
                        original_response="original",
                        corrected_response=None,
                        action=CorrectionAction(strategy=CorrectionStrategy.DECOMPOSE, reason="test"),
                        accepted=False,
                        attempts=1,
                    ),
                )
            else:
                # 3rd: correction accepted → convergence
                return GuardrailDecision(
                    response="corrected!",
                    accepted=True,
                    uncertainty_score=None,
                    triggered_patterns=(
                        PatternMatch(pattern_type=PatternType.CIRCULAR_REASONING, confidence=0.7, evidence="x"),
                    ),
                    correction_result=CorrectionResult(
                        original_response="original",
                        corrected_response="corrected!",
                        action=CorrectionAction(strategy=CorrectionStrategy.DECOMPOSE, reason="test"),
                        accepted=True,
                        attempts=1,
                    ),
                )

        mock_monitor = MagicMock(spec=GuardrailMonitor)
        mock_monitor.run.side_effect = mock_run

        loop = ReactiveReasoningLoop(
            agent=mock_agent,
            monitor=mock_monitor,
            max_attempts=5,
            convergence_threshold=0.5,
        )

        result = loop.run("Test")

        # Should stop at 3 attempts when last correction accepted
        assert mock_monitor.run.call_count == 3
        assert result.convergence_achieved is True
        assert result.response == "corrected!"

    def test_audit_log_recorded_on_run(self) -> None:
        """Each run produces an audit log entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            audit_log = ReasoningAuditLog(log_path=log_path)

            mock_agent = MagicMock(spec=ReasoningAgent)
            mock_agent.run.return_value = ReasoningResult(
                trace=ReasoningTrace(
                    steps=(ReasoningStep(1, "Thinking..."),),
                    final_answer="answer",
                    metadata={},
                ),
                response="response",
                accepted=True,
            )

            mock_monitor = MagicMock(spec=GuardrailMonitor)
            mock_monitor.run.return_value = GuardrailDecision(
                response="response",
                accepted=True,
                uncertainty_score=None,
                triggered_patterns=(),
            )

            loop = ReactiveReasoningLoop(
                agent=mock_agent,
                monitor=mock_monitor,
                audit_log=audit_log,
            )
            loop.run("Test task")

            assert log_path.exists()
            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["task"] == "Test task"
            assert parsed["num_attempts"] == 1
            assert parsed["convergence_achieved"] is True


class TestPatternDetectorRegistry:
    def test_registry_provides_all_four_detectors(self) -> None:
        """PatternDetectorRegistry returns all four pattern detectors."""
        registry = PatternDetectorRegistry()

        detectors = registry.detectors
        assert len(detectors) == 4

        pattern_types = {d.pattern_type for d in detectors}
        expected = {
            PatternType.EARLY_PRUNING,
            PatternType.PATH_LOCK_IN,
            PatternType.BOUNDARY_VIOLATION,
            PatternType.KNOWLEDGE_PRIORITIZATION_FAILURE,
        }
        assert pattern_types == expected

    def test_registry_can_register_with_monitor(self) -> None:
        """PatternDetectorRegistry.register_with() adds all detectors to a monitor."""
        mock_agent = MagicMock(spec=ReasoningAgent)
        monitor = GuardrailMonitor(agent=mock_agent)

        registry = PatternDetectorRegistry()
        registry.register_with(monitor)

        assert len(monitor.pattern_detectors) == 4
