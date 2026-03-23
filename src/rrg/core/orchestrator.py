"""ReactiveReasoningLoop — main orchestrator for reactive reasoning with guardrails.

Coordinates ReasoningAgent, GuardrailMonitor, and iterative correction
in a convergence loop with full audit logging.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from rrg.corrector import CorrectionResult
from rrg.estimator import UncertaintyScore
from rrg.patterns import (
    PatternDetector,
    PatternMatch,
    EarlyPruningDetector,
    PathLockInDetector,
    BoundaryViolationDetector,
    KnowledgeGuidedPrioritizationChecker,
)

if TYPE_CHECKING:
    from rrg.core import ReasoningAgent, ReasoningResult, ReasoningTrace
    from rrg.monitor import GuardrailMonitor, GuardrailDecision

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Audit Logging
# ---------------------------------------------------------------------------


@dataclass
class ReasoningAuditLogEntry:
    """Single audit log entry for one full ReactiveReasoningLoop run."""

    run_id: str
    task: str
    timestamp: str  # ISO-8601
    num_attempts: int
    final_response: str
    accepted: bool
    triggered_patterns: list[tuple[str, float]] = field(default_factory=list)
    total_uncertainty_score: float | None = None
    convergence_achieved: bool = False
    correction_actions: list[tuple[str, bool]] = field(default_factory=list)
    trace_summaries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "timestamp": self.timestamp,
            "num_attempts": self.num_attempts,
            "final_response": self.final_response,
            "accepted": self.accepted,
            "triggered_patterns": self.triggered_patterns,
            "total_uncertainty_score": self.total_uncertainty_score,
            "convergence_achieved": self.convergence_achieved,
            "correction_actions": self.correction_actions,
            "trace_summaries": self.trace_summaries,
        }


class ReasoningAuditLog:
    """File-backed audit logger for ReactiveReasoningLoop runs.

    Appends one JSON line per run to ``rrg_audit_log.jsonl``.
    """

    def __init__(self, log_path: str | Path | None = None) -> None:
        self.log_path = Path(log_path) if log_path else Path("rrg_audit_log.jsonl")
        self._logger = logger.bind(component="ReasoningAuditLog")

    def record(self, entry: ReasoningAuditLogEntry) -> None:
        """Append a JSON line for the given audit entry."""
        try:
            line = json.dumps(entry.to_dict(), ensure_ascii=False)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._logger.debug("audit_logged", run_id=entry.run_id, path=str(self.log_path))
        except Exception as exc:
            self._logger.error("audit_log_write_failed", run_id=entry.run_id, error=str(exc))
            raise


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ReactiveReasoningResult:
    """Output of a full ReactiveReasoningLoop run."""

    response: str
    accepted: bool
    num_attempts: int
    convergence_achieved: bool
    audit_log_entry: ReasoningAuditLogEntry
    triggered_patterns: tuple[PatternMatch, ...] = field(default_factory=tuple)
    uncertainty_score: UncertaintyScore | None = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ReactiveReasoningLoop:
    """Main orchestrator for reactive reasoning with guardrails.

    Wraps a ReasoningAgent with GuardrailMonitor and runs an iterative
    correction loop until convergence or max_attempts is reached.

    Convergence is achieved when:
    - The last GuardrailDecision.accepted is True, OR
    - The correction acceptance rate across all attempts >= convergence_threshold
    """

    def __init__(
        self,
        agent: ReasoningAgent,
        monitor: GuardrailMonitor,
        audit_log: ReasoningAuditLog | None = None,
        max_attempts: int = 3,
        convergence_threshold: float = 0.8,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.agent = agent
        self.monitor = monitor
        self.audit_log = audit_log or ReasoningAuditLog()
        self.max_attempts = max_attempts
        self.convergence_threshold = convergence_threshold
        self.timeout_seconds = timeout_seconds
        self._logger = logger.bind(component="ReactiveReasoningLoop")

    def run(self, task: str) -> ReactiveReasoningResult:
        """Run the reactive reasoning loop on a task.

        Returns ReactiveReasoningResult with final answer, audit info, etc.
        """
        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        self._logger.info(
            "rrl_run_start",
            run_id=run_id,
            task=task[:100],
            max_attempts=self.max_attempts,
        )

        all_decisions: list[GuardrailDecision] = []
        all_correction_results: list[CorrectionResult] = []

        for attempt in range(1, self.max_attempts + 1):
            self._logger.info("rrl_attempt_start", run_id=run_id, attempt=attempt)

            decision = self.monitor.run(task)
            all_decisions.append(decision)

            if decision.correction_result:
                all_correction_results.append(decision.correction_result)

            # Build trace summaries
            trace_summaries: list[str] = []
            if hasattr(self.agent, "run"):
                # Attempt to extract trace from last decision via agent
                # The monitor stores nothing persistent, so we synthesize a summary
                pass
            trace_summaries.append(f"attempt_{attempt}: {decision.response[:80]}...")

            self._logger.info(
                "rrl_attempt_complete",
                run_id=run_id,
                attempt=attempt,
                accepted=decision.accepted,
                patterns=len(decision.triggered_patterns),
            )

            # Convergence check: last decision accepted OR acceptance rate high enough
            if self._check_convergence(all_correction_results):
                self._logger.info("rrl_converged", run_id=run_id, attempt=attempt)
                break

        # Build final result from last decision
        last = all_decisions[-1]
        num_attempts = len(all_decisions)
        convergence_achieved = self._check_convergence(all_correction_results)

        # Collect triggered patterns across all attempts
        all_patterns: list[PatternMatch] = []
        for d in all_decisions:
            all_patterns.extend(d.triggered_patterns)

        # Correction actions summary
        correction_actions: list[tuple[str, bool]] = []
        for cr in all_correction_results:
            correction_actions.append((cr.action.strategy.name, cr.accepted))

        # Build audit entry
        audit_entry = ReasoningAuditLogEntry(
            run_id=run_id,
            task=task,
            timestamp=timestamp,
            num_attempts=num_attempts,
            final_response=last.response,
            accepted=last.accepted,
            triggered_patterns=[(pt.pattern_type.name, pt.confidence) for pt in last.triggered_patterns],
            total_uncertainty_score=last.uncertainty_score.score if last.uncertainty_score else None,
            convergence_achieved=convergence_achieved,
            correction_actions=correction_actions,
            trace_summaries=trace_summaries,
        )

        # Record to audit log
        try:
            self.audit_log.record(audit_entry)
        except Exception as exc:
            self._logger.warning("audit_record_failed", run_id=run_id, error=str(exc))

        self._logger.info(
            "rrl_run_complete",
            run_id=run_id,
            num_attempts=num_attempts,
            convergence_achieved=convergence_achieved,
        )

        return ReactiveReasoningResult(
            response=last.response,
            accepted=last.accepted,
            num_attempts=num_attempts,
            convergence_achieved=convergence_achieved,
            audit_log_entry=audit_entry,
            triggered_patterns=tuple(all_patterns),
            uncertainty_score=last.uncertainty_score,
        )

    def _check_convergence(self, correction_results: list[CorrectionResult]) -> bool:
        """Check if corrections have converged.

        Convergence is achieved when:
        - The last correction was accepted, OR
        - The acceptance rate across all correction attempts >= convergence_threshold
        """
        if not correction_results:
            # No corrections attempted means no issues found = converged
            return True

        last = correction_results[-1]
        if last.accepted:
            return True

        accepted_count = sum(1 for cr in correction_results if cr.accepted)
        acceptance_rate = accepted_count / len(correction_results)
        return acceptance_rate >= self.convergence_threshold


# ---------------------------------------------------------------------------
# Pattern Detector Registry
# ---------------------------------------------------------------------------


class PatternDetectorRegistry:
    """Pre-configured registry of all four RRG pattern detectors.

    Provides a simple way to instantiate and register all detectors
    with a GuardrailMonitor.
    """

    # Default salient terms for general reasoning tasks
    DEFAULT_SALIENT_TERMS: list[str] = [
        "reasoning",
        "evidence",
        "conclusion",
        "assumption",
        "premise",
        "logic",
    ]

    def __init__(
        self,
        early_pruning: EarlyPruningDetector | None = None,
        path_lockin: PathLockInDetector | None = None,
        boundary_violation: BoundaryViolationDetector | None = None,
        knowledge_prioritization: KnowledgeGuidedPrioritizationChecker | None = None,
        knowledge_salient_terms: list[str] | None = None,
    ) -> None:
        self._detectors: list[PatternDetector] = []
        self._early_pruning = early_pruning or EarlyPruningDetector()
        self._path_lockin = path_lockin or PathLockInDetector()
        self._boundary_violation = boundary_violation or BoundaryViolationDetector()
        self._knowledge_prioritization = (
            knowledge_prioritization
            or KnowledgeGuidedPrioritizationChecker(
                salient_terms=knowledge_salient_terms or self.DEFAULT_SALIENT_TERMS
            )
        )

    @property
    def detectors(self) -> list[PatternDetector]:
        """All registered pattern detectors."""
        return [
            self._early_pruning,
            self._path_lockin,
            self._boundary_violation,
            self._knowledge_prioritization,
        ]

    def register_with(self, monitor: GuardrailMonitor) -> None:
        """Register all detectors with a GuardrailMonitor."""
        for detector in self.detectors:
            monitor.register_detector(detector)
