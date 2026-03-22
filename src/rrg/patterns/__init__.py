"""Pattern detector base classes for reactive reasoning guardrails."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, runtime_checkable


class PatternType(Enum):
    """Categories of reasoning patterns to detect."""

    CIRCULAR_REASONING = auto()
    CONTRADICTION = auto()
    HALLUCINATION_INDICATOR = auto()
    UNCERTAINTY_MISMATCH = auto()
    INCOMPLETE_REASONING = auto()
    SELF_CONTRADICTION = auto()
    UNGROUNDED_ASSUMPTION = auto()
    EARLY_PRUNING = auto()
    PATH_LOCK_IN = auto()
    BOUNDARY_VIOLATION = auto()
    KNOWLEDGE_PRIORITIZATION_FAILURE = auto()


@dataclass(frozen=True)
class PatternMatch:
    """Result of a pattern detector scan."""

    pattern_type: PatternType
    confidence: float  # 0.0 to 1.0
    evidence: str  # human-readable excerpt from the trace
    span: tuple[int, int] | None = None  # character offsets in source


@runtime_checkable
class PatternDetector(Protocol):
    """Protocol for pattern detection strategies."""

    @property
    def pattern_type(self) -> PatternType:
        """The type of pattern this detector identifies."""

    def detect(self, trace: ReasoningTrace) -> list[PatternMatch]:
        """Scan a reasoning trace and return any pattern matches."""


class BasePatternDetector(ABC):
    """Base class for pattern detectors with shared utilities."""

    @property
    @abstractmethod
    def pattern_type(self) -> PatternType:
        """The type of pattern this detector identifies."""

    @abstractmethod
    def detect(self, trace: ReasoningTrace) -> list[PatternMatch]:
        """Scan a reasoning trace and return any pattern matches."""


@dataclass(frozen=True)
class ReasoningTrace:
    """Immutable record of a reasoning session."""

    steps: tuple[ReasoningStep, ...]
    final_answer: str
    metadata: dict[str, object]

    def __repr__(self) -> str:
        return f"<ReasoningTrace steps={len(self.steps)}>"


@dataclass(frozen=True)
class ReasoningStep:
    """A single step in a reasoning chain."""

    step_id: int
    content: str
    is_cot: bool = True  # chain-of-thought marker

    def __repr__(self) -> str:
        preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return f"<ReasoningStep {self.step_id}: {preview}>"


# ---------------------------------------------------------------------------
# Exports — step 2: four implicit pattern detectors
# ---------------------------------------------------------------------------
from rrg.patterns.early_pruning import EarlyPruningDetector
from rrg.patterns.path_lockin import PathLockInDetector
from rrg.patterns.boundary_violation import BoundaryViolationDetector, BoundarySpec
from rrg.patterns.knowledge_prioritization import KnowledgeGuidedPrioritizationChecker

__all__ = [
    # Base
    "PatternType",
    "PatternMatch",
    "PatternDetector",
    "BasePatternDetector",
    "ReasoningTrace",
    "ReasoningStep",
    # Step 2 detectors
    "EarlyPruningDetector",
    "PathLockInDetector",
    "BoundaryViolationDetector",
    "BoundarySpec",
    "KnowledgeGuidedPrioritizationChecker",
]
