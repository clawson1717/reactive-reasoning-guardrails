"""Core reasoning agent wrapper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import structlog

from rrg.patterns import PatternMatch, ReasoningTrace, ReasoningStep

logger = structlog.get_logger()

T = TypeVar("T")

# Orchestrator (must be imported after defining base classes to avoid circular deps)
from rrg.core.orchestrator import (
    PatternDetectorRegistry,
    ReactiveReasoningLoop,
    ReactiveReasoningResult,
    ReasoningAuditLog,
    ReasoningAuditLogEntry,
)


@dataclass
class ReasoningResult:
    """Output from a reasoning agent run."""

    trace: ReasoningTrace
    response: str
    accepted: bool  # True if guardrails accepted the response
    triggered_patterns: tuple[PatternMatch, ...] = field(default_factory=tuple)
    uncertainty_score: float | None = None  # 0.0=confident, 1.0=uncertain


class LLMBackend(ABC):
    """Abstract interface for an LLM backend.

    Implement this to connect to any LLM API (OpenAI, Anthropic, local, etc.).
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from a text prompt."""

    @abstractmethod
    def generate_with_reasoning(
        self, prompt: str, **kwargs: Any
    ) -> tuple[str, list[ReasoningStep]]:
        """Generate a response with structured reasoning steps."""


@dataclass
class ReasoningAgentConfig:
    """Configuration for a ReasoningAgent."""

    max_steps: int = 10
    temperature: float = 0.7
    timeout_seconds: float = 30.0
    enable_cot: bool = True


class ReasoningAgent(ABC):
    """Main LLM reasoning agent wrapper.

    Subclass this and implement `llm_backend` to connect to any LLM.
    """

    def __init__(
        self,
        config: ReasoningAgentConfig | None = None,
        backend: LLMBackend | None = None,
    ) -> None:
        self.config = config or ReasoningAgentConfig()
        self.llm_backend = backend
        self._logger = logger.bind(component="ReasoningAgent")

    @abstractmethod
    def run(self, prompt: str, **kwargs: Any) -> ReasoningResult:
        """Run the agent on a prompt and return structured result."""

    def _build_trace(self, steps: list[ReasoningStep], final_answer: str) -> ReasoningTrace:
        """Build a ReasoningTrace from step list and final answer."""
        return ReasoningTrace(
            steps=tuple(steps),
            final_answer=final_answer,
            metadata={"agent": self.__class__.__name__, "config": asdict_safe(self.config)},
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} max_steps={self.config.max_steps}>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def asdict_safe(obj: object) -> dict[str, Any]:
    """Safely convert a dataclass to dict, handling non-serialisable fields."""
    import dataclasses

    if dataclasses.is_dataclass(obj):
        result: dict[str, Any] = {}
        for k, v in dataclasses.asdict(obj).items():
            try:
                import json

                json.dumps(v)
                result[k] = v
            except TypeError:
                result[k] = repr(v)
        return result
    return {"repr": repr(obj)}
