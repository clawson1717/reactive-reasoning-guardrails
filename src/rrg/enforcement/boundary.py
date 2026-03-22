"""Boundary specification types and violation errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class StepLabel:
    """A labelled reasoning step parsed from structured output.

    Attributes:
        label: The raw label text (e.g. "Step 1", "Step 2").
        step_type: The categorical type of the step.
        metadata: Additional parsed metadata about the step.
    """

    label: str
    step_type: Literal["premise", "analysis", "revision", "conclusion"]
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BoundarySpec:
    """Specification of operational boundaries for an agent.

    Attributes:
        allowed_tools: Tools the agent may call. If empty, all tools are allowed
            unless explicitly prohibited.
        prohibited_tools: Tools explicitly forbidden regardless of allowed_tools.
        prohibited_domains: Topics/domains the agent may not reason about.
            Detected via keyword matching.
        max_reasoning_steps: Maximum reasoning steps before escalation.
            None means unlimited.
        max_tool_calls_per_step: Maximum tool calls allowed per reasoning step.
            Defaults to 5.
        domain_whitelist: If set, only these domains are allowed.
            None means all non-prohibited domains are allowed.
    """

    allowed_tools: list[str] = field(default_factory=list)
    prohibited_tools: list[str] = field(default_factory=list)
    prohibited_domains: list[str] = field(default_factory=list)
    max_reasoning_steps: int | None = None
    max_tool_calls_per_step: int = 5
    domain_whitelist: list[str] | None = None


class BoundaryViolationError(Exception):
    """Raised when an agent action violates a BoundarySpec.

    Attributes:
        violated_spec: Human-readable name of the spec field that was violated.
        reason: Detailed explanation of the violation.
    """

    def __init__(self, violated_spec: str, reason: str) -> None:
        self.violated_spec = violated_spec
        self.reason = reason
        super().__init__(f"[{violated_spec}] {reason}")
