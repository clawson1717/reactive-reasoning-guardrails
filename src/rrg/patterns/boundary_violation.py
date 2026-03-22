"""BoundaryViolationDetector — detects when reasoning exceeds permitted boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rrg.patterns import (
    BasePatternDetector,
    PatternMatch,
    PatternType,
    ReasoningTrace,
)

if TYPE_CHECKING:
    pass


# Default dangerous keywords scanned when no explicit BoundarySpec is provided.
_DEFAULT_DANGEROUS_KEYWORDS: tuple[str, ...] = (
    "delete",
    "sudo",
    "rm -rf",
    "DROP TABLE",
    "exec(",
    "eval(",
)


@dataclass(frozen=True)
class BoundarySpec:
    """Specification of allowed tools, domains, and constraints.

    Attributes:
        allowed_tools: Set of tool names the agent is permitted to use.
        prohibited_actions: Set of action names or keywords that are forbidden.
        max_reasoning_steps: Maximum number of reasoning steps allowed.
        domain_scope: Set of domain keywords that are within scope.
    """

    allowed_tools: frozenset[str]
    prohibited_actions: frozenset[str]
    max_reasoning_steps: int
    domain_scope: frozenset[str]


class BoundaryViolationDetector(BasePatternDetector):
    """Detects when reasoning exceeds permitted boundaries.

    Maintains BoundarySpec; intercepts tool calls and intermediate
    conclusions against spec.

    If no BoundarySpec is provided, uses a permissive default that only
    scans for dangerous keywords ("delete", "sudo", "rm -rf", etc.).
    """

    def __init__(
        self,
        boundary_spec: BoundarySpec | None = None,
        dangerous_keywords: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize the detector.

        Args:
            boundary_spec: Optional specification of allowed boundaries.
                If None, uses permissive defaults with dangerous keyword scanning.
            dangerous_keywords: Keywords to scan for when no explicit spec is set.
        """
        self._boundary_spec = boundary_spec
        self._dangerous_keywords = dangerous_keywords or _DEFAULT_DANGEROUS_KEYWORDS

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.BOUNDARY_VIOLATION

    def _scan_for_dangerous_keywords(self, text: str) -> list[str]:
        """Return list of dangerous keywords found in text."""
        found: list[str] = []
        text_lower = text.lower()
        for keyword in self._dangerous_keywords:
            if keyword.lower() in text_lower:
                found.append(keyword)
        return found

    def _check_prohibited_actions(self, text: str, prohibited: frozenset[str]) -> list[str]:
        """Return prohibited action keywords found in text."""
        found: list[str] = []
        text_lower = text.lower()
        for action in prohibited:
            if action.lower() in text_lower:
                found.append(action)
        return found

    def _check_out_of_scope_tools(self, text: str, allowed_tools: frozenset[str]) -> list[str]:
        """Return tool names mentioned that are not in allowed_tools."""
        # Simple keyword extraction: look for camelCase or snake_case identifiers
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text)
        mentioned_tools = {w for w in words if w[0].islower() and len(w) > 2}
        out_of_scope: list[str] = []
        for tool in mentioned_tools:
            if tool not in allowed_tools and any(
                tool.lower() in action.lower() or action.lower() in tool.lower()
                for action in ["tool", "call", "invoke", "execute", "run"]
            ):
                out_of_scope.append(tool)
        return out_of_scope

    def detect(self, trace: ReasoningTrace) -> list[PatternMatch]:
        """Scan the trace for boundary violations."""
        matches: list[PatternMatch] = []
        spec = self._boundary_spec
        steps = trace.steps

        # --- Violation 1: Exceeded max reasoning steps ---
        if spec is not None and len(steps) > spec.max_reasoning_steps:
            evidence = (
                f"Reasoning trace has {len(steps)} steps, "
                f"exceeding the maximum of {spec.max_reasoning_steps}."
            )
            severity = min(1.0, (len(steps) - spec.max_reasoning_steps) / spec.max_reasoning_steps + 0.5)
            matches.append(
                PatternMatch(
                    pattern_type=PatternType.BOUNDARY_VIOLATION,
                    confidence=severity,
                    evidence=evidence,
                    span=None,
                )
            )

        # Combine all step content for scanning
        all_text = "\n".join(s.content for s in steps)

        if spec is not None:
            # --- Violation 2: Prohibited actions ---
            prohibited_found = self._check_prohibited_actions(
                all_text, spec.prohibited_actions
            )
            if prohibited_found:
                evidence = (
                    f"Found prohibited action(s)/keyword(s): {prohibited_found}. "
                    f"Allowed actions do not include these."
                )
                matches.append(
                    PatternMatch(
                        pattern_type=PatternType.BOUNDARY_VIOLATION,
                        confidence=0.9,
                        evidence=evidence,
                        span=None,
                    )
                )

            # --- Violation 3: Out-of-scope tools ---
            out_of_scope = self._check_out_of_scope_tools(all_text, spec.allowed_tools)
            if out_of_scope:
                evidence = (
                    f"Potentially out-of-scope tool(s) mentioned: {out_of_scope}. "
                    f"Allowed tools: {spec.allowed_tools}."
                )
                matches.append(
                    PatternMatch(
                        pattern_type=PatternType.BOUNDARY_VIOLATION,
                        confidence=0.75,
                        evidence=evidence,
                        span=None,
                    )
                )

            # --- Violation 4: Out-of-scope domain ---
            if spec.domain_scope:
                text_lower = all_text.lower()
                domain_mentions = sum(
                    1 for kw in spec.domain_scope if kw.lower() in text_lower
                )
                if domain_mentions == 0 and len(steps) > 2:
                    # Reasoning doesn't mention any domain keywords — might be off-topic
                    evidence = (
                        f"No domain-specific keywords from scope {spec.domain_scope} "
                        f"found in {len(steps)} reasoning steps. Reasoning may be off-topic."
                    )
                    matches.append(
                        PatternMatch(
                            pattern_type=PatternType.BOUNDARY_VIOLATION,
                            confidence=0.6,
                            evidence=evidence,
                            span=None,
                        )
                    )
        else:
            # No explicit spec: scan for dangerous keywords
            dangerous_found = self._scan_for_dangerous_keywords(all_text)
            if dangerous_found:
                evidence = (
                    f"Potentially dangerous keyword(s)/pattern(s) found: {dangerous_found}. "
                    f"These patterns may indicate unsafe operations."
                )
                matches.append(
                    PatternMatch(
                        pattern_type=PatternType.BOUNDARY_VIOLATION,
                        confidence=0.85,
                        evidence=evidence,
                        span=None,
                    )
                )

        return matches

    def __repr__(self) -> str:
        spec = self._boundary_spec
        if spec:
            return (
                f"BoundaryViolationDetector("
                f"max_reasoning_steps={spec.max_reasoning_steps}, "
                f"allowed_tools={set(spec.allowed_tools)}, "
                f"prohibited_actions={set(spec.prohibited_actions)})"
            )
        return (
            f"BoundaryViolationDetector("
            f"dangerous_keywords={self._dangerous_keywords})"
        )
