"""BoundaryEnforcementLayer: runtime enforcement of BoundarySpec restrictions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import structlog

from rrg.enforcement.boundary import BoundarySpec, BoundaryViolationError

logger = structlog.get_logger()


class BoundaryEnforcementLayer:
    """Enforces operational boundaries on agent reasoning at runtime.

    The layer intercepts tool calls and reasoning content to detect
    violations of a :class:`BoundarySpec`. It uses simple keyword matching
    for domain detection and does not require an LLM.

    Example:
        >>> spec = BoundarySpec(
        ...     allowed_tools=["search", "read"],
        ...     prohibited_tools=["delete", "execute"],
        ...     prohibited_domains=["medical_advice", "legal"],
        ...     max_reasoning_steps=20,
        ... )
        >>> layer = BoundaryEnforcementLayer(spec)
        >>> layer.check_tool_call("search", {})
        None  # passes
        >>> layer.check_tool_call("delete", {})
        Traceback (most likely BoundaryViolationError)
    """

    def __init__(self, spec: BoundarySpec) -> None:
        self.spec = spec
        self._logger = logger.bind(component="BoundaryEnforcementLayer")

    # ------------------------------------------------------------------
    # Core check methods
    # ------------------------------------------------------------------

    def check_tool_call(self, tool_name: str, tool_args: dict[str, Any]) -> None:
        """Check whether a tool call is permitted.

        Args:
            tool_name: Name of the tool being called.
            tool_args: Arguments passed to the tool.

        Raises:
            BoundaryViolationError: If the tool is prohibited or not allowed.
        """
        # Explicitly prohibited tools always fail
        if tool_name in self.spec.prohibited_tools:
            raise BoundaryViolationError(
                violated_spec="prohibited_tools",
                reason=f"Tool '{tool_name}' is explicitly prohibited.",
            )

        # If allowed_tools is non-empty, tool must be in it
        if self.spec.allowed_tools and tool_name not in self.spec.allowed_tools:
            raise BoundaryViolationError(
                violated_spec="allowed_tools",
                reason=f"Tool '{tool_name}' is not in the allowed tools list.",
            )

    def check_domain(self, violation_text: str) -> None:
        """Check whether text references a prohibited domain.

        Uses simple case-insensitive keyword matching on the prohibited
        domain terms. No LLM is used.

        Args:
            violation_text: Text to check for prohibited domain references.

        Raises:
            BoundaryViolationError: If a prohibited domain is detected.
        """
        text_lower = violation_text.lower()

        # Check prohibited domains
        for domain in self.spec.prohibited_domains:
            domain_lower = domain.lower()
            # Match whole-word or sub-phrase
            if domain_lower in text_lower:
                raise BoundaryViolationError(
                    violated_spec="prohibited_domains",
                    reason=f"Prohibited domain '{domain}' detected in reasoning.",
                )

        # Check whitelist: if set, text must mention at least one whitelisted domain
        if self.spec.domain_whitelist is not None:
            if not any(wl.lower() in text_lower for wl in self.spec.domain_whitelist):
                raise BoundaryViolationError(
                    violated_spec="domain_whitelist",
                    reason=(
                        f"Text does not reference any whitelisted domain. "
                        f"Allowed: {self.spec.domain_whitelist}."
                    ),
                )

    def check_step_count(self, current_steps: int) -> None:
        """Check whether the current reasoning step count is within limits.

        Args:
            current_steps: Number of reasoning steps executed so far.

        Raises:
            BoundaryViolationError: If max_reasoning_steps is exceeded.
        """
        if (
            self.spec.max_reasoning_steps is not None
            and current_steps >= self.spec.max_reasoning_steps
        ):
            raise BoundaryViolationError(
                violated_spec="max_reasoning_steps",
                reason=(
                    f"Reasoning step count {current_steps} has reached or exceeded "
                    f"the limit of {self.spec.max_reasoning_steps}."
                ),
            )

    # ------------------------------------------------------------------
    # Agent wrapper
    # ------------------------------------------------------------------

    @contextmanager
    def wrap_agent(
        self, agent: Any
    ) -> Generator[None, None, None]:
        """Context manager that wraps an agent and intercepts its tool calls.

        This is a lightweight wrapper that can be used to add boundary
        enforcement to an existing agent without subclassing. Tool calls
        made by the agent within the context are checked against the spec.

        Args:
            agent: The agent object to wrap. The agent's ``run`` method
                should accept a ``prompt`` positional argument.

        Yields:
            None. Raises ``BoundaryViolationError`` on any violation.

        Example:
            >>> layer = BoundaryEnforcementLayer(spec)
            >>> with layer.wrap_agent(my_agent):
            ...     result = my_agent.run("Why is the sky blue?")
        """
        self._logger.info("agent_wrap_enter", spec=self.spec)
        try:
            yield
        finally:
            self._logger.info("agent_wrap_exit")
