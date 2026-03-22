"""Tests for rrg.enforcement.boundary and BoundaryEnforcementLayer."""

from __future__ import annotations

import pytest

from rrg.enforcement.boundary import BoundarySpec, BoundaryViolationError, StepLabel
from rrg.enforcement.enforcement_layer import BoundaryEnforcementLayer


class TestBoundarySpec:
    def test_defaults(self) -> None:
        spec = BoundarySpec()
        assert spec.allowed_tools == []
        assert spec.prohibited_tools == []
        assert spec.prohibited_domains == []
        assert spec.max_reasoning_steps is None
        assert spec.max_tool_calls_per_step == 5
        assert spec.domain_whitelist is None

    def test_custom_values(self) -> None:
        spec = BoundarySpec(
            allowed_tools=["search", "read"],
            prohibited_tools=["delete", "execute"],
            prohibited_domains=["medical"],
            max_reasoning_steps=10,
            max_tool_calls_per_step=3,
            domain_whitelist=["science"],
        )
        assert spec.allowed_tools == ["search", "read"]
        assert spec.prohibited_tools == ["delete", "execute"]
        assert spec.prohibited_domains == ["medical"]
        assert spec.max_reasoning_steps == 10
        assert spec.max_tool_calls_per_step == 3
        assert spec.domain_whitelist == ["science"]


class TestBoundaryViolationError:
    def test_message_format(self) -> None:
        err = BoundaryViolationError("prohibited_tools", "Tool 'foo' is not allowed.")
        assert err.violated_spec == "prohibited_tools"
        assert err.reason == "Tool 'foo' is not allowed."
        assert "[prohibited_tools]" in str(err)
        assert "foo" in str(err)

    def test_is_exception(self) -> None:
        err = BoundaryViolationError("x", "y")
        assert isinstance(err, Exception)


class TestBoundaryEnforcementLayer:
    def test_init_stores_spec(self) -> None:
        spec = BoundarySpec(max_reasoning_steps=5)
        layer = BoundaryEnforcementLayer(spec)
        assert layer.spec is spec

    # ------------------------------------------------------------------
    # check_tool_call
    # ------------------------------------------------------------------

    def test_check_tool_call_allowed_empty_list(self) -> None:
        # Empty allowed_tools means all tools allowed
        spec = BoundarySpec(allowed_tools=[])
        layer = BoundaryEnforcementLayer(spec)
        layer.check_tool_call("any_tool", {})  # should not raise

    def test_check_tool_call_allowed_in_list(self) -> None:
        spec = BoundarySpec(allowed_tools=["search", "read"])
        layer = BoundaryEnforcementLayer(spec)
        layer.check_tool_call("search", {})  # should not raise
        layer.check_tool_call("read", {})  # should not raise

    def test_check_tool_call_not_in_allowed_list(self) -> None:
        spec = BoundarySpec(allowed_tools=["search", "read"])
        layer = BoundaryEnforcementLayer(spec)
        with pytest.raises(BoundaryViolationError) as exc_info:
            layer.check_tool_call("dangerous_tool", {})
        assert "dangerous_tool" in str(exc_info.value)
        assert "allowed_tools" in exc_info.value.violated_spec

    def test_check_tool_call_prohibited_list(self) -> None:
        spec = BoundarySpec(allowed_tools=[], prohibited_tools=["dangerous_tool"])
        layer = BoundaryEnforcementLayer(spec)
        with pytest.raises(BoundaryViolationError) as exc_info:
            layer.check_tool_call("dangerous_tool", {})
        assert "dangerous_tool" in str(exc_info.value)
        assert "prohibited_tools" in exc_info.value.violated_spec

    def test_check_tool_call_both_allowed_and_prohibited_prohibited_wins(
        self,
    ) -> None:
        # If tool is in both allowed and prohibited, prohibited should take precedence
        spec = BoundarySpec(allowed_tools=["dangerous_tool"], prohibited_tools=["dangerous_tool"])
        layer = BoundaryEnforcementLayer(spec)
        with pytest.raises(BoundaryViolationError) as exc_info:
            layer.check_tool_call("dangerous_tool", {})
        assert "prohibited_tools" in exc_info.value.violated_spec

    # ------------------------------------------------------------------
    # check_domain
    # ------------------------------------------------------------------

    def test_check_domain_allowed(self) -> None:
        spec = BoundarySpec(prohibited_domains=["medical"])
        layer = BoundaryEnforcementLayer(spec)
        layer.check_domain("The weather is sunny today.")  # should not raise

    def test_check_domain_prohibited(self) -> None:
        spec = BoundarySpec(prohibited_domains=["medical", "legal"])
        layer = BoundaryEnforcementLayer(spec)
        with pytest.raises(BoundaryViolationError) as exc_info:
            layer.check_domain("This is a medical question about drugs.")
        assert "medical" in str(exc_info.value)
        assert "prohibited_domains" in exc_info.value.violated_spec

    def test_check_domain_whitelist_pass(self) -> None:
        spec = BoundarySpec(domain_whitelist=["science", "math"])
        layer = BoundaryEnforcementLayer(spec)
        layer.check_domain("This is a science question about physics.")  # should not raise

    def test_check_domain_whitelist_fail(self) -> None:
        spec = BoundarySpec(domain_whitelist=["science", "math"])
        layer = BoundaryEnforcementLayer(spec)
        with pytest.raises(BoundaryViolationError) as exc_info:
            layer.check_domain("Tell me about cooking recipes.")
        assert "domain_whitelist" in exc_info.value.violated_spec

    def test_check_domain_case_insensitive(self) -> None:
        spec = BoundarySpec(prohibited_domains=["Medical"])
        layer = BoundaryEnforcementLayer(spec)
        with pytest.raises(BoundaryViolationError):
            layer.check_domain("This is a MEDICAL question.")

    # ------------------------------------------------------------------
    # check_step_count
    # ------------------------------------------------------------------

    def test_check_step_count_none_limit(self) -> None:
        spec = BoundarySpec(max_reasoning_steps=None)
        layer = BoundaryEnforcementLayer(spec)
        layer.check_step_count(999999)  # should not raise

    def test_check_step_count_under_limit(self) -> None:
        spec = BoundarySpec(max_reasoning_steps=10)
        layer = BoundaryEnforcementLayer(spec)
        layer.check_step_count(5)  # should not raise

    def test_check_step_count_at_limit(self) -> None:
        spec = BoundarySpec(max_reasoning_steps=10)
        layer = BoundaryEnforcementLayer(spec)
        with pytest.raises(BoundaryViolationError) as exc_info:
            layer.check_step_count(10)
        assert "max_reasoning_steps" in exc_info.value.violated_spec

    def test_check_step_count_exceeds_limit(self) -> None:
        spec = BoundarySpec(max_reasoning_steps=5)
        layer = BoundaryEnforcementLayer(spec)
        with pytest.raises(BoundaryViolationError) as exc_info:
            layer.check_step_count(20)
        assert "max_reasoning_steps" in exc_info.value.violated_spec

    # ------------------------------------------------------------------
    # wrap_agent context manager
    # ------------------------------------------------------------------

    def test_wrap_agent_yields(self) -> None:
        spec = BoundarySpec()
        layer = BoundaryEnforcementLayer(spec)
        called = False

        class DummyAgent:
            def run(self) -> str:
                nonlocal called
                called = True
                return "ok"

        with layer.wrap_agent(DummyAgent()):
            DummyAgent().run()

        assert called
