"""Tests for rrg.core module — ReasoningAgent and related classes."""

from __future__ import annotations

import pytest

from rrg.core import (
    LLMBackend,
    ReasoningAgent,
    ReasoningAgentConfig,
    ReasoningResult,
    ReasoningStep,
    ReasoningTrace,
    asdict_safe,
)
from tests.conftest import MockLLMBackend, MockReasoningAgent


class TestReasoningStep:
    def test_repr_truncates_long_content(self) -> None:
        long_content = "x" * 100
        step = ReasoningStep(step_id=1, content=long_content)
        assert "..." in repr(step)
        assert len(repr(step)) < 120

    def test_repr_short_content(self) -> None:
        step = ReasoningStep(step_id=1, content="short")
        assert "short" in repr(step)

    def test_immutability(self) -> None:
        step = ReasoningStep(step_id=1, content="test")
        with pytest.raises(Exception):  # frozen dataclass
            step.step_id = 2  # type: ignore


class TestReasoningTrace:
    def test_repr_shows_step_count(self, valid_trace: ReasoningTrace) -> None:
        assert "steps=5" in repr(valid_trace)

    def test_steps_tuple_immutable(self, valid_trace: ReasoningTrace) -> None:
        with pytest.raises(Exception):  # frozen dataclass
            valid_trace.steps = ()  # type: ignore


class TestReasoningAgentConfig:
    def test_defaults(self) -> None:
        cfg = ReasoningAgentConfig()
        assert cfg.max_steps == 10
        assert cfg.temperature == 0.7
        assert cfg.enable_cot is True

    def test_custom_config(self) -> None:
        cfg = ReasoningAgentConfig(max_steps=5, temperature=0.5)
        assert cfg.max_steps == 5
        assert cfg.temperature == 0.5


class TestMockLLMBackend:
    def test_generate_returns_response(self) -> None:
        backend = MockLLMBackend(responses=["hello"])
        result = backend.generate("test prompt")
        assert result == "hello"

    def test_generate_cycles_responses(self) -> None:
        backend = MockLLMBackend(responses=["a", "b", "c"])
        assert backend.generate("p1") == "a"
        assert backend.generate("p2") == "b"
        assert backend.generate("p3") == "c"
        assert backend.generate("p4") == "c"  # wraps to last

    def test_generate_with_reasoning_returns_steps(self) -> None:
        backend = MockLLMBackend(responses=["answer"])
        response, steps = backend.generate_with_reasoning("test prompt")
        assert response == "answer"
        assert len(steps) == 2
        assert all(isinstance(s, ReasoningStep) for s in steps)


class TestMockReasoningAgent:
    def test_run_returns_result(self, mock_agent: MockReasoningAgent) -> None:
        result = mock_agent.run("What is 2+2?")
        assert isinstance(result, ReasoningResult)
        assert isinstance(result.trace, ReasoningTrace)
        assert result.accepted is True
        assert isinstance(result.response, str)

    def test_run_increments_call_count(self, mock_agent: MockReasoningAgent) -> None:
        mock_agent.run("q1")
        assert mock_agent.backend.call_count == 1
        mock_agent.run("q2")
        assert mock_agent.backend.call_count == 2

    def test_trace_built_correctly(self, mock_agent: MockReasoningAgent) -> None:
        result = mock_agent.run("test")
        assert len(result.trace.steps) == 2
        assert result.trace.final_answer == result.response


class TestReasoningAgentConfigAsDict:
    def test_asdict_safe_with_dataclass(self) -> None:
        cfg = ReasoningAgentConfig(max_steps=7)
        d = asdict_safe(cfg)
        assert d["max_steps"] == 7
        assert "temperature" in d

    def test_asdict_safe_with_non_dataclass(self) -> None:
        d = asdict_safe("hello")
        assert "repr" in d


class TestLLMBackend:
    def test_is_abstract(self) -> None:
        # LLMBackend cannot be instantiated directly
        with pytest.raises(TypeError):
            LLMBackend()  # type: ignore


class TestReasoningAgent:
    def test_is_abstract(self) -> None:
        # ReasoningAgent cannot be instantiated directly
        with pytest.raises(TypeError):
            ReasoningAgent()  # type: ignore

    def test_repr(self, mock_agent: MockReasoningAgent) -> None:
        r = repr(mock_agent)
        assert "MockReasoningAgent" in r
        assert "max_steps" in r
