"""Tests for rrg.enforcement.grounding — MemoryGroundingLayer."""

from __future__ import annotations

import pytest

from rrg.enforcement.grounding import MemoryGroundingLayer


class TestMemoryGroundingLayerInit:
    def test_default_max_episodes(self) -> None:
        layer = MemoryGroundingLayer()
        assert layer.max_episodes == 100

    def test_custom_max_episodes(self) -> None:
        layer = MemoryGroundingLayer(max_episodes=10)
        assert layer.max_episodes == 10


class TestAddContext:
    def test_add_single_context(self) -> None:
        layer = MemoryGroundingLayer()
        layer.add_context("ep1", "Paris is the capital of France.")
        assert len(layer.get_full_context()) == 1
        assert "Paris" in layer.get_full_context()[0]

    def test_add_multiple_contexts_same_episode(self) -> None:
        layer = MemoryGroundingLayer()
        layer.add_context("ep1", "Fact 1.")
        layer.add_context("ep1", "Fact 2.")
        ctx = layer.get_full_context("ep1")
        assert len(ctx) == 2
        assert "Fact 1" in ctx[0]

    def test_add_context_with_metadata(self) -> None:
        layer = MemoryGroundingLayer()
        layer.add_context("ep1", "Something.", metadata={"source": "test"})
        # Metadata is stored internally but get_full_context only returns strings
        chunk = layer._chunks[0]
        assert chunk.metadata["source"] == "test"


class TestGetFullContext:
    def test_get_all_episodes(self) -> None:
        layer = MemoryGroundingLayer()
        layer.add_context("ep1", "Content 1.")
        layer.add_context("ep2", "Content 2.")
        ctx = layer.get_full_context()
        assert len(ctx) == 2

    def test_get_specific_episode(self) -> None:
        layer = MemoryGroundingLayer()
        layer.add_context("ep1", "Only for ep1.")
        layer.add_context("ep2", "Only for ep2.")
        ctx = layer.get_full_context("ep1")
        assert len(ctx) == 1
        assert "ep1" in ctx[0] or "Only for ep1" in ctx[0]

    def test_get_nonexistent_episode(self) -> None:
        layer = MemoryGroundingLayer()
        layer.add_context("ep1", "Something.")
        ctx = layer.get_full_context("nonexistent")
        assert ctx == []


class TestGetRelevantContext:
    def test_retrieve_top_k(self) -> None:
        layer = MemoryGroundingLayer()
        layer.add_context("ep1", "Python is a programming language.")
        layer.add_context("ep1", "Java is also a programming language.")
        layer.add_context("ep1", "The sky is blue.")
        layer.add_context("ep1", "Water is wet.")
        layer.add_context("ep1", "Mars is red.")

        results = layer.get_relevant_context("Python programming", k=3)
        assert len(results) <= 3
        # "Python is a programming language" should score highest
        assert "Python" in results[0][0]

    def test_no_match_returns_empty(self) -> None:
        layer = MemoryGroundingLayer()
        layer.add_context("ep1", "Something unrelated.")
        results = layer.get_relevant_context("zzznomatchzzz", k=5)
        assert results == []

    def test_k_respected(self) -> None:
        layer = MemoryGroundingLayer()
        for i in range(10):
            layer.add_context("ep1", f"Document number {i}.")
        results = layer.get_relevant_context("Document", k=3)
        assert len(results) == 3


class TestCheckGrounding:
    def test_no_flags_when_facts_in_context(self) -> None:
        layer = MemoryGroundingLayer()
        layer.add_context("ep1", "Paris is the capital of France.")
        flags = layer.check_grounding("Paris is the capital of France.")
        # The text itself is in context so no ungrounded flags
        assert "Paris" not in flags

    def test_flags_hallucinated_capital(self) -> None:
        layer = MemoryGroundingLayer()
        layer.add_context("ep1", "Paris is the capital of France.")
        # "Berlin" is a capitalized phrase not supported by context
        flags = layer.check_grounding("The capital of France is Berlin.")
        assert "Berlin" in flags

    def test_flags_hallucinated_number(self) -> None:
        layer = MemoryGroundingLayer()
        layer.add_context("ep1", "The population is 1000.")
        flags = layer.check_grounding("The population is 999999.")
        assert "999999" in flags

    def test_empty_context_flags_all_capitalized(self) -> None:
        layer = MemoryGroundingLayer()
        flags = layer.check_grounding("Einstein was a physicist.")
        # "Einstein" is capitalized and not in any context
        assert "Einstein" in flags

    def test_empty_reasoning_returns_empty_flags(self) -> None:
        layer = MemoryGroundingLayer()
        layer.add_context("ep1", "Some fact.")
        flags = layer.check_grounding("")
        assert flags == []

    def test_context_with_multiple_facts_partial_hallucination(self) -> None:
        layer = MemoryGroundingLayer()
        layer.add_context("ep1", "The capital of France is Paris.")
        layer.add_context("ep1", "The capital of Germany is Berlin.")
        flags = layer.check_grounding(
            "The capital of France is Paris. The capital of Germany is Bonn."
        )
        # "Bonn" is not supported (Berlin is the actual capital)
        assert "Bonn" in flags


class TestEpisodicEviction:
    def test_eviction_after_max_episodes(self) -> None:
        layer = MemoryGroundingLayer(max_episodes=3)
        layer.add_context("ep1", "Episode 1.")
        layer.add_context("ep2", "Episode 2.")
        layer.add_context("ep3", "Episode 3.")
        layer.add_context("ep4", "Episode 4.")  # should evict ep1

        ctx = layer.get_full_context()
        episodes_present = set()
        for chunk in layer._chunks:
            episodes_present.add(chunk.episode_id)

        assert "ep1" not in episodes_present
        assert "ep4" in episodes_present
