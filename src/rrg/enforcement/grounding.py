"""MemoryGroundingLayer: episodic buffer and hallucination detection via keyword overlap."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class EpisodeChunk:
    """A chunk of episodic memory with its content and metadata."""

    episode_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryGroundingLayer:
    """Episodic grounding buffer with simple keyword-overlap hallucination detection.

    This layer maintains an episodic buffer of context pieces and can
    detect claims in reasoning that are not supported by any stored context.
    It uses simple term-frequency (TF) keyword overlap rather than embeddings.

    Attributes:
        max_episodes: Maximum number of episodes to retain. When exceeded,
            the oldest episodes are evicted.

    Example:
        >>> grounding = MemoryGroundingLayer(max_episodes=50)
        >>> grounding.add_context("ep1", "Paris is the capital of France.")
        >>> grounding.add_context("ep1", "France is in Western Europe.")
        >>> flags = grounding.check_grounding("The capital of France is Berlin.")
        >>> print(flags)  # ['Berlin']
    """

    def __init__(self, max_episodes: int = 100) -> None:
        self.max_episodes = max_episodes
        self._chunks: list[EpisodeChunk] = []
        self._episode_index: dict[str, list[int]] = defaultdict(list)
        self._logger = logger.bind(component="MemoryGroundingLayer")

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def add_context(
        self,
        episode_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a chunk of context to the episodic buffer.

        Args:
            episode_id: Identifier for the episode this chunk belongs to.
            content: The text content to store.
            metadata: Optional metadata (source, timestamp, etc.).
        """
        chunk = EpisodeChunk(
            episode_id=episode_id,
            content=content,
            metadata=metadata or {},
        )
        self._chunks.append(chunk)
        idx = len(self._chunks) - 1
        self._episode_index[episode_id].append(idx)

        # Evict oldest episodes if over limit
        if len(self._episode_index) > self.max_episodes:
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        """Remove the oldest episode when max_episodes is exceeded."""
        if not self._episode_index:
            return
        oldest_episode_id = next(iter(self._episode_index))
        indices = self._episode_index.pop(oldest_episode_id)
        self._chunks = [c for i, c in enumerate(self._chunks) if i not in indices]
        # Rebuild index with updated positions
        self._reindex()

    def _reindex(self) -> None:
        """Rebuild the episode index after eviction."""
        self._episode_index.clear()
        for idx, chunk in enumerate(self._chunks):
            self._episode_index[chunk.episode_id].append(idx)

    def get_full_context(self, episode_id: str | None = None) -> list[str]:
        """Retrieve full context strings.

        Args:
            episode_id: If provided, only return chunks from this episode.
                If None, return all chunks from all episodes.

        Returns:
            List of content strings in insertion order.
        """
        if episode_id is not None:
            indices = self._episode_index.get(episode_id, [])
            return [self._chunks[i].content for i in indices]
        return [chunk.content for chunk in self._chunks]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple whitespace/punctuation tokenisation returning lowercase tokens."""
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return set(tokens)

    def _tf_score(self, query_tokens: set[str], text: str) -> float:
        """Score text by term-frequency keyword overlap with query tokens."""
        text_tokens = self._tokenize(text)
        if not query_tokens:
            return 0.0
        overlap = query_tokens & text_tokens
        return len(overlap) / len(query_tokens)

    def get_relevant_context(
        self, query: str, k: int = 5
    ) -> list[tuple[str, float]]:
        """Retrieve the top-k context chunks most relevant to the query.

        Uses simple keyword overlap scoring. Returns chunks sorted by
        descending score.

        Args:
            query: The query text.
            k: Maximum number of chunks to return.

        Returns:
            List of (content, score) tuples, most relevant first.
        """
        query_tokens = self._tokenize(query)
        scored: list[tuple[str, float]] = []
        for chunk in self._chunks:
            score = self._tf_score(query_tokens, chunk.content)
            if score > 0:
                scored.append((chunk.content, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ------------------------------------------------------------------
    # Hallucination detection
    # ------------------------------------------------------------------

    def _extract_facts(self, text: str) -> list[str]:
        """Extract potential factual claims from text.

        Uses a simple heuristic: sentences containing copula verbs,
        numbers, or key domain nouns are treated as candidate facts.
        This is a rule-based approach — not an LLM.
        """
        # Split into sentences
        sentences = re.split(r"[.!?\n]+", text)
        facts: list[str] = []
        for s in sentences:
            s = s.strip()
            if len(s) < 5:
                continue
            # Heuristic: if sentence contains a verb and some content
            if any(tok in s.lower() for tok in ["is", "are", "was", "were", "has", "have", "contains", "equals"]):
                facts.append(s)
        return facts

    def check_grounding(self, reasoning_text: str) -> list[str]:
        """Detect potentially ungrounded facts in reasoning text.

        Compares facts extracted from ``reasoning_text`` against stored
        episodic context. Any named entity or noun phrase in a reasoning
        fact that cannot be matched to context context is flagged.

        Args:
            reasoning_text: The reasoning output to check.

        Returns:
            List of ungrounded terms (words or short phrases) found in
            the reasoning that have no support in the episodic buffer.
        """
        context_texts = self.get_full_context()
        context_tokens: set[str] = set()
        for ctx in context_texts:
            context_tokens |= self._tokenize(ctx)

        ungrounded_flags: list[str] = []

        # Extract candidate named entities / factual terms using simple Noun-phrase patterns
        # Match capitalized multi-word phrases and numbers
        capitalized_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", reasoning_text)
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", reasoning_text)

        for phrase in capitalized_phrases:
            phrase_tokens = self._tokenize(phrase)
            # If none of the tokens in the phrase appear in context, flag it
            if phrase_tokens and not (phrase_tokens & context_tokens):
                ungrounded_flags.append(phrase)

        for num in numbers:
            # Flag standalone numbers that don't appear in any context
            if num not in context_tokens:
                ungrounded_flags.append(num)

        return ungrounded_flags
