"""KnowledgeGuidedPrioritizationChecker — checks for high-salience domain terms in reasoning."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rrg.patterns import (
    BasePatternDetector,
    PatternMatch,
    PatternType,
    ReasoningTrace,
)

if TYPE_CHECKING:
    pass


class KnowledgeGuidedPrioritizationChecker(BasePatternDetector):
    """Checks whether high-salience domain facts appear in the reasoning.

    Uses a simple keyword-in-context approach (BM25-like without actual index).
    Takes a list of high-salience terms that SHOULD appear in reasoning.

    Detection triggers:
    - Fewer than 50% of salient terms appear in the trace
    - No domain-specific terms appear in the first half of the trace
    """

    def __init__(
        self,
        salient_terms: list[str],
        min_presence_ratio: float = 0.5,
    ) -> None:
        """Initialize the checker.

        Args:
            salient_terms: List of high-salience domain terms that should appear
                in the reasoning (e.g., ["algorithm", "complexity", "O(n)", "sorting"]).
            min_presence_ratio: Fraction of salient terms that must appear (0.0 to 1.0).
                Defaults to 0.5 (50%).
        """
        if not salient_terms:
            raise ValueError("salient_terms must be a non-empty list")
        self._salient_terms = salient_terms
        self._min_presence_ratio = min_presence_ratio

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.KNOWLEDGE_PRIORITIZATION_FAILURE

    def _term_present(self, text: str, term: str) -> bool:
        """Check if a term appears in text (case-insensitive, whole-word-ish)."""
        # Escape special regex characters in the term
        escaped = re.escape(term)
        # Use word boundary for longer terms, substring for short ones
        if len(term) <= 2:
            pattern = re.compile(escaped, re.IGNORECASE)
        else:
            pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        return pattern.search(text) is not None

    def _first_half_text(self, trace: ReasoningTrace) -> str:
        """Return the content of the first half of the reasoning steps."""
        steps = trace.steps
        half = len(steps) // 2
        return "\n".join(s.content for s in steps[:half])

    def detect(self, trace: ReasoningTrace) -> list[PatternMatch]:
        """Scan the trace for knowledge prioritization failures."""
        matches: list[PatternMatch] = []
        steps = trace.steps
        total_terms = len(self._salient_terms)

        if total_terms == 0:
            return matches

        all_text = "\n".join(s.content for s in steps)

        # Count how many salient terms appear
        present_terms: list[str] = []
        missing_terms: list[str] = []
        for term in self._salient_terms:
            if self._term_present(all_text, term):
                present_terms.append(term)
            else:
                missing_terms.append(term)

        presence_ratio = len(present_terms) / total_terms

        # --- Check 1: Overall presence ratio below threshold ---
        if presence_ratio < self._min_presence_ratio:
            evidence = (
                f"Only {len(present_terms)}/{total_terms} salient terms present "
                f"({presence_ratio:.0%}, threshold {self._min_presence_ratio:.0%}). "
                f"Missing terms: {missing_terms}."
            )
            confidence = (len(missing_terms) / total_terms) * 0.8
            matches.append(
                PatternMatch(
                    pattern_type=PatternType.KNOWLEDGE_PRIORITIZATION_FAILURE,
                    confidence=confidence,
                    evidence=evidence,
                    span=None,
                )
            )

        # --- Check 2: Early prioritization failure ---
        # If no domain-specific terms appear in the first half of the trace
        first_half_text = self._first_half_text(trace)
        early_presence_count = sum(
            1 for term in self._salient_terms
            if self._term_present(first_half_text, term)
        )

        # A term is considered "domain-specific" if it's more than 2 chars
        domain_terms = [t for t in self._salient_terms if len(t) > 2]
        if domain_terms and early_presence_count == 0 and len(steps) >= 2:
            evidence = (
                f"No salient domain terms found in first {len(steps)//2 + len(steps)%2} "
                f"step(s). The agent may have failed to prioritize key domain knowledge early."
            )
            early_bonus = 0.4
            matches.append(
                PatternMatch(
                    pattern_type=PatternType.KNOWLEDGE_PRIORITIZATION_FAILURE,
                    confidence=early_bonus,
                    evidence=evidence,
                    span=None,
                )
            )

        return matches

    def __repr__(self) -> str:
        return (
            f"KnowledgeGuidedPrioritizationChecker("
            f"salient_terms={self._salient_terms!r}, "
            f"min_presence_ratio={self._min_presence_ratio})"
        )
