"""PathLockInDetector — detects when agent commits to a path without exploring alternatives."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rrg.patterns import (
    BasePatternDetector,
    PatternMatch,
    PatternType,
    ReasoningTrace,
    ReasoningStep,
)

if TYPE_CHECKING:
    pass


class PathLockInDetector(BasePatternDetector):
    """Detects when agent commits to a reasoning path without exploring alternatives.

    Flags when divergence < 2 branches after 3+ reasoning steps;
    detects repeated near-identical token sequences.

    Detection triggers:
    - After 3+ steps, all steps share > 70% token vocabulary overlap
    - Any 3-gram appears in >= 2 steps (repetition flag)
    - After 3+ steps, each new step contains > 80% of previous step's tokens (monotonic lock-in)
    """

    def __init__(
        self,
        vocab_overlap_threshold: float = 0.70,
        repetition_threshold: int = 2,
        monotonic_threshold: float = 0.80,
        ngram_size: int = 3,
        min_steps_for_check: int = 3,
    ) -> None:
        """Initialize the detector.

        Args:
            vocab_overlap_threshold: Fraction of shared vocabulary to flag lock-in.
            repetition_threshold: Minimum step count for a repeated n-gram to be flagged.
            monotonic_threshold: Fraction of previous step's tokens in new step to flag monotonicity.
            ngram_size: Size of token n-grams to check for repetition.
            min_steps_for_check: Minimum steps before performing lock-in checks.
        """
        self._vocab_overlap_threshold = vocab_overlap_threshold
        self._repetition_threshold = repetition_threshold
        self._monotonic_threshold = monotonic_threshold
        self._ngram_size = ngram_size
        self._min_steps_for_check = min_steps_for_check

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.PATH_LOCK_IN

    def _tokenize(self, text: str) -> tuple[str, ...]:
        """Split text into tokens (words)."""
        return tuple(re.findall(r"\b\w+\b", text.lower()))

    def _get_ngrams(self, tokens: tuple[str, ...], n: int) -> frozenset[tuple[str, ...]]:
        """Extract n-grams from a token sequence."""
        if len(tokens) < n:
            return frozenset()
        return frozenset(tokens[i : i + n] for i in range(len(tokens) - n + 1))

    def _vocab_overlap(self, tokens1: tuple[str, ...], tokens2: tuple[str, ...]) -> float:
        """Compute vocabulary overlap between two token sequences."""
        vocab1 = set(tokens1)
        vocab2 = set(tokens2)
        if not vocab1 or not vocab2:
            return 0.0
        intersection = vocab1 & vocab2
        union = vocab1 | vocab2
        return len(intersection) / len(union)

    def _monotonicity_score(self, steps: tuple[ReasoningStep, ...]) -> float:
        """Check if each step mostly contains the previous step's tokens.

        Returns a score 0.0-1.0 where 1.0 means fully monotonic (each step
        is a subset of the previous).
        """
        if len(steps) < 2:
            return 0.0

        monotonic_count = 0
        total_pairs = 0
        for i in range(1, len(steps)):
            prev_tokens = self._tokenize(steps[i - 1].content)
            curr_tokens = self._tokenize(steps[i].content)
            if not prev_tokens:
                continue
            # What fraction of current tokens appear in previous
            prev_set = set(prev_tokens)
            overlap = sum(1 for t in curr_tokens if t in prev_set)
            frac = overlap / len(curr_tokens) if curr_tokens else 0.0
            if frac > self._monotonic_threshold:
                monotonic_count += 1
            total_pairs += 1

        return monotonic_count / total_pairs if total_pairs > 0 else 0.0

    def detect(self, trace: ReasoningTrace) -> list[PatternMatch]:
        """Scan the trace for path lock-in indicators."""
        matches: list[PatternMatch] = []
        steps = trace.steps

        if len(steps) < self._min_steps_for_check:
            return matches

        # Collect vocabularies and n-grams across all steps
        all_tokens = [self._tokenize(s.content) for s in steps]
        all_vocabs = [set(toks) for toks in all_tokens]

        # Check 1: Global vocabulary overlap
        # Pairwise overlap — if all pairs are above threshold, flag it
        pairwise_overlaps = []
        for i in range(len(all_vocabs)):
            for j in range(i + 1, len(all_vocabs)):
                overlap = self._vocab_overlap(
                    tuple(all_vocabs[i]), tuple(all_vocabs[j])
                )
                pairwise_overlaps.append(overlap)

        if pairwise_overlaps:
            avg_overlap = sum(pairwise_overlaps) / len(pairwise_overlaps)
            if avg_overlap > self._vocab_overlap_threshold:
                evidence = (
                    f"Vocabulary overlap across {len(steps)} steps is "
                    f"{avg_overlap:.2%} (threshold {self._vocab_overlap_threshold:.0%}). "
                    f"Steps may be repeating the same concepts without exploration."
                )
                confidence = avg_overlap * 0.7
                matches.append(
                    PatternMatch(
                        pattern_type=PatternType.PATH_LOCK_IN,
                        confidence=confidence,
                        evidence=evidence,
                        span=None,
                    )
                )

        # Check 2: Repeated n-gram sequences across steps
        ngram_counter: dict[tuple[str, ...], int] = {}
        for step_tokens in all_tokens:
            ngrams = self._get_ngrams(step_tokens, self._ngram_size)
            for ng in ngrams:
                ngram_counter[ng] = ngram_counter.get(ng, 0) + 1

        repeated_ngrams = {
            ng for ng, count in ngram_counter.items() if count >= self._repetition_threshold
        }
        if repeated_ngrams:
            # Count how many n-grams are repeated (to scale confidence)
            num_repeated = len(repeated_ngrams)
            evidence = (
                f"Found {num_repeated} repeated {self._ngram_size}-gram(s) "
                f"appearing in >= {self._repetition_threshold} steps. "
                f"Examples: {list(repeated_ngrams)[:3]}"
            )
            confidence = min(1.0, (num_repeated * 0.1) + 0.3)
            matches.append(
                PatternMatch(
                    pattern_type=PatternType.PATH_LOCK_IN,
                    confidence=confidence,
                    evidence=evidence,
                    span=None,
                )
            )

        # Check 3: Monotonic content — each step contains most of previous step's tokens
        monotonic_score = self._monotonicity_score(steps)
        if monotonic_score > 0.5:  # More than half the step pairs are monotonic
            evidence = (
                f"Monotonic content score: {monotonic_score:.2f} "
                f"(threshold {self._monotonic_threshold:.0%}). "
                f"Each new step largely repeats previous step's tokens "
                f"without meaningful new exploration."
            )
            matches.append(
                PatternMatch(
                    pattern_type=PatternType.PATH_LOCK_IN,
                    confidence=monotonic_score * 0.7,
                    evidence=evidence,
                    span=None,
                )
            )

        return matches

    def __repr__(self) -> str:
        return (
            f"PathLockInDetector("
            f"vocab_overlap_threshold={self._vocab_overlap_threshold}, "
            f"repetition_threshold={self._repetition_threshold}, "
            f"monotonic_threshold={self._monotonic_threshold})"
        )
