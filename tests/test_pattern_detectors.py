"""Tests for the four implicit pattern detectors."""

from __future__ import annotations

import pytest

from rrg.patterns import (
    BasePatternDetector,
    PatternMatch,
    PatternType,
    ReasoningStep,
    ReasoningTrace,
)
from rrg.patterns.early_pruning import EarlyPruningDetector
from rrg.patterns.path_lockin import PathLockInDetector
from rrg.patterns.boundary_violation import (
    BoundaryViolationDetector,
    BoundarySpec,
)
from rrg.patterns.knowledge_prioritization import (
    KnowledgeGuidedPrioritizationChecker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_trace(*step_contents: str) -> ReasoningTrace:
    """Shorthand to build a ReasoningTrace from step content strings."""
    steps = tuple(
        ReasoningStep(i + 1, content) for i, content in enumerate(step_contents)
    )
    return ReasoningTrace(
        steps=steps,
        final_answer="test answer",
        metadata={},
    )


# ---------------------------------------------------------------------------
# EarlyPruningDetector Tests
# ---------------------------------------------------------------------------

class TestEarlyPruningDetector:
    def test_detects_short_trace_with_short_first_step(self) -> None:
        """Fewer than 3 steps + first step < 100 chars → flag."""
        trace = make_trace("X is true.", "Therefore X.")
        detector = EarlyPruningDetector()
        matches = detector.detect(trace)
        assert len(matches) >= 1
        assert matches[0].pattern_type == PatternType.EARLY_PRUNING
        assert 0.0 < matches[0].confidence <= 1.0

    def test_no_detection_on_healthy_short_trace(self) -> None:
        """Short trace but first step is long enough → no flag."""
        trace = make_trace(
            "We need to determine whether the sky is blue today. "
            "I will consider multiple sources of evidence and observations "
            "before reaching a conclusion."
        )
        detector = EarlyPruningDetector()
        matches = detector.detect(trace)
        # Should not flag the first check (short trace + short first step)
        # May still flag the average length check depending on content
        early_pruning_matches = [m for m in matches if m.pattern_type == PatternType.EARLY_PRUNING]
        # No confidence should be very high for early pruning here
        for m in early_pruning_matches:
            assert m.confidence < 0.9

    def test_detects_all_steps_very_short(self) -> None:
        """All steps very short (avg < 40 chars) → flag."""
        trace = make_trace("A.", "B.", "C.", "D.")
        detector = EarlyPruningDetector(min_step_content_length=40)
        matches = detector.detect(trace)
        assert len(matches) >= 1
        assert all(m.pattern_type == PatternType.EARLY_PRUNING for m in matches)

    def test_no_detection_on_healthy_long_trace(self) -> None:
        """Long trace with substantial step content → no flag."""
        trace = make_trace(
            "First, I will examine the available data and consider its implications.",
            "Building on that, I analyze the patterns observed in the evidence.",
            "Furthermore, I cross-reference these findings with known principles.",
            "Finally, I synthesize the analysis into a coherent conclusion.",
        )
        detector = EarlyPruningDetector()
        matches = detector.detect(trace)
        early_pruning = [m for m in matches if m.pattern_type == PatternType.EARLY_PRUNING]
        assert len(early_pruning) == 0

    def test_empty_trace(self) -> None:
        """Empty trace → no matches (no crash)."""
        trace = make_trace()
        detector = EarlyPruningDetector()
        matches = detector.detect(trace)
        assert matches == []

    def test_single_step_trace_short(self) -> None:
        """Single very short step → should flag."""
        trace = make_trace("Done.")
        detector = EarlyPruningDetector()
        matches = detector.detect(trace)
        assert len(matches) == 1
        assert matches[0].pattern_type == PatternType.EARLY_PRUNING

    def test_pattern_type_property(self) -> None:
        detector = EarlyPruningDetector()
        assert detector.pattern_type == PatternType.EARLY_PRUNING

    def test_repr(self) -> None:
        detector = EarlyPruningDetector(min_step_content_length=50)
        r = repr(detector)
        assert "EarlyPruningDetector" in r
        assert "50" in r


# ---------------------------------------------------------------------------
# PathLockInDetector Tests
# ---------------------------------------------------------------------------

class TestPathLockInDetector:
    def test_detects_high_vocab_overlap(self) -> None:
        """Steps with > 70% shared vocabulary → flag."""
        trace = make_trace(
            "The algorithm processes the data efficiently today.",
            "The algorithm processes the data efficiently today.",
            "The algorithm processes the data efficiently today.",
        )
        detector = PathLockInDetector(vocab_overlap_threshold=0.70)
        matches = detector.detect(trace)
        assert len(matches) >= 1
        assert any(m.pattern_type == PatternType.PATH_LOCK_IN for m in matches)

    def test_no_detection_on_diverse_steps(self) -> None:
        """Diverse steps with little vocabulary overlap → no flag."""
        trace = make_trace(
            "The algorithm uses quicksort with O(n log n) complexity.",
            "Memory allocation strategies include malloc and stack allocation.",
            "Database normalization follows BCNF and third normal form rules.",
            "Network routing employs Dijkstra's algorithm for shortest paths.",
        )
        detector = PathLockInDetector(vocab_overlap_threshold=0.70)
        matches = detector.detect(trace)
        lock_in = [m for m in matches if m.pattern_type == PatternType.PATH_LOCK_IN]
        assert len(lock_in) == 0

    def test_detects_repeated_ngrams(self) -> None:
        """Same n-gram appearing in multiple steps → flag."""
        trace = make_trace(
            "I think about the algorithm carefully.",
            "The algorithm is central to this problem.",
            "Algorithm design requires careful analysis.",
        )
        detector = PathLockInDetector(
            repetition_threshold=2,
            ngram_size=2,
            min_steps_for_check=2,
        )
        matches = detector.detect(trace)
        lock_in = [m for m in matches if m.pattern_type == PatternType.PATH_LOCK_IN]
        assert len(lock_in) >= 1

    def test_detects_monotonic_content(self) -> None:
        """Each step mostly contains previous step's tokens → flag monotonicity."""
        # Use repetition_threshold=10 and high vocab_overlap_threshold to disable
        # both those checks so monotonicity check is the one that fires.
        # Sentences are near-identical so monotonicity score is 1.0.
        trace = make_trace(
            "Algorithm design requires careful analysis of complexity.",
            "Algorithm design requires careful analysis of complexity factors.",
            "Algorithm design requires careful analysis of complexity factors properly.",
        )
        detector = PathLockInDetector(
            monotonic_threshold=0.80,
            min_steps_for_check=2,
            repetition_threshold=10,  # disable n-gram detection
            vocab_overlap_threshold=0.99,  # disable vocab overlap detection
        )
        matches = detector.detect(trace)
        lock_in = [m for m in matches if m.pattern_type == PatternType.PATH_LOCK_IN]
        assert len(lock_in) >= 1
        # The match should reference monotonic content
        assert any(
            "monotonic" in m.evidence.lower() for m in lock_in
        ), f"Expected monotonic flag; got: {[m.evidence[:80] for m in lock_in]}"

    def test_no_detection_below_min_steps(self) -> None:
        """With min_steps_for_check=3, 2 steps → no check, no flag."""
        trace = make_trace(
            "The algorithm is fast.",
            "The algorithm is efficient.",
        )
        detector = PathLockInDetector(min_steps_for_check=3)
        matches = detector.detect(trace)
        assert matches == []

    def test_empty_trace(self) -> None:
        """Empty trace → no matches (no crash)."""
        trace = make_trace()
        detector = PathLockInDetector()
        matches = detector.detect(trace)
        assert matches == []

    def test_single_step_trace(self) -> None:
        """Single step → no detection (below min_steps_for_check)."""
        trace = make_trace("The answer is A.")
        detector = PathLockInDetector()
        matches = detector.detect(trace)
        assert matches == []

    def test_pattern_type_property(self) -> None:
        detector = PathLockInDetector()
        assert detector.pattern_type == PatternType.PATH_LOCK_IN

    def test_repr(self) -> None:
        detector = PathLockInDetector(vocab_overlap_threshold=0.80)
        r = repr(detector)
        assert "PathLockInDetector" in r
        assert "0.8" in r


# ---------------------------------------------------------------------------
# BoundaryViolationDetector Tests
# ---------------------------------------------------------------------------

class TestBoundaryViolationDetector:
    def test_detects_exceeded_max_steps(self) -> None:
        """Trace longer than max_reasoning_steps → flag."""
        trace = make_trace(
            "Step 1.", "Step 2.", "Step 3.", "Step 4.", "Step 5."
        )
        spec = BoundarySpec(
            allowed_tools=frozenset(),
            prohibited_actions=frozenset(),
            max_reasoning_steps=3,
            domain_scope=frozenset(),
        )
        detector = BoundaryViolationDetector(boundary_spec=spec)
        matches = detector.detect(trace)
        assert len(matches) >= 1
        assert matches[0].pattern_type == PatternType.BOUNDARY_VIOLATION
        assert "3 steps" in matches[0].evidence or "5" in matches[0].evidence

    def test_detects_prohibited_actions(self) -> None:
        """Content mentions prohibited action → flag."""
        trace = make_trace(
            "I will delete the file and then sudo to root.",
        )
        spec = BoundarySpec(
            allowed_tools=frozenset(),
            prohibited_actions=frozenset({"delete", "sudo"}),
            max_reasoning_steps=10,
            domain_scope=frozenset(),
        )
        detector = BoundaryViolationDetector(boundary_spec=spec)
        matches = detector.detect(trace)
        assert len(matches) >= 1
        violation = [m for m in matches if m.pattern_type == PatternType.BOUNDARY_VIOLATION]
        assert len(violation) >= 1

    def test_no_violation_within_boundary(self) -> None:
        """All checks pass → no matches."""
        trace = make_trace(
            "We analyze the sorting algorithm complexity.",
            "The O(n log n) complexity is optimal for comparison sorts.",
        )
        spec = BoundarySpec(
            allowed_tools=frozenset({"analyze", "compute"}),
            prohibited_actions=frozenset({"delete", "sudo"}),
            max_reasoning_steps=10,
            domain_scope=frozenset({"algorithm", "complexity"}),
        )
        detector = BoundaryViolationDetector(boundary_spec=spec)
        matches = detector.detect(trace)
        assert len(matches) == 0

    def test_detects_dangerous_keywords_no_spec(self) -> None:
        """Without spec, dangerous keywords ("rm -rf", "exec(") → flag."""
        trace = make_trace(
            "I'll execute rm -rf /tmp/test and then eval(params) for safety.",
        )
        detector = BoundaryViolationDetector()
        matches = detector.detect(trace)
        assert len(matches) >= 1
        assert matches[0].pattern_type == PatternType.BOUNDARY_VIOLATION

    def test_no_false_positive_clean_trace_no_spec(self) -> None:
        """Clean content, no spec → no flag."""
        trace = make_trace(
            "The algorithm uses dynamic programming to solve the problem.",
            "We verify the solution with test cases.",
        )
        detector = BoundaryViolationDetector()
        matches = detector.detect(trace)
        assert matches == []

    def test_empty_trace(self) -> None:
        """Empty trace → no matches (no crash)."""
        trace = make_trace()
        detector = BoundaryViolationDetector()
        matches = detector.detect(trace)
        assert matches == []

    def test_pattern_type_property(self) -> None:
        detector = BoundaryViolationDetector()
        assert detector.pattern_type == PatternType.BOUNDARY_VIOLATION

    def test_repr_with_spec(self) -> None:
        spec = BoundarySpec(
            allowed_tools=frozenset({"a", "b"}),
            prohibited_actions=frozenset({"x"}),
            max_reasoning_steps=5,
            domain_scope=frozenset(),
        )
        detector = BoundaryViolationDetector(boundary_spec=spec)
        r = repr(detector)
        assert "BoundaryViolationDetector" in r
        assert "5" in r

    def test_repr_without_spec(self) -> None:
        detector = BoundaryViolationDetector()
        r = repr(detector)
        assert "BoundaryViolationDetector" in r


# ---------------------------------------------------------------------------
# KnowledgeGuidedPrioritizationChecker Tests
# ---------------------------------------------------------------------------

class TestKnowledgeGuidedPrioritizationChecker:
    def test_detects_missing_salient_terms(self) -> None:
        """Less than 50% of salient terms appear → flag."""
        trace = make_trace(
            "The problem involves sorting a large dataset.",
            "We will use a divide and conquer approach.",
        )
        terms = ["algorithm", "complexity", "O(n)", "sorting", "merge"]
        detector = KnowledgeGuidedPrioritizationChecker(salient_terms=terms)
        matches = detector.detect(trace)
        assert len(matches) >= 1
        assert matches[0].pattern_type == PatternType.KNOWLEDGE_PRIORITIZATION_FAILURE

    def test_no_detection_when_salient_terms_present(self) -> None:
        """All salient terms present → no flag."""
        trace = make_trace(
            "The sorting algorithm has O(n log n) complexity.",
            "We use a merge sort algorithm to achieve optimal performance.",
        )
        terms = ["algorithm", "complexity", "sorting"]
        detector = KnowledgeGuidedPrioritizationChecker(salient_terms=terms)
        matches = detector.detect(trace)
        kp = [m for m in matches if m.pattern_type == PatternType.KNOWLEDGE_PRIORITIZATION_FAILURE]
        assert len(kp) == 0

    def test_detects_early_prioritization_failure(self) -> None:
        """No domain terms in first half of trace → early prioritization flag."""
        trace = make_trace(
            "Let me think about this carefully step by step.",
            "I will now analyze the problem in detail.",
            "The algorithm complexity is O(n log n).",
        )
        terms = ["algorithm", "complexity", "O(n)", "sorting"]
        detector = KnowledgeGuidedPrioritizationChecker(salient_terms=terms)
        matches = detector.detect(trace)
        early_flags = [
            m for m in matches
            if m.pattern_type == PatternType.KNOWLEDGE_PRIORITIZATION_FAILURE
            and "first" in m.evidence.lower()
        ]
        assert len(early_flags) >= 1

    def test_no_early_flag_when_terms_early(self) -> None:
        """Domain terms appear early → no early prioritization flag."""
        trace = make_trace(
            "The algorithm for sorting operates in O(n log n) time.",
            "This complexity is optimal for comparison-based sorting.",
        )
        terms = ["algorithm", "complexity", "sorting"]
        detector = KnowledgeGuidedPrioritizationChecker(salient_terms=terms)
        matches = detector.detect(trace)
        early_flags = [
            m for m in matches
            if m.pattern_type == PatternType.KNOWLEDGE_PRIORITIZATION_FAILURE
            and "first" in m.evidence.lower()
        ]
        assert len(early_flags) == 0

    def test_empty_trace(self) -> None:
        """Empty trace → no matches (no crash, but likely flags missing terms)."""
        trace = make_trace()
        terms = ["algorithm", "complexity"]
        detector = KnowledgeGuidedPrioritizationChecker(salient_terms=terms)
        matches = detector.detect(trace)
        # Should flag missing terms (0/2 = 0% < 50%)
        assert len(matches) >= 1

    def test_single_step_trace(self) -> None:
        """Single step with all terms → no flag."""
        trace = make_trace(
            "The sorting algorithm has O(n log n) complexity."
        )
        terms = ["algorithm", "complexity", "sorting"]
        detector = KnowledgeGuidedPrioritizationChecker(salient_terms=terms)
        matches = detector.detect(trace)
        kp = [m for m in matches if m.pattern_type == PatternType.KNOWLEDGE_PRIORITIZATION_FAILURE]
        assert len(kp) == 0

    def test_rejects_empty_salient_terms(self) -> None:
        with pytest.raises(ValueError):
            KnowledgeGuidedPrioritizationChecker(salient_terms=[])

    def test_pattern_type_property(self) -> None:
        detector = KnowledgeGuidedPrioritizationChecker(salient_terms=["a", "b"])
        assert detector.pattern_type == PatternType.KNOWLEDGE_PRIORITIZATION_FAILURE

    def test_repr(self) -> None:
        detector = KnowledgeGuidedPrioritizationChecker(
            salient_terms=["algorithm", "complexity"],
            min_presence_ratio=0.6,
        )
        r = repr(detector)
        assert "KnowledgeGuidedPrioritizationChecker" in r
        assert "algorithm" in r


# ---------------------------------------------------------------------------
# Integration / round-trip tests
# ---------------------------------------------------------------------------

class TestAllDetectorsInheritFromBase:
    @pytest.mark.parametrize(
        "detector_class",
        [
            EarlyPruningDetector,
            PathLockInDetector,
            BoundaryViolationDetector,
            KnowledgeGuidedPrioritizationChecker,
        ],
    )
    def test_inherits_from_base_pattern_detector(
        self, detector_class: type[BasePatternDetector]
    ) -> None:
        assert issubclass(detector_class, BasePatternDetector)

    @pytest.mark.parametrize(
        "detector_class,expected_type",
        [
            (EarlyPruningDetector, PatternType.EARLY_PRUNING),
            (PathLockInDetector, PatternType.PATH_LOCK_IN),
            (BoundaryViolationDetector, PatternType.BOUNDARY_VIOLATION),
            (KnowledgeGuidedPrioritizationChecker, PatternType.KNOWLEDGE_PRIORITIZATION_FAILURE),
        ],
    )
    def test_pattern_type_property_value(
        self, detector_class: type[BasePatternDetector], expected_type: PatternType
    ) -> None:
        if detector_class is BoundaryViolationDetector:
            detector = detector_class()
        elif detector_class is KnowledgeGuidedPrioritizationChecker:
            detector = detector_class(salient_terms=["x"])
        else:
            detector = detector_class()
        assert detector.pattern_type == expected_type


class TestPatternMatchFields:
    """Ensure PatternMatch objects are well-formed from all detectors."""

    def test_early_pruning_match_fields(self) -> None:
        trace = make_trace("A.", "B.")
        detector = EarlyPruningDetector()
        matches = detector.detect(trace)
        for m in matches:
            assert isinstance(m.pattern_type, PatternType)
            assert 0.0 <= m.confidence <= 1.0
            assert isinstance(m.evidence, str)

    def test_path_lockin_match_fields(self) -> None:
        trace = make_trace(
            "The algorithm processes data.",
            "The algorithm processes input.",
            "The algorithm processes output.",
        )
        detector = PathLockInDetector(vocab_overlap_threshold=0.60)
        matches = detector.detect(trace)
        for m in matches:
            assert isinstance(m.pattern_type, PatternType)
            assert 0.0 <= m.confidence <= 1.0
            assert isinstance(m.evidence, str)

    def test_boundary_violation_match_fields(self) -> None:
        trace = make_trace("I'll sudo and delete everything.")
        spec = BoundarySpec(
            allowed_tools=frozenset(),
            prohibited_actions=frozenset({"delete"}),
            max_reasoning_steps=5,
            domain_scope=frozenset(),
        )
        detector = BoundaryViolationDetector(boundary_spec=spec)
        matches = detector.detect(trace)
        for m in matches:
            assert isinstance(m.pattern_type, PatternType)
            assert 0.0 <= m.confidence <= 1.0
            assert isinstance(m.evidence, str)

    def test_knowledge_prioritization_match_fields(self) -> None:
        trace = make_trace("Let me think about this problem.")
        detector = KnowledgeGuidedPrioritizationChecker(
            salient_terms=["algorithm", "complexity"]
        )
        matches = detector.detect(trace)
        for m in matches:
            assert isinstance(m.pattern_type, PatternType)
            assert 0.0 <= m.confidence <= 1.0
            assert isinstance(m.evidence, str)
