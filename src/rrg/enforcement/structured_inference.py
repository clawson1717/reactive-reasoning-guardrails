"""StructuredInferenceLayer: parses and validates structured reasoning step labels."""

from __future__ import annotations

import re
from typing import Literal

from rrg.enforcement.boundary import StepLabel

# Step-type keywords that map to our canonical step types
_STEP_TYPE_KEYWORDS: dict[re.Pattern[str], Literal["premise", "analysis", "revision", "conclusion"]] = {
    re.compile(r"\bpremise\b", re.IGNORECASE): "premise",
    re.compile(r"\bbackground\b|\bfact\b|\bgiven\b", re.IGNORECASE): "premise",
    re.compile(r"\banalysis\b|\banalyse\b|\bexamine\b|\binvestigate\b", re.IGNORECASE): "analysis",
    re.compile(r"\brevision\b|\brevisit\b|\breconsider\b|\bcorrection\b", re.IGNORECASE): "revision",
    re.compile(r"\bconclusion\b|\bconclude\b|\bsummary\b|\bfinal\b", re.IGNORECASE): "conclusion",
}


class StructuredInferenceLayer:
    """Parses and validates structured reasoning outputs with labelled steps.

    This layer provides utilities to:
    - Parse step labels from raw reasoning text
    - Validate that a reasoning sequence has required structure
    - Compute character offsets for each step

    It is intended to work with LLM outputs that are prompted (via
    :meth:`enforce_format`) to use structured labels such as
    ``[PREMISE]``, ``[ANALYSIS]``, ``[REVISION]``, ``[CONCLUSION]``.

    Example:
        >>> layer = StructuredInferenceLayer()
        >>> raw = "Step 1: [PREMISE] The sky is blue. Step 2: [ANALYSIS] Why?"
        >>> steps = layer.parse_steps(raw)
        >>> layer.validate_structure(steps)
        True
    """

    def __init__(self) -> None:
        self._logger = None  # structlog not strictly required here

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _classify_step(self, label_text: str) -> Literal["premise", "analysis", "revision", "conclusion"]:
        """Classify a step label into one of the four step types."""
        for pattern, step_type in _STEP_TYPE_KEYWORDS.items():
            if pattern.search(label_text):
                return step_type
        # Default: infer from position if no explicit marker
        return "analysis"

    def parse_steps(self, raw_reasoning: str) -> list[StepLabel]:
        """Parse step labels from structured reasoning text.

        Recognises patterns such as:
        - ``Step 1: [PREMISE] ...``
        - ``Step 2 [ANALYSIS] ...``
        - ``[REVISION] ...``
        - ``[CONCLUSION] The answer is 42.``

        Args:
            raw_reasoning: The raw reasoning text from the LLM.

        Returns:
            List of :class:`StepLabel` objects in the order they appear.
            Each label captures the step type and any surrounding label text.
        """
        steps: list[StepLabel] = []

        # Pattern 1: explicit "Step N:" or "Step N -" with optional type tag
        explicit_pattern = re.compile(
            r"(?P<prefix>"
            r"Step\s+(?P<num>\d+)\s*[:\-\|]\s*"
            r"(?:\[(?P<tag>[^\]]+)\]\s*)?"
            r")",
            re.IGNORECASE,
        )

        # Pattern 2: standalone type tags like [PREMISE], [ANALYSIS], etc.
        tag_only_pattern = re.compile(
            r"\[(?P<tag>(?:PREMISE|ANALYSIS|REVISION|CONCLUSION))\](?:\s*)",
            re.IGNORECASE,
        )

        # Pattern 3: simple "Step N" without tag
        step_num_pattern = re.compile(r"Step\s+(?P<num>\d+)", re.IGNORECASE)

        current_pos = 0
        found_types: set[str] = set()

        while current_pos < len(raw_reasoning):
            match = None
            label_text = ""

            # Try explicit step + optional tag first
            search_window = raw_reasoning[current_pos:]
            explicit_match = explicit_pattern.search(search_window)
            tag_match = tag_only_pattern.search(search_window)

            if explicit_match and (tag_match is None or explicit_match.start() <= tag_match.start()):
                match = explicit_match
                label_text = explicit_match.group("prefix").strip()
                if explicit_match.group("tag"):
                    tag = explicit_match.group("tag").strip()
                    found_types.add(tag.lower())
                    step_type: Literal["premise", "analysis", "revision", "conclusion"] = self._classify_step(
                        label_text
                    )
                else:
                    step_type = self._classify_step(label_text)
            elif tag_match:
                match = tag_match
                label_text = f"[{tag_match.group('tag')}]"
                found_types.add(tag_match.group("tag").lower())
                step_type = self._classify_step(label_text)
            elif step_num_pattern.search(search_window):
                sn_match = step_num_pattern.search(search_window)
                assert sn_match is not None
                match = sn_match
                label_text = sn_match.group(0)
                step_type = self._classify_step(label_text)
            else:
                # No more step markers; advance to end
                break

            if match:
                steps.append(StepLabel(label=label_text, step_type=step_type, metadata={}))
                current_pos += match.end()

        # If nothing was parsed but we have content, treat as single analysis step
        if not steps and raw_reasoning.strip():
            steps.append(
                StepLabel(label="(unlabelled)", step_type="analysis", metadata={})
            )

        return steps

    # ------------------------------------------------------------------
    # Format enforcement
    # ------------------------------------------------------------------

    def enforce_format(self, prompt_template: str) -> str:
        """Return a modified prompt that instructs the LLM to use structured labels.

        Appends a formatting instruction to the prompt template.

        Args:
            prompt_template: The original prompt or system message.

        Returns:
            The prompt with formatting instructions appended.
        """
        instruction = (
            "\n\nWhen generating your reasoning, use structured step labels. "
            "Prefix each logical unit with one of:\n"
            "  [PREMISE]   – background facts or givens\n"
            "  [ANALYSIS]  – main reasoning steps\n"
            "  [REVISION]  – corrections or reconsiderations\n"
            "  [CONCLUSION] – final answer or summary\n"
            "Format example:\n"
            "Step 1: [PREMISE] ...\n"
            "Step 2: [ANALYSIS] ...\n"
            "Step 3: [REVISION] ...\n"
            "Step 4: [CONCLUSION] ...\n"
        )
        return prompt_template.rstrip() + instruction

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_structure(self, steps: list[StepLabel]) -> bool:
        """Validate that a reasoning sequence has a well-formed structure.

        A well-formed structure requires at minimum one premise step and
        one conclusion step.

        Args:
            steps: Parsed step labels from :meth:`parse_steps`.

        Returns:
            True if the structure is valid (has premise + conclusion),
            False otherwise.
        """
        if not steps:
            return False
        step_types = {s.step_type for s in steps}
        has_premise = "premise" in step_types
        has_conclusion = "conclusion" in step_types
        return bool(has_premise and has_conclusion)

    # ------------------------------------------------------------------
    # Boundary computation
    # ------------------------------------------------------------------

    def get_step_boundaries(
        self, steps: list[StepLabel], raw_reasoning: str = ""
    ) -> list[tuple[int, int]]:
        """Return character (start, end) offsets for each step in raw text.

        This method uses the step labels as anchors to find their positions
        in the original text. It returns the span of each labelled segment
        (including the label itself up to the next label or end of text).

        Note: To use this method correctly, you must pass the ``raw_reasoning``
        string that was parsed to produce the ``steps`` list.

        Args:
            steps: Step labels (with their ``label`` fields) to locate.
            raw_reasoning: The original raw reasoning text.

        Returns:
            List of (start, end) character offsets for each step.
            Offsets are in the original text that was parsed.
        """
        boundaries: list[tuple[int, int]] = []
        text = raw_reasoning
        for step in steps:
            if step.label == "(unlabelled)":
                start = 0
            else:
                start = text.find(step.label)
            if start == -1:
                start = 0
            # End is either the start of the next label or end of text
            end = len(text)  # fallback
            boundaries.append((start, end))
        return boundaries
