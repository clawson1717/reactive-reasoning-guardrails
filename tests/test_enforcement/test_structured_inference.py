"""Tests for rrg.enforcement.structured_inference — StructuredInferenceLayer."""

from __future__ import annotations

import pytest

from rrg.enforcement.boundary import StepLabel
from rrg.enforcement.structured_inference import StructuredInferenceLayer


class TestStructuredInferenceLayerInit:
    def test_init_no_args(self) -> None:
        layer = StructuredInferenceLayer()
        assert layer is not None


class TestParseSteps:
    def test_parses_step_with_explicit_tag(self) -> None:
        layer = StructuredInferenceLayer()
        raw = "Step 1: [PREMISE] The sky is blue. Step 2: [ANALYSIS] Rayleigh scattering explains it."
        steps = layer.parse_steps(raw)
        assert len(steps) >= 1
        assert any(s.step_type == "premise" for s in steps)

    def test_parses_standalone_tag(self) -> None:
        layer = StructuredInferenceLayer()
        raw = "[PREMISE] Some facts. [ANALYSIS] More reasoning. [CONCLUSION] Done."
        steps = layer.parse_steps(raw)
        step_types = {s.step_type for s in steps}
        assert "premise" in step_types
        assert "analysis" in step_types
        assert "conclusion" in step_types

    def test_parses_step_numbers_without_tags(self) -> None:
        layer = StructuredInferenceLayer()
        raw = "Step 1: First idea. Step 2: Second idea."
        steps = layer.parse_steps(raw)
        assert len(steps) >= 1

    def test_parses_revision_tag(self) -> None:
        layer = StructuredInferenceLayer()
        raw = "[REVISION] Actually, I need to reconsider."
        steps = layer.parse_steps(raw)
        assert len(steps) == 1
        assert steps[0].step_type == "revision"

    def test_parses_conclusion_tag(self) -> None:
        layer = StructuredInferenceLayer()
        raw = "[CONCLUSION] Therefore, the answer is 42."
        steps = layer.parse_steps(raw)
        assert len(steps) == 1
        assert steps[0].step_type == "conclusion"

    def test_falls_back_to_analysis_for_unlabelled(self) -> None:
        layer = StructuredInferenceLayer()
        raw = "Just some free-form reasoning without any tags."
        steps = layer.parse_steps(raw)
        assert len(steps) == 1
        assert steps[0].step_type == "analysis"
        assert steps[0].label == "(unlabelled)"

    def test_empty_raw_returns_empty(self) -> None:
        layer = StructuredInferenceLayer()
        steps = layer.parse_steps("")
        assert steps == []


class TestEnforceFormat:
    def test_returns_string_with_instruction(self) -> None:
        layer = StructuredInferenceLayer()
        prompt = "Answer the following question."
        result = layer.enforce_format(prompt)
        assert "[PREMISE]" in result
        assert "[ANALYSIS]" in result
        assert "[REVISION]" in result
        assert "[CONCLUSION]" in result
        assert "Step 1:" in result

    def test_preserves_original_prompt(self) -> None:
        layer = StructuredInferenceLayer()
        prompt = "What is 2+2?"
        result = layer.enforce_format(prompt)
        assert result.startswith("What is 2+2?")


class TestValidateStructure:
    def test_valid_has_premise_and_conclusion(self) -> None:
        layer = StructuredInferenceLayer()
        steps = [
            StepLabel(label="[PREMISE]", step_type="premise"),
            StepLabel(label="[ANALYSIS]", step_type="analysis"),
            StepLabel(label="[CONCLUSION]", step_type="conclusion"),
        ]
        assert layer.validate_structure(steps) is True

    def test_invalid_missing_premise(self) -> None:
        layer = StructuredInferenceLayer()
        steps = [
            StepLabel(label="[ANALYSIS]", step_type="analysis"),
            StepLabel(label="[CONCLUSION]", step_type="conclusion"),
        ]
        assert layer.validate_structure(steps) is False

    def test_invalid_missing_conclusion(self) -> None:
        layer = StructuredInferenceLayer()
        steps = [
            StepLabel(label="[PREMISE]", step_type="premise"),
            StepLabel(label="[ANALYSIS]", step_type="analysis"),
        ]
        assert layer.validate_structure(steps) is False

    def test_invalid_single_step(self) -> None:
        layer = StructuredInferenceLayer()
        steps = [StepLabel(label="[ANALYSIS]", step_type="analysis")]
        assert layer.validate_structure(steps) is False

    def test_invalid_empty_steps(self) -> None:
        layer = StructuredInferenceLayer()
        assert layer.validate_structure([]) is False

    def test_revision_plus_premise_plus_conclusion_is_valid(self) -> None:
        layer = StructuredInferenceLayer()
        steps = [
            StepLabel(label="[PREMISE]", step_type="premise"),
            StepLabel(label="[ANALYSIS]", step_type="analysis"),
            StepLabel(label="[REVISION]", step_type="revision"),
            StepLabel(label="[CONCLUSION]", step_type="conclusion"),
        ]
        assert layer.validate_structure(steps) is True


class TestGetStepBoundaries:
    def test_returns_list_of_tuples(self) -> None:
        layer = StructuredInferenceLayer()
        raw = "Step 1: [PREMISE] Hello."
        steps = layer.parse_steps(raw)
        boundaries = layer.get_step_boundaries(steps, raw)
        assert len(boundaries) == len(steps)
        assert all(isinstance(b, tuple) and len(b) == 2 for b in boundaries)

    def test_boundaries_have_valid_offsets(self) -> None:
        layer = StructuredInferenceLayer()
        raw = "Step 1: [PREMISE] Hello."
        steps = layer.parse_steps(raw)
        boundaries = layer.get_step_boundaries(steps, raw)
        for start, end in boundaries:
            assert 0 <= start <= end <= len(raw)

    def test_unlabelled_step_gets_zero_start(self) -> None:
        layer = StructuredInferenceLayer()
        raw = "Some free text without labels."
        steps = layer.parse_steps(raw)
        boundaries = layer.get_step_boundaries(steps, raw)
        # Unlabelled step should start at 0
        assert boundaries[0][0] == 0
