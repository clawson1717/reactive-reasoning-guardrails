"""Boundary Enforcement Layer: enforces operational boundaries on agent reasoning."""

from __future__ import annotations

from rrg.enforcement.boundary import BoundarySpec, BoundaryViolationError, StepLabel
from rrg.enforcement.enforcement_layer import BoundaryEnforcementLayer
from rrg.enforcement.grounding import ContextExpander, MemoryGroundingLayer
from rrg.enforcement.structured_inference import StructuredInferenceLayer

__all__ = [
    "BoundarySpec",
    "BoundaryViolationError",
    "BoundaryEnforcementLayer",
    "ContextExpander",
    "MemoryGroundingLayer",
    "StructuredInferenceLayer",
    "StepLabel",
]
