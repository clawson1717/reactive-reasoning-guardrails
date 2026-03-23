"""reactive-reasoning-guardrails: Reactive guardrails for LLM reasoning agents.

Top-level public API exports.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Core
from rrg.core import (
    LLMBackend,
    ReasoningAgent,
    ReasoningAgentConfig,
    ReasoningResult,
    ReasoningStep,
    ReasoningTrace,
)

# Orchestrator
from rrg.core import (
    PatternDetectorRegistry,
    ReactiveReasoningLoop,
    ReactiveReasoningResult,
    ReasoningAuditLog,
)

# Patterns
from rrg.patterns import (
    BasePatternDetector,
    PatternDetector,
    PatternMatch,
    PatternType,
    ReasoningTrace as ReasoningTraceAlias,
    ReasoningStep as ReasoningStepAlias,
)

# Estimator
from rrg.estimator import (
    EmbeddingModel,
    UncertaintyEstimator,
    UncertaintyScore,
)

# Corrector
from rrg.corrector import (
    CorrectionAction,
    CorrectionEngine,
    CorrectionResult,
    CorrectionStrategy,
    CorrectionStrategyHandler,
)

# Monitor
from rrg.monitor import (
    GuardrailConfig,
    GuardrailDecision,
    GuardrailMonitor,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "LLMBackend",
    "ReasoningAgent",
    "ReasoningAgentConfig",
    "ReasoningResult",
    "ReasoningStep",
    "ReasoningTrace",
    # Orchestrator
    "PatternDetectorRegistry",
    "ReactiveReasoningLoop",
    "ReactiveReasoningResult",
    "ReasoningAuditLog",
    # Patterns
    "BasePatternDetector",
    "PatternDetector",
    "PatternMatch",
    "PatternType",
    # Estimator
    "EmbeddingModel",
    "UncertaintyEstimator",
    "UncertaintyScore",
    # Corrector
    "CorrectionAction",
    "CorrectionEngine",
    "CorrectionResult",
    "CorrectionStrategy",
    "CorrectionStrategyHandler",
    # Monitor
    "GuardrailConfig",
    "GuardrailDecision",
    "GuardrailMonitor",
]
