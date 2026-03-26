"""Hybrid uncertainty estimation using 2-sample consistency + verbalized confidence fusion."""
from dataclasses import dataclass
from typing import Protocol, Any
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_auc_score, precision_recall_curve


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""
    def embed(self, text: str) -> np.ndarray: ...


@dataclass
class UncertaintyEstimate:
    score: float  # fused uncertainty U = α*consistency + (1-α)*verbalized
    consistency: float  # semantic consistency between two samples (0=disagree, 1=agree)
    verbalized: float  # verbalized confidence extracted from answer (0-1)
    above_threshold: bool
    sample1: str = ""
    sample2: str = ""


class HybridUncertaintyEstimator:
    def __init__(
        self,
        llm_backend: Any,  # LLMBackend from rrg.core
        embedding_model: EmbeddingModel,
        alpha: float = 0.7,
        tau: float = 0.5,
    ):
        self.llm = llm_backend
        self.embedding = embedding_model
        self.alpha = alpha
        self.tau = tau

    def _generate_samples(self, prompt: str, reasoning_step_text: str) -> tuple[str, str]:
        """Generate two temperature-shifted samples."""
        base = f"{prompt}\n{reasoning_step_text}"
        s1 = self.llm.complete(base, temperature=0.7, seed=42)
        s2 = self.llm.complete(base, temperature=1.3, seed=42)
        return s1, s2

    def _semantic_consistency(self, texts: list[str]) -> float:
        """Compute cosine similarity between embeddings of multiple samples.

        Returns a value in [0, 1], clamped so the result is always a valid
        consistency score (1=agree, 0=disagree).
        """
        if len(texts) < 2:
            return 1.0
        embeddings = [self.embedding.embed(t) for t in texts]
        # Pairwise cosine similarity
        sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = 1 - cosine(embeddings[i], embeddings[j])
                sims.append(sim)
        return float(np.clip(np.mean(sims), 0.0, 1.0))

    def _extract_verbalized_confidence(self, text: str) -> float:
        """Extract verbalized confidence from text (0-1)."""
        import re
        text_lower = text.lower()
        # Pattern: "i'm X% confident", "i am X% confident" (X% BEFORE confident)
        m = re.search(r"(?:i\s+am|i'm)\s*(\d+(?:\.\d+)?)\s*%\s*confident", text_lower)
        if m:
            return float(m.group(1)) / 100.0
        # Pattern: "confidence X%", "confident: X%", "probability X%"
        m = re.search(r"(?:confident|confidence|probability)[:\s]*(\d+(?:\.\d+)?)\s*%", text_lower)
        if m:
            return float(m.group(1)) / 100.0
        # Pattern: "i am certain", "i am sure" -> high
        if any(k in text_lower for k in ["i am certain", "i am sure", "definitely", "absolutely"]):
            return 0.9
        # Pattern: "i don't know", "uncertain", "not sure" -> low
        if any(k in text_lower for k in ["i don't know", "i am not sure", "uncertain", "i'm not sure"]):
            return 0.1
        # Pattern: "likely", "probably" -> moderate
        if "likely" in text_lower:
            return 0.65
        if "probably" in text_lower:
            return 0.55
        if "possibly" in text_lower:
            return 0.35
        # Default: moderate
        return 0.5

    def estimate(self, reasoning_step: Any) -> UncertaintyEstimate:
        """
        Given a ReasoningStep, generate two samples, compute consistency,
        extract verbalized confidence, and fuse into a single uncertainty score.
        """
        # Get text from reasoning step (support both actual ReasoningStep schema
        # and generic objects with 'text' or 'content' attributes)
        step_text = getattr(reasoning_step, 'content', None)
        if step_text is None:
            step_text = getattr(reasoning_step, 'text', None)
        if step_text is None:
            step_text = str(reasoning_step)
        prompt = getattr(reasoning_step, 'prompt', '')

        # Generate two samples
        s1, s2 = self._generate_samples(prompt, step_text)

        # Compute consistency
        consistency = self._semantic_consistency([step_text, s1, s2])

        # Extract verbalized confidence
        verbalized = self._extract_verbalized_confidence(s1)

        # Fuse
        score = self.alpha * (1 - consistency) + (1 - self.alpha) * (1 - verbalized)
        # Note: we invert so high uncertainty = high score
        # consistency=1 (agree) → 0 uncertainty; consistency=0 (disagree) → 1 uncertainty
        # verbalized=1 (confident) → 0 uncertainty; verbalized=0 (uncertain) → 1 uncertainty

        return UncertaintyEstimate(
            score=score,
            consistency=consistency,
            verbalized=verbalized,
            above_threshold=score > self.tau,
            sample1=s1,
            sample2=s2,
        )

    def calibrate(
        self, reasoning_pairs: list[tuple[Any, bool]],
    ) -> tuple[float, float]:
        """
        Given list of (ReasoningStep, is_error) pairs, find optimal alpha and tau
        that maximize AUROC for detecting errors.
        """
        estimates = [self.estimate(rs) for rs, _ in reasoning_pairs]
        scores = [e.score for e in estimates]
        labels = [int(is_err) for _, is_err in reasoning_pairs]

        # Grid search for best alpha (keep tau fixed for now, then optimize tau)
        best_auroc = 0
        best_alpha = self.alpha
        best_tau = self.tau

        for alpha in np.arange(0.0, 1.05, 0.1):
            # Recompute scores with this alpha
            adjusted_scores = []
            for e in estimates:
                adj = alpha * (1 - e.consistency) + (1 - alpha) * (1 - e.verbalized)
                adjusted_scores.append(adj)

            try:
                auroc = roc_auc_score(labels, adjusted_scores)
            except ValueError:
                continue

            if auroc > best_auroc:
                best_auroc = auroc
                best_alpha = alpha

        # Now optimize tau using Youden's J on best alpha
        final_scores = []
        for e in estimates:
            adj = best_alpha * (1 - e.consistency) + (1 - best_alpha) * (1 - e.verbalized)
            final_scores.append(adj)

        # Find optimal threshold using precision-recall curve
        precision, recall, thresholds = precision_recall_curve(labels, final_scores)
        # Youden's J: maximize (recall + (1 - precision) - 1) = recall - precision
        # Actually maximize F1-like: 2*prec*rec/(prec+rec)
        f1_scores = []
        for p, r in zip(precision, recall):
            if p + r > 0:
                f1_scores.append(2 * p * r / (p + r))
            else:
                f1_scores.append(0)
        # We want max F1, corresponding threshold
        best_idx = np.argmax(f1_scores)
        if best_idx < len(thresholds):
            best_tau = float(thresholds[best_idx])
        else:
            best_tau = self.tau

        self.alpha = best_alpha
        self.tau = best_tau
        return best_alpha, best_tau

    def get_auroc_score(
        self, reasoning_pairs: list[tuple[Any, bool]], alpha: float, tau: float
    ) -> float:
        """Compute AUROC score for given alpha/tau."""
        estimates = [self.estimate(rs) for rs, _ in reasoning_pairs]
        scores = [alpha * (1 - e.consistency) + (1 - alpha) * (1 - e.verbalized) for e in estimates]
        labels = [int(is_err) for _, is_err in reasoning_pairs]
        try:
            return roc_auc_score(labels, scores)
        except ValueError:
            return 0.0


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

import re as _re

# Compiled regex patterns for verbalized confidence extraction.
_CONFIDENCE_PATTERNS = [
    # Explicit percentage: "I'm 80% confident", "confidence: 85%"
    (_re.compile(r"(\d+(?:\.\d+)?)\s*%", _re.IGNORECASE), lambda m: float(m.group(1)) / 100.0),
    # Explicit probability: "probability ~0.7", "with probability 0.9"
    (_re.compile(r"(?:probability|likelihood)[:\s]+(?:~|of\s+)?(0?\.\d+)", _re.IGNORECASE), lambda m: float(m.group(1))),
    # Fraction: "about 3 out of 4"
    (_re.compile(r"(\d+)\s*/\s*(\d+)", _re.IGNORECASE), lambda m: float(m.group(1)) / float(m.group(2))),
    # Qualitative strong uncertainty cues
    (_re.compile(r"\b(very\s+)?certain\b", _re.IGNORECASE), lambda _: 0.9),
    (_re.compile(r"\b(very\s+)?confident\b", _re.IGNORECASE), lambda _: 0.8),
    (_re.compile(r"\blikely\b", _re.IGNORECASE), lambda _: 0.65),
    (_re.compile(r"\bmaybe\b", _re.IGNORECASE), lambda _: 0.45),
    (_re.compile(r"\b(very\s+)?uncertain\b", _re.IGNORECASE), lambda _: 0.2),
    (_re.compile(r"\b(very\s+)?unsure\b", _re.IGNORECASE), lambda _: 0.15),
    (_re.compile(r"\bnot\s+sure\b", _re.IGNORECASE), lambda _: 0.2),
    (_re.compile(r"\bdon't\s+know\b", _re.IGNORECASE), lambda _: 0.1),
    # Qualitative certainty cues
    (_re.compile(r"\bdefinitely\b", _re.IGNORECASE), lambda _: 0.95),
    (_re.compile(r"\bcertainly\b", _re.IGNORECASE), lambda _: 0.9),
    (_re.compile(r"\bobviously\b", _re.IGNORECASE), lambda _: 0.9),
    (_re.compile(r"\bclearly\b", _re.IGNORECASE), lambda _: 0.85),
    (_re.compile(r"\bnot\s+certain\b", _re.IGNORECASE), lambda _: 0.3),
    (_re.compile(r"\b(i'm|i am)\s+not\s+sure\b", _re.IGNORECASE), lambda _: 0.25),
    (_re.compile(r"\bno\s+idea\b", _re.IGNORECASE), lambda _: 0.1),
    (_re.compile(r"\allow\s+confidence\b", _re.IGNORECASE), lambda _: 0.2),
    (_re.compile(r"\bhigh\s+confidence\b", _re.IGNORECASE), lambda _: 0.85),
]


def extract_verbalized_confidence(text: str) -> float:
    """Extract a normalized confidence value [0, 1] from text.

    Scans for common calibration phrases and returns the first match.
    Falls back to 0.5 (neutral) when no phrase is found.

    Args:
        text: The model-generated text to parse.

    Returns:
        Normalized confidence in [0, 1].
    """
    for pattern, parser in _CONFIDENCE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                value = parser(match)
                return float(np.clip(value, 0.0, 1.0))
            except (ValueError, ZeroDivisionError):
                continue
    return 0.5


def find_optimal_threshold(
    uncertainty_scores: "np.ndarray",
    is_error_labels: "np.ndarray",
) -> float:
    """Find the optimal threshold using Youden's J statistic.

    Youden's J = max_{tau} (TPR - FPR)

    Args:
        uncertainty_scores: Array of uncertainty scores.
        is_error_labels: Array of binary error labels (1=error, 0=no error).

    Returns:
        Optimal threshold value.
    """
    import numpy as _np

    if len(uncertainty_scores) == 0:
        return 0.5

    scores = _np.asarray(uncertainty_scores)
    labels = _np.asarray(is_error_labels)

    unique_thresholds = _np.unique(scores)
    if len(unique_thresholds) < 2:
        return float(scores.mean()) if len(scores) > 0 else 0.5

    best_j = -1.0
    best_tau = 0.5

    for tau in unique_thresholds:
        predicted = (scores >= float(tau)).astype(int)
        tp = int(((predicted == 1) & (labels == 1)).sum())
        fp = int(((predicted == 1) & (labels == 0)).sum())
        tn = int(((predicted == 0) & (labels == 0)).sum())
        fn = int(((predicted == 0) & (labels == 1)).sum())

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        j = tpr - fpr

        if j > best_j:
            best_j = j
            best_tau = float(tau)

    return best_tau
