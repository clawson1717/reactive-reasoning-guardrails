"""Hybrid 2-sample AUROC uncertainty estimator.

Uses semantic embedding disagreement between model-generated samples
and a reference/larger-model set to estimate uncertainty.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()


class EmbeddingModel(ABC):
    """Abstract embedding provider for semantic similarity."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a dense embedding vector for text."""

    @abstractmethod
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embedding vectors."""


@dataclass(frozen=True)
class UncertaintyScore:
    """Output of the uncertainty estimator."""

    score: float  # 0.0 (certain) to 1.0 (highly uncertain)
    auroc: float  # AUROC value for the 2-sample comparison
    mean_agreement: float  # mean pairwise similarity within samples
    n_samples: int  # number of samples used
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    @property
    def is_uncertain(self) -> bool:
        """Whether the score exceeds a conservative threshold."""
        return self.score > 0.5


class UncertaintyEstimator(ABC):
    """Hybrid 2-sample AUROC estimator for LLM reasoning uncertainty.

    The estimator compares embedding-level agreement between:
    - Primary samples: outputs from the model being monitored
    - Reference samples: outputs from a larger/trusted model or a second run

    High disagreement (low AUROC / low similarity) suggests the primary
    model is uncertain about its own reasoning.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        sample_size: int = 5,
        seed: int | None = 42,
    ) -> None:
        self.embedding_model = embedding_model
        self.sample_size = sample_size
        self.rng = np.random.default_rng(seed)
        self._logger = logger.bind(component="UncertaintyEstimator")

    def estimate(
        self,
        primary_samples: list[str],
        reference_samples: list[str],
    ) -> UncertaintyScore:
        """Compute uncertainty score from two sets of samples.

        Args:
            primary_samples: Outputs from the monitored model.
            reference_samples: Outputs from a reference/larger model.

        Returns:
            UncertaintyScore with score, AUROC, and metadata.
        """
        if len(primary_samples) < 2 or len(reference_samples) < 2:
            logger.warning("insufficient_samples", primary=len(primary_samples), reference=len(reference_samples))
            return UncertaintyScore(
                score=1.0,
                auroc=0.5,
                mean_agreement=0.0,
                n_samples=min(len(primary_samples), len(reference_samples)),
            )

        auroc, mean_agreement = self._compute_2_sample_auroc(
            primary_samples, reference_samples
        )
        score = float(np.clip(1.0 - auroc, 0.0, 1.0))

        return UncertaintyScore(
            score=score,
            auroc=auroc,
            mean_agreement=mean_agreement,
            n_samples=len(primary_samples) + len(reference_samples),
        )

    def _compute_2_sample_auroc(
        self,
        primary_samples: list[str],
        reference_samples: list[str],
    ) -> tuple[float, float]:
        """Compute AUROC between two sample groups using embedding similarity."""
        from scipy.spatial.distance import pdist, squareform
        from scipy.stats import rankdata

        # Embed all texts
        primary_emb = np.stack([self.embedding_model.embed(t) for t in primary_samples])
        reference_emb = np.stack([self.embedding_model.embed(t) for t in reference_samples])

        # Concatenate for pairwise similarity matrix
        all_emb = np.vstack([primary_emb, reference_emb])
        n_p = len(primary_emb)
        n_r = len(reference_emb)

        sim_matrix = np.zeros((n_p + n_r, n_p + n_r))
        for i in range(n_p + n_r):
            for j in range(i + 1, n_p + n_r):
                s = self.embedding_model.similarity(all_emb[i], all_emb[j])
                sim_matrix[i, j] = s
                sim_matrix[j, i] = s

        # Within-group similarities
        # Primary-primary similarities
        pp_dists: list[float] = []
        for i in range(n_p):
            for j in range(i + 1, n_p):
                pp_dists.append(sim_matrix[i, j])

        # Reference-reference similarities
        rr_dists: list[float] = []
        for i in range(n_p, n_p + n_r):
            for j in range(i + 1, n_p + n_r):
                rr_dists.append(sim_matrix[i, j])

        # Cross-group similarities
        cross_dists: list[float] = []
        for i in range(n_p):
            for j in range(n_p, n_p + n_r):
                cross_dists.append(sim_matrix[i, j])

        all_within = pp_dists + rr_dists
        mean_agreement = float(np.mean(all_within)) if all_within else 0.0

        # Simple AUROC: what fraction of cross-pairs are more similar than within-pairs?
        if not all_within or not cross_dists:
            return 0.5, mean_agreement

        all_within_arr = np.array(all_within)
        cross_arr = np.array(cross_dists)

        # AUROC computation
        n1, n2 = len(all_within_arr), len(cross_arr)
        if n1 == 0 or n2 == 0:
            return 0.5, mean_agreement

        # Mann-Whitney U statistic (higher within-group sim = "positive" class)
        # We want to know: is cross-group similarity lower?
        ranks = rankdata(np.concatenate([all_within_arr, cross_arr]))
        R1 = np.sum(ranks[:n1])  # ranks of within-group
        u1 = R1 - n1 * (n1 + 1) / 2
        auroc = u1 / (n1 * n2)

        return float(np.clip(auroc, 0.0, 1.0)), mean_agreement

    def estimate_from_single(
        self,
        samples: list[str],
        rerun_samples: list[str] | None = None,
    ) -> UncertaintyScore:
        """Convenience: estimate uncertainty from one or two sets of samples.

        If only one set is provided, split it in half as primary/reference.
        """
        if rerun_samples is not None:
            return self.estimate(samples, rerun_samples)

        n = len(samples)
        if n < 4:
            return UncertaintyScore(score=1.0, auroc=0.5, mean_agreement=0.0, n_samples=n)

        mid = n // 2
        return self.estimate(samples[:mid], samples[mid:])


__all__ = [
    "EmbeddingModel",
    "UncertaintyEstimator",
    "UncertaintyScore",
]
