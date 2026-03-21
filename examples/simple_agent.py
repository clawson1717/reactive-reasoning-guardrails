"""Minimal example: wrap an LLM with reactive reasoning guardrails.

This example demonstrates the basic architecture without requiring
any actual LLM API calls — it uses the mock backend from tests.
"""

from __future__ import annotations

from tests.conftest import MockReasoningAgent, MockEmbeddingModel
from rrg.estimator import UncertaintyEstimator
from rrg.monitor import GuardrailMonitor


def main() -> None:
    """Run a prompt through a guarded reasoning agent."""
    # 1. Create a mock LLM agent (swap MockReasoningAgent for your impl)
    agent = MockReasoningAgent()

    # 2. Wire up an uncertainty estimator (mock embedding for demo)
    embedding_model = MockEmbeddingModel(dim=64)
    uncertainty = UncertaintyEstimator(embedding_model=embedding_model, sample_size=4)

    # 3. Build the guardrail monitor
    monitor = GuardrailMonitor(
        agent=agent,
        uncertainty_estimator=uncertainty,
    )

    # 4. Run your prompt through the monitor
    prompt = "What is the capital of France?"
    decision = monitor.run(prompt)

    # 5. Inspect the decision
    print(f"Prompt:     {prompt}")
    print(f"Response:   {decision.response}")
    print(f"Accepted:   {decision.accepted}")
    if decision.uncertainty_score:
        print(f"Uncertainty: {decision.uncertainty_score.score:.3f}")
    print(f"Patterns:   {len(decision.triggered_patterns)} triggered")


if __name__ == "__main__":
    main()
