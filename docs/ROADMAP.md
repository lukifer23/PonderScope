# Roadmap

PonderScope stays focused on one scientific goal: **understand whether, and to
what measurable extent, changing the deployment of the same underlying model
changes its reasoning behavior.** It does not drift into generic benchmarking.

## Open research questions

1. Does a controlled Q4 representation of the same source revision change native
   reasoning closure or time-to-closure, relative to native BF16?
2. Is any change uniform across task families, or concentrated (e.g. `logic`)?
3. Does any observed effect exceed within-deployment stochastic variation?
4. Does the effect survive a larger independent-task replication?
5. Can an adaptive-stopping policy calibrated on one deployment be evaluated
   unchanged on another? *(contract documented; NOT YET TESTED)*

## Immediate next experiment

**Bounded prefix-probe diagnostic (Option C).** Determine whether censoring
conceals recoverable correct answers and whether Q4 changes answer availability
or stability along a reasoning trajectory, using the already-measured test/1729
tasks and fixed checkpoints (256/512/1024/1536). Exploratory, oracle-style
intervention, ≈2 h compute; see `docs/PHASE1_7_ANALYSIS.md`. Requires explicit
authorization before model inference. A **larger independent-task replication**
(Option A) remains the eventual confirmatory step; test/1729 must not be reused
as a fresh confirmatory population.

## Later, only after the above is established

- A larger independent-task replication if the primary contrast lacks precision.
- A second controlled quantization (e.g. 8-bit) as a dose-response contrast.
- A deployment-drift matrix (runtime/hardware) on the same machine.
- Adaptive-stopping policy-transfer evaluation (calibrate on A, evaluate on B).

## Explicitly out of scope for now

- Early-stopping controllers as a product.
- CUDA / llama.cpp / Ollama / vLLM / PyTorch backends.
- New task families before the original controlled contrast is established.
- Semantic reasoning-quality judges.
