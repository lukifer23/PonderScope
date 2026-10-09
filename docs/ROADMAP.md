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

The Phase 1.6 independent replication is **complete** (30 unseen tasks, 60 draws
per arm, under a frozen population lock): no distinguishable reasoning-behavior
difference (closure +0.067 [−0.033, +0.183]; RMST −48.1 [−171.9, +62.3]; success
+0.050 [−0.017, 0.133]) with a large speed advantage (≈2.9×). Because the
closure-rate interval is still within the ≈0.165 planning half-width, the next
step is a **larger independent-task population** (more tasks, not more seeds) if
a small closure-rate effect is to be resolved — the held-out test/1729
population must not be reused for a new primary claim.

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
