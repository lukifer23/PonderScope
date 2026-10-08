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

The **full 30-draw BF16-vs-controlled-Q4 contrast** on the frozen protocol
(`specs/phase1_4-4b-q4-calibration.json`), reported with task-clustered CIs and
censor-aware RMST. This is the single next experiment.

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
