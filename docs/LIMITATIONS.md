# Limitations

PonderScope is intentionally conservative about what its evidence supports.

## Scope

- **Single machine, single runtime device.** This pass measures one Apple
  Silicon machine under MLX/MLX-LM. Results are properties of *this deployment*,
  not of the model alone.
- **Single model revision.** `Qwen/Qwen3.5-0.8B` at an exact revision. No
  generalization to other weights or sizes.
- **Small procedural tasks.** The `tasks-v1` pack validates methodology. It is
  not a universal intelligence benchmark and its difficulty scale is structural,
  not semantic.
- **No causal claims.** Deployment differences are associations observed under
  this protocol. Runtime, precision, quantization, and hardware are not
  independently randomized.
- **Multimodal checkpoint caveat.** The text architecture drops vision-tower and
  MTP weights at load (`strict=False`), recorded in every manifest.

## Measurement

- Greedy determinism is a same-process, same-machine replay test; it does not
  prove determinism across processes, sessions, or machines.
- Per-token timing includes capture overhead; a bare baseline is measured
  separately.
- Prefix probes force finalization with the model's think-end token. This is an
  **oracle-style intervention**, not a deployable stopping policy, and it biases
  the model toward answering at the probe point.
- Where the thinking channel was never closed, trajectories are **censored**, not
  natural stops. Censoring is reported, never treated as termination.
- Semantic trace similarity is **NOT YET TESTED**; only exact/token-level and
  lexical measures are implemented. No proprietary cloud judge is used.
- Entropy and logprobs are only available because MLX-LM exposes them. Other
  backends may not, and would be marked UNSUPPORTED.

## Statistics

- Accuracy CIs and paired deltas are **task-clustered** bootstraps (the task is
  the statistical unit); this avoids pseudo-replication from repeated seeds.
- Comparisons are preceded by a configuration-difference / confound check and
  are refused when the contrast is not clean, unless run as exploratory.
- Bootstrap CIs do not account for multiple comparisons across every reported
  metric.
- With few tasks per family, seed-driven variance estimates are coarse.

## Not yet implemented / tested

- Learned stopping controllers (by design, future scope).
- Policy transfer between deployments (contract documented in
  `METHODOLOGY.md`; experiment NOT YET TESTED).
- Cross-runtime and cross-hardware comparisons (only one deployment is
  guaranteed in this pass).
- Semantic reasoning-equivalence measures.

## Measured on this deployment (2026-10-05; amended Phase 1.2)

- **Natural closure is expensive.** On `Qwen/Qwen3.5-0.8B`@`2fc06364…`, the live
  capability gate found natural closure after ~593–601 reasoning tokens on a
  trivial arithmetic prompt. Budgets of 192–512 were fully censored.
- **Censoring is not incorrectness.** Under the 640-token budget, 8/10 greedy and
  20/20 sampled generations remained in the reasoning channel. `success_at_budget`
  was 0.2 (greedy) and 0.0 (sampled); among greedy runs that actually completed,
  conditional accuracy was 1.0. A capped trajectory is never scored as a wrong
  answer (`PHASE1_2_CORRECTION.md`).
- **Instrumentation is not free.** Research capture cost ~23% and digest capture
  ~25% of decode throughput versus minimal capture. The minimal lane is the
  primary performance measurement; research/digest throughput is never presented
  as native deployment throughput.
- Greedy replay and same-seed sampled replay were exactly token-identical in the
  smoke. Across-seed **final-answer** variation is unobserved because sampled runs
  were censored; token-level variation remains measurable.
- **Sealing defect (Phase 1.1, fixed in Phase 1.2).** The old `seal()` hashed the
  manifest before its final status/timestamp mutation, so the sealed smoke fails
  final-manifest verification. New runs verify PASS; the old run is preserved as
  historical.
