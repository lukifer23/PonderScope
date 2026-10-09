# Phase 1.7 — analysis, censoring diagnostics, and research direction

This document interprets the existing BF16/Q4 evidence, records the Phase 1.7
measurement correction, and proposes the next experiment. It is **manual
interpretation of generated analysis**, kept distinct from the recorded raw
evidence and the automatically generated reports.

## 1. What Phase 1.4 established

On 10 calibration tasks (3 seeds, 30 matched pairs) the controlled Q4 deployment
was not distinguishable from native BF16 on primary reasoning behavior:
closure-rate +0.033, RMST(2048) −117.3 tokens, success 0.000 — no interval
excluded zero — while Q4 was ≈3.1× faster. `logic` was censored 6/6 in both arms.

## 2. What Phase 1.6 independently reproduced

On 30 previously unseen test tasks (2 seeds, 60 matched pairs), under a frozen
population lock: closure-rate +0.067 (95% CI [−0.033, +0.183]), RMST −48.1
([−171.9, +62.3]), success +0.050 ([−0.017, 0.133]) — again no interval excluded
zero — with Q4 ≈2.9× faster (37.0 vs 12.8 tok/s). The held-out population is now
measured.

## 3. Are the two results consistent?

Yes. Both show a small, positive, non-significant closure-rate point estimate, a
small negative RMST point estimate, and a large efficiency advantage. Neither
study is powered to resolve a ~0.10 closure-rate effect (Phase 1.6 planning
half-width ≈0.165). The two are **consistent and jointly inconclusive**, not
confirmatory evidence of equivalence.

## 4. What the primary metrics do and do not establish

- **Do:** the same source revision run as Q4 is ~3× faster with no *detectable*
  primary reasoning-behavior difference at this precision.
- **Do not:** establish equivalence, causality, or that a stopping policy
  transfers. A null result is not equivalence; a `[0,0]` interval (e.g.
  conditional accuracy) reflects absent variation, not proven equality.

## 5. What censoring prevents us from observing

At 2048 tokens most trajectories are right-censored: BF16 33/60, Q4 29/60
(Phase 1.6, corrected). Censoring is concentrated in `logic` (0/12 closed in both
arms) and `path`. For these tasks we never observe a natural final answer, so
success-at-budget and closure rate cannot separate "would have answered correctly
with more tokens" from "looped indefinitely".

## 6. Task-family and difficulty breakdown (Phase 1.6)

| family | BF16 closures | Q4 closures | discordant pairs |
|---|---|---|---|
| arith | 9/12 | 11/12 | Q4-only 3, BF16-only 1 |
| order | 10/12 | 11/12 | Q4-only 1 |
| sm | 7/12 | 7/12 | Q4-only 2, BF16-only 2 |
| path | 1/12 | 2/12 | Q4-only 2, BF16-only 1 |
| logic | 0/12 | 0/12 | none |

Paired transitions (`runs/phase1_7-paired-outcomes-replication.json`): closure
`both 23, neither 25, Q4-only 8, BF16-only 4`; correctness `both 23, neither 28,
Q4-only 6, BF16-only 3`. The net closure advantage is +4 draws (8 vs 4
discordant); it is not a uniform shift but a handful of task/seed flips.

Difficulty is a strong correlate: `hard` tasks contribute the most discordance
and censoring (`per_difficulty`: hard `neither 10, both 3, Q4-only 4, BF16-only
3`). This is an **association**, not a causal explanation.

## 7. Cross-population difference is a task-population effect

The calibration population was easy/medium only (5/5); the test population adds
`hard` (10/10/10). Success dropped from 0.633 to 0.433/0.483 across the two
studies. This is explained by the **harder task mix**, not by any deployment
change; the two populations are not comparable as if they were the same tasks.

## 8. Outcome-classification correction (Phase 1.7)

A confirmed defect: an extracted final-channel string that fails normalization
was classified `incorrect`. Phase 1.7 recomputes `natural_final_status` from the
immutable record fields: a raw-but-unscorable answer is `unparseable`, never a
wrong answer, and a generation capped **after** the native close
(`budget_exhausted_after_closure`) is distinguished from one capped inside
reasoning. Effect on existing derived results:

| run | change |
|---|---|
| Phase 1.6 BF16 | 1 record `incorrect` → `unparseable`; `unparseable_rate` 0 → 0.017 |
| Phase 1.6 Q4 | 1 of 2 `incorrect` → `unparseable`; 1 remains genuinely incorrect |
| Phase 1.4 full | unchanged (2 genuinely scorable-incorrect) |

Primary metrics (closure, RMST, success, conditional accuracy) are **unchanged**;
only the `unparseable` category and rate change. Interpretation is versioned
`phase1.7`; prior analyses are preserved as `analysis.phase1_6.json`.

## 9. Secondary repetition finding (exploratory)

The repeated-4gram difference flips sign between studies: Phase 1.4 Q4 higher
(+0.013), Phase 1.6 Q4 lower (−0.017, unadjusted CI [−0.0355, −0.0003]).
Stratification (`repetition` block) shows the Phase 1.6 difference is concentrated
in `order`/`path`/`sm`, in **both-closed** trajectories (+0.039), and in short
reasoning (+0.042) — not in the censored loops and not in `logic` (+0.001).
**Conclusion:** this is ordinary repeated phrasing in closed reasoning, not
evidence of a quantization-induced reduction in degenerate loops. It is reported
as **exploratory** and is not retrofitted into the primary hypotheses; a
confirmatory test would need a preregistered metric.

## 10. The next research question

> Do BF16 and Q4 differ in how early a correct answer becomes recoverable from a
> reasoning prefix — including trajectories that never terminate naturally?

This is distinct from final-answer accuracy, and censoring currently hides it.

## Proposed bounded prefix-probe diagnostic (Option C, principal next experiment)

**Design (exploratory, post-outcome).** Reuse the already-measured test/1729
population under the frozen lock (`specs/phase1_7-prefix-probe-diagnostic.json`).
Per task, one greedy trajectory per deployment; forced finalization from fixed
checkpoints **256/512/1024/1536** tokens (bounded by the actual reasoning length;
short trajectories simply lack the longer checkpoints), using the model's
validated native closing sequence, `probe_max_tokens=64`. Record exact prompt and
prefix identities, checkpoint lengths, intervention semantics, trial IDs, and
provenance. New probes are stored as **separate verified evidence**, never
appended to historical sealed runs.

**Interpretability constraints.** Forced finalization is an oracle-style
intervention. A correct forced answer does **not** establish natural closure,
natural accuracy, or a deployable stopping policy. Recoverability, closure,
natural accuracy, stable recoverability, and policy transfer are kept separate.

**Predeclared selection.** Analyze all 30 tasks; report `logic` and `path`
(censored) and the discordant-closure tasks separately. Selection is
post-outcome and labelled exploratory; the tasks are not a fresh confirmatory
population.

**Outcomes (all informative):** A both recover early despite censoring; B one
deployment needs longer prefixes; C both fail despite extensive reasoning; D
stability differs; E too noisy/intervention-sensitive.

## Candidate directions — decision table

| option | information gained | relation to hypothesis | cost | power/limits | new work | risk |
|---|---|---|---|---|---|---|
| **A** larger replication (50–100 tasks) | tighter primary CIs | direct | 50 tasks ≈4 h; 100 ≈8 h | resolves ~0.10 only at ~100 tasks | new population | diminishing value on censored families |
| **B** longer horizon (4096) | do logic/path terminate? | direct | ~4 h for logic+path | separate estimand | none | mixes horizon with estimand |
| **C** prefix-probe diagnostic | answer recoverability under censoring | direct (answer recoverability) | **≈2 h** | exploratory; intervention-sensitive | checkpoint support (done) | over-interpreting oracle probes |
| **D** `tasks-v2` | better discrimination | indirect | high (design) | new estimand | large | outcome-driven task selection |
| **E** policy transfer | adaptive-compute value | long-term | high | needs validated probes first | large | premature |

## Recommendation

**Principal next experiment: Option C** (bounded prefix-probe diagnostic). It
produces information the current measurements cannot reveal (whether censoring
conceals recoverable correct answers and whether Q4 changes answer availability
or stability), at ≈2 h compute, using existing infrastructure. **Option A**
(a larger independent-task replication) remains the eventual confirmatory step
once the diagnostic indicates whether closure differences matter. Do **not**
pursue D or E yet.

## Compute estimates (from measured throughput)

- **C:** BF16 30×(1 greedy ≈113 s + ~5 probes ≈75 s) ≈ 1.6 h; Q4 30×(33 s + ~25 s)
  ≈ 0.5 h → **≈2 h total**. Disk ≪ 100 MB.
- **A (50 tasks):** BF16 100×112.6 s ≈ 3.1 h; Q4 100×33 s ≈ 0.9 h → **≈4 h**.
- **B (logic+path, 24 tasks × 2 seeds, 4096):** ≈2× per generation → BF16 ≈3 h,
  Q4 ≈0.9 h → **≈4 h**.

No experiment in this document is executed. Option C requires explicit
authorization before model inference.

## What should NOT be done

- Do not pool the calibration and replication populations into one estimate.
- Do not reuse test/1729 as a fresh confirmatory population.
- Do not claim equivalence from a null result or a `[0,0]` interval.
- Do not describe the repeated-4gram difference as a validated loop reduction.
- Do not treat forced-probe recoverability as natural closure or a policy.
- Do not build `tasks-v2` or a learned controller yet.
