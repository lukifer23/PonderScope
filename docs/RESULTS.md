# Results index

**Generated from saved evidence; no numbers are hand-entered.** Regenerate any
derived number with `ponderscope analyze` / `report` from `runs/<id>/`. The
detailed original reports are preserved and linked below; this page indexes the
current state and clearly marks superseded values.

## Current best evidence

### Primary BF16 baseline — `Qwen/Qwen3.5-4B` @ `851bf6e8…`

- Run: `runs/20261006T024435Z-calibration-4b-generated-history-api-dep-e1c20569173a`
- Deployment: `dep-e1c20569173a`, weight variant `wvar-9b6f9b40b789`, source
  `src-6209ef70e318`; native BF16, MLX-LM 0.32.0.
- Design: tasks-v1 calibration, pp-v1, generated-history presence policy,
  2048-token horizon, seeds {0,1,2} × one execution = **30 unique stochastic
  draws**. `ponderscope verify` = PASS.
- Native reasoning closures **20/30 (66.7%)**; censored 10/30; success-at-budget
  0.633; conditional accuracy given completion 1.0.
- Corrected phase1.4 endpoint (event time = tokens through the native think-end):
  **RMST(2048) = 1368.8**, median tokens-to-closure **1569**.
  Historical Phase 1.3B endpoint (event time = total generated tokens):
  RMST 1511.9, median 1718 — superseded, preserved in the on-disk
  `analysis.json` and in `runs/phase1_4-reanalysis-4b-baseline.json`.
- Per family closures: arith 6/6, order 6/6, sm 6/6, path 2/6, **logic 0/6**.

### Termination-stress model — `Qwen/Qwen3.5-0.8B`

Retained as a stress deployment, not a baseline. Severe censoring persists across
greedy, MLX-window upstream values, generated-history T=1.0, and generated-history
T=0.6. See `PHASE1_3_REPORT.md` and `PHASE1_3B_REPORT.md`.

## Quantization-drift experiment (Phase 1.4)

### Controlled Q4 derived from the identical source revision

- Artifact: `models/qwen3.5-4b-mlx-q4-affine-g64` (gitignored weights; provenance
  `ponderscope_variant.json`). Source `src-6209ef70e318` (same as the BF16
  baseline), derived variant `wvar-b7a30758909a`.
- Actual representation: 248 `QuantizedLinear` modules at affine/4-bit/group-64 =
  **4.503 bits per weight** — *not* a uniformly 4-bit checkpoint.
- Load audit (`runs/model-load-audit-4b-q4.phase1_4.json`): load ok, tokenizer
  semantics identical to the pinned source, native reasoning channel detected,
  load 1.9 s, MLX peak 2.37 GB after load / 2.56 GB after generate.

**Pilot (10 draws, seed 0):** run
`runs/20261008T224538Z-phase1-4-4b-q4-pilot-q4-pilot-dep-4f8e7958df4f`
(`verify` PASS; condition `cond-f5e762d3078f` identical to BF16; presentation and
structural task ids identical). 7/10 native closures, success 0.60, RMST(2048)
1195.2, median 833.

### Full 30-draw primary contrast (MEASURED)

Run `runs/20261008T225839Z-phase1-4-4b-q4-calibration-q4-full-dep-4f8e7958df4f`
(30 stochastic draws, seeds {0,1,2}, 10 tasks, `verify` PASS, publication-grade).
Comparison `…/comparisons/20261008T232827Z-…-sampled.json`: contrast
`weight_representation`, **complete** population (30/30 matched, 0 duplicates),
both seals verified.

Primary outcomes (task-clustered paired bootstrap, 30 matched pairs, 10 clusters):

| outcome | delta (Q4 − BF16) | 95% CI | excludes zero |
|---|---|---|---|
| native reasoning closure rate | +0.033 | [0.00, 0.10] | no |
| RMST to reasoning closure (2048) | −117.3 | [−307.9, +63.5] | no |
| success_at_budget | 0.000 | [−0.133, 0.133] | no |
| EOS termination rate | 0.000 | [−0.133, 0.133] | no |
| censoring rate | −0.033 | [−0.10, 0.00] | no |

No primary interval excludes zero: **the reasoning-behavior effect is not
distinguishable at this sample size.** The practical advantage is large and
unambiguous: 41.7 vs 13.4 tok/s (≈3.1×), 33.0 vs 112.6 s per generation (≈3.4×),
TTFT 286 vs 417 ms, MLX peak 2.56 vs 8.5 GB.

Secondary: reasoning tokens −117.5 [−307.9, +63.4]; repeated-4gram fraction
+0.013 (CI includes 0).

Family reconciliation (n=6/family; native closures vs EOS are **distinct**):

| family | Q4 closures | Q4 EOS | BF16 closures | BF16 EOS |
|---|---|---|---|---|
| arith | 6/6 | 5/6 | 6/6 | 6/6 |
| order | 6/6 | 5/6 | 6/6 | 6/6 |
| sm | 6/6 | 6/6 | 6/6 | 5/6 |
| path | 3/6 | 3/6 | 2/6 | 2/6 |
| logic | 0/6 | 0/6 | 0/6 | 0/6 |
| **total** | **21/30** | **19/30** | **20/30** | **19/30** |

> Correction (Phase 1.6): an earlier draft of this page and of
> `PHASE1_4_REPORT.md` listed Q4 `arith 5/6, order 5/6` as *closures*; those are
> the EOS counts. The native-closure totals are 21/30 (Q4) and 20/30 (BF16). The
> generated report's "Family reconciliation" table carries explicit closure and
> EOS columns; totals reconcile with the aggregate.

**Interpretation:** at n=10 tasks / 30 draws, Q4 does not measurably change
native reasoning closure or time-to-closure, while it is substantially faster and
smaller. The study is underpowered for a small effect; a larger independent-task
replication is required before any generalization.

### Reanalysis / version history

| version | change | effect |
|---|---|---|
| phase1.2 | censored ≠ incorrect; seal ordering | corrected smoke interpretation |
| phase1.3b | closure ≠ EOS; stochastic-draw units | Stage A reanalysis |
| phase1.4 | event time = native think-end index | baseline RMST 1511.9→1368.8, median 1718→1569 |
| phase1.5 | matched-draw paired RMST; independent EOS time; competing events; evidence gating | pilot/full comparisons regenerated |
| phase1.6 | condition-aware draw identity; paired-draw alignment; draw-unit standalone metrics; compact analyses | primary values unchanged; committed analyses no longer embed raw traces |

Historical derived artifacts (`analysis.json`, old comparison files) are
preserved; reanalyses are written as separate `phase1_4`/`phase1_5` artifacts.

## Historical smoke experiment

`runs/20261005T221943Z-smoke-v2-live-smoke-2-dep-8b5f653b14f6` — the Phase 1.1
smoke. Its final-manifest seal **FAILS** verification (confirmed Phase 1.1 sealing
defect) and is preserved as historical, not repaired. Its derived analysis was
corrected in Phase 1.2 from the immutable raw traces; the wrong Phase 1.1
interpretation is preserved alongside as `analysis.phase1_1.json`. Details and
the original tables are in the "Historical smoke detail" section below.

## Corrected historical results

- **Phase 1.2 censoring correction:** a capped generation is a budget failure, not
  an observed wrong answer. `PHASE1_2_CORRECTION.md`.
- **Phase 1.3B Stage A reanalysis:** the old "upstream" condition is
  `qwen-upstream-values-mlx-window20`; at the correct unit it is 60 executions →
  30 stochastic draws → 2 closures → 6.67%. `runs/stage-a-mlx-window20-reanalysis.phase1_3b.json`.
- **Phase 1.4 endpoint correction:** time-to-closure now uses the native think-end
  index, not total generated tokens. `runs/phase1_4-reanalysis-4b-baseline.json`.

## Model-scale experiment (not a deployment-drift causal result)

4B BF16 vs 0.8B generated-history: reasoning-closure-rate delta +0.633
(task-clustered 95% CI [0.367, 0.867]); RMST delta −475.8 tokens
([−700.5, −253.4]). `runs/model-scale-comparison-4b-vs-08b.phase1_3b.json`.

## Pending research questions

1. Does the controlled Q4 change native reasoning closure or time-to-closure on
   the full 10-task calibration population? **MEASURED, not distinguishable**
   (closure +0.033 [0.00, 0.10]; RMST −117.3 [−307.9, +63.5]).
2. Is the change uniform across task families? *(family-level claims are not
   supported by 2 tasks/family.)*
3. Does the effect survive a larger independent-task replication? *(recommended
   next experiment.)*
4. Do early-stopping policies transfer across deployments? *(NOT YET TESTED.)*

---

## Historical smoke detail (Phase 1.2 corrected; preserved)

> Phase 1.3 annotation. The measured numbers below are unchanged. The
> classification language is corrected: the looping observed here is
> **termination stress under the tested greedy thinking condition**, not a
> property of the model under its upstream-recommended sampling policy. See
> `PHASE1_3_REPORT.md`.

### Run identity

- run id: `runs/20261005T221943Z-smoke-v2-live-smoke-2-dep-8b5f653b14f6`
- artifact id: `art-d4247b37d98a`
- deployment id: `dep-8b5f653b14f6`
- model: `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`
- status: `EVIDENCE_COMPLETE`; final-manifest seal **FAILS** (historical defect)
- evidence seal sha256 (manifest): `15d7075b27acee23…`

### Tiny smoke (5 families × 1 task, budget 640)

_Phase 1.2 correction: budget outcomes, censoring, and trajectory states were
regenerated from the immutable raw traces. The Phase 1.1 interpretation was
wrong (see `PHASE1_2_CORRECTION.md`)._

| condition | mode | n | success_at_budget | completion_rate | censored_rate | cond_acc\|completed |
|---|---|---|---|---|---|---|
| cond-19ba9bb0a86d | sampled | 20 | 0.0 | 0.0 | 1.0 | — (no completed answers) |
| cond-481d400047e2 | greedy | 10 | 0.2 | 0.2 | 0.8 | 1.0 |

- greedy replay: token-identical 5/5; same-seed sampled replay: token-identical
  rate 1.0, ambiguous seeds 0.
- prefix states (corrected): `{never_correct: 3, wrong_to_correct: 2}`.

### GO / NO-GO (historical)

> **The Phase 1.1 statement below is historical and was superseded by Phase 1.2.**
> The Phase 1.1 claim that "the task difficulty must be versioned" was made before
> the censoring bug was understood. Phase 1.2 does **not** manufacture an easier
> `tasks-v2`; it treats failure to terminate as a phenomenon to measure.

**CONDITIONAL GO (Phase 1.1, historical).** Software correctness and live
capability gates pass. The current `tasks-v1` pack is censored for 4/5 families
at this budget/model. **Phase 1.2 replaces this recommendation with a termination
calibration experiment** (see `EXPERIMENT_DESIGN.md` and the Phase 1.2 report).
