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
structural task ids identical).

| metric | Q4 pilot | BF16 seed-0 pilot |
|---|---|---|
| native closures | 7/10 | 7/10 |
| success_at_budget | 0.60 | — |
| conditional accuracy | 1.0 | — |
| RMST(2048) | 1195.2 | — |
| median tokens-to-closure | 833 | — |

Pilot-vs-BF16-seed0 comparison classified the contrast as **`weight_representation`**
(clean). RMST reasoning-closure delta −202.9 (95% CI [−422, +9.7], 10 clusters) is
**not distinguishable** at this size. This is an operational validation, not a
powered conclusion.

**Full 30-draw contrast: PENDING.** Not yet run.

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
   the full 10-task calibration population? *(PENDING — full contrast not run.)*
2. Is the change uniform across task families? *(family-level claims are not
   supported by 2 tasks/family.)*
3. Does the effect survive a larger independent-task replication?
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
