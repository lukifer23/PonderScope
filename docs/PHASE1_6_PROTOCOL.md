# Phase 1.6 replication protocol (preregistered)

**Status:** preregistered before any replication outcome is inspected. The task
population and protocol below are frozen; they are not changed after seeing
model outputs. Classification key: `PREREGISTERED`, `MEASURED`, `INCONCLUSIVE`,
`NOT ESTIMABLE`, `NOT YET TESTED`.

## Research question

Does the observed similarity between native BF16 and controlled MLX Q4 reasoning
behavior persist on **previously unseen procedural tasks**? The original Phase
1.4 contrast used 10 calibration tasks; this replication uses a disjoint held-out
population to test generalization beyond that sample.

The study is **not** optimized to manufacture a detectable difference.

## Design (PREREGISTERED)

- **Model:** `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- **Arms:** native BF16 vs controlled MLX affine Q4/group-64 derived from the
  identical revision (`models/qwen3.5-4b-mlx-q4-affine-g64`, 4.503 bits/weight).
- **Task pack:** `tasks-v1`, `split="test"`, `task_seed=1729`.
- **Population:** 5 families × 6 structural tasks = **30 tasks**.
- **Seeds:** `{0, 1}`; one execution per stochastic draw.
- **Per arm:** 30 tasks × 2 seeds = **60 stochastic draws**; **120 generations**
  total.
- **Horizon:** 2048 tokens (an observation horizon, never a natural stop).
- **Decoding:** pp-v1 prompt policy, generated-history presence semantics
  (`qwen-generated-history-presence-v1`), T=1.0, top_p 0.95, top_k 20, min_p 0,
  presence 1.5, repetition 1.0 — identical to the Phase 1.4 protocol.
- **Capture:** minimal. **Runtime/hardware:** MLX-LM 0.32.0 on the same M3 Pro.

Specs: `specs/phase1_6-replication-bf16.json` and
`specs/phase1_6-replication-q4.json` differ only in name/notes; the weight
variant is selected via `--artifact-path`.

## Task-population audit (MEASURED, read-only, pre-outcome)

- 30 tasks generated deterministically; **0 task-id overlap** and **0 structural
  signature overlap** with the Phase 1.4 calibration population
  (`split="calibration"`, `task_seed=0`); collision audit clean.
- Per-family counts 6 each; difficulty easy/medium/hard = 10/10/10; presentation
  ids unique; answers re-derived by the independent per-family invariants.
- Population digests are recorded by `ponderscope preflight` (task_ids and
  presentation_ids sha256). Frozen.

## Estimands and analysis

**Primary (declared before outcomes; interpreted jointly, no post-hoc selection):**
1. Difference in native reasoning-closure probability (Q4 − BF16).
2. Difference in RMST(2048) to native reasoning closure.
3. Difference in correct answers within budget.

**Secondary:** EOS termination rate, censoring rate, reasoning/total token
distributions, latency/throughput/TTFT, repetition and loop diagnostics,
family-specific descriptive results, across-seed variability, answer
disagreement, unparseable/error rates.

- **Analysis population:** matched stochastic draws only (same-seed technical
  repeats collapsed; ambiguous draws excluded). Primary requires a **complete**
  declared population; incomplete/unverified comparisons are not
  publication-grade.
- **Censoring:** right-censored at 2048. A capped run is a budget outcome, never a
  wrong answer.
- **Competing events:** EOS-without-close is reported as a competing terminal
  event; the primary closure KM is cause-specific (documented limitation).
- **Intervals:** task-clustered paired bootstrap (2000 resamples). Within-
  deployment noise is not estimable (one execution per draw), so effects are
  reported as `ci_only`, not noise-floor-calibrated.
- **Zero-event families:** a family with no closure events in either arm
  contributes a delta of 0 with no estimable variation; this is reported as
  `NOT ESTIMABLE`, never as equivalence.
- **Technical errors:** backend errors are reported separately and never scored
  as wrong answers.
- **Missing data:** a missing expected trial fails completeness; it is not
  imputed.
- **Smallest practically meaningful effect:** for closure probability, ~0.10;
  for RMST, ~150 tokens. These are interpretation aids, not equivalence margins.

## Precision planning (design calculation, not from the old effect)

With 30 task clusters the achievable precision is limited. A design simulation
(no model data) gives, for a paired closure-rate difference, a worst-case
(p≈0.5) 95% CI half-width of ≈0.24; for RMST, a half-width of ≈55–145 tokens for
per-task SD 150–400. **30 tasks is likely underpowered for a small effect.** This
is stated plainly: the replication can strengthen or challenge the original
conclusion only for effects larger than this precision, and a null result will
not be reported as equivalence.

## Logic-family limitation

Both deployments were unclosed on all six logic trials at 2048 in Phase 1.4. The
logic family is retained unchanged; family-level claims are descriptive only. A
longer-horizon logic experiment is separate and is **NOT** mixed into this
2048-token primary contrast.

## Compute estimate

BF16 ≈112.6 s/generation × 60 ≈ 1.9 h; Q4 ≈33.0 s/generation × 60 ≈ 0.55 h.
Model compute ≈2.4 h; with loading/verification/analysis ≈2.5–3 h. Disk: raw
traces ~10 MB, analyses ~15 MB. **No model downloads** (source cached; Q4
artifact present).

## Execution gates

- **Gate A — scientific integrity:** all Phase 1.6 correctness fixes pass; family
  totals reconcile; historical evidence unchanged and still verifies.
- **Gate B — preregistration:** population frozen and disjoint; spec-equality
  test passes; compute documented.
- **Gate C — operational pilot:** a dev-subset pilot (5 families × 1 task, seed 0)
  on both arms verifies generation, scoring, sealing, completeness, runtime and
  memory. Pilot outcomes are never used to tune the held-out population.
- **Gate D — authorization:** explicit approval is required before the
  120-generation run.

## Preflight

```bash
uv run ponderscope preflight \
  --spec specs/phase1_6-replication-q4.json \
  --model-repo Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --artifact-path models/qwen3.5-4b-mlx-q4-affine-g64 \
  --per-generation-seconds 33.0 \
  --compare-split calibration --compare-seed 0 --compare-n-per-family 2
```

Checks (no inference): source cache, derived provenance, disk/memory, runtime
versions, frozen population digest, disjointness, expected generation count,
compute estimate, duplicate runs, active/incomplete runs.

## Evidence and reproducibility

Raw traces are excluded from Git. For external reanalysis, build a deterministic
verified bundle with `ponderscope bundle` (it refuses an unverifiable run) and
distribute it through an external artifact location. Large raw bundles are not
uploaded publicly without explicit approval.
