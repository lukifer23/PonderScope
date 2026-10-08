# PonderScope Phase 1.4 report — controlled quantization drift (implementation + pilot)

**Status.** Implementation complete; Q4 conversion and a 10-draw pilot are
**MEASURED**. The full 30-draw BF16-vs-Q4 contrast is **PENDING** (gated on
operator approval) and is reported here as **NOT YET TESTED**.

**Classification key.** `IMPLEMENTED`, `LIVE VALIDATED`, `MEASURED`, `HYPOTHESIS`,
`UNSUPPORTED`, `NOT YET TESTED`.

## Starting state

- Starting `main` SHA: `906c8b4590b0b13e3b42b17293558b770c8a474e` (matched
  `origin/main`; clean worktree; CI green).
- Ending `main` SHA at this report: `3518f17ebb51343962248979a7389ebd6a0f4d0d`
  (pushed to `origin/main`). No history rewritten; no feature branch; no Docker;
  no mock model results.
- Baseline gates on the starting SHA: `ruff check`, `ruff format --check`,
  `mypy src`, `pytest` (173 tests), `git diff --check` all pass.

## Commits created (in order)

1. `a9198eb` fix: correct time-to-closure endpoint and split no-closure termination.
2. `e3a7898` fix: structural evidence verification and reanalysis guards.
3. `467c851` feat: explicit contrast classification, clustered RMST, clearer comparison diagnostics.
4. `c6b81ac` feat: reproducible MLX Q4 conversion, derived-variant loading, provenance.
5. `0d854e4` research: controlled Qwen3.5-4B MLX Q4 conversion and real load audit.
6. `58af78e` feat: Phase 1.4 Q4 pilot and primary specs matching the BF16 baseline.
7. `3518f17` research: Phase 1.4 Q4 pilot (10 draws) on the controlled variant.

## Confirmed defects fixed

| # | Defect | Evidence | Fix |
|---|---|---|---|
| F3 | Time-to-closure used `total_tokens` (includes the post-closure answer channel) | 4B baseline RMST 1511.9 / median 1718 from `total_tokens` vs 1368.8 / 1569 from the native think-end index; closed records carry ~216 answer tokens | Event time = `reasoning_tokens + 1`; interpretation versioned `phase1.4`; historical derived evidence preserved |
| F6 | EOS without a native close was conflated with `unparseable` | Code path returned `unparseable` for a stop with no think-end | Distinct `terminated_no_closure` status; 0 occurrences in existing sealed runs, so no historical derived result changed |
| F5 | `verify` hashed bytes only; truncated/duplicate/inconsistent evidence could pass | No structural check existed | `verify_run` now fails closed on malformed JSONL, duplicate trial ids, undeclared conditions, missing condition metadata, and deployment/source/variant mismatch; `analyze` refuses a sealed run with no raw traces |
| F4/F8 | Comparison had no RMST metric, no explicit contrast, and returned `insufficient_data` for every metric when noise was unestimable | Existing 4B-vs-0.8B comparison: all metrics `insufficient_data` | Added clustered censor-aware RMST; `classify_contrast`; `ci_only` classification; clearer CLI diagnostics; report reads `comparisons/` |

## New capabilities

- **Controlled Q4 conversion** (`conversion.py`, `ponderscope convert`):
  create-once output, storage guard, source-identity hashing, and a full
  provenance record. `IMPLEMENTED` + `LIVE VALIDATED`.
- **Derived-variant loading** (`MlxBackend.load(artifact_path=...)`): verifies
  provenance schema, recomputes derived hashes, checks source lineage, and
  validates the actual quantized scheme with a real load audit.
- **`ponderscope variant-audit`**: real load + short generation, semantic
  tokenizer comparison, measured memory. `LIVE VALIDATED`.
- **Explicit contrast classification** and **censor-aware RMST** in comparisons.

## Q4 artifact (LIVE VALIDATED)

- Source: `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
  source artifact `src-6209ef70e318` (identical to the BF16 baseline).
- Artifact: `models/qwen3.5-4b-mlx-q4-affine-g64` (gitignored weights;
  `ponderscope_variant.json` provenance). Derived variant `wvar-b7a30758909a`.
- **Actual representation:** 248 `QuantizedLinear` modules at affine / 4-bit /
  group-64 = **4.503 bits per weight**. It is *not* a uniformly four-bit
  checkpoint; non-linear/embedding weights remain bfloat16.
- Load audit (`runs/model-load-audit-4b-q4.phase1_4.json`, sha256
  `3c4843e272d48a05…`): load ok, 248 quantized modules, tokenizer semantics
  identical to the pinned source (vocab size, think/EOS ids, probe
  encode/decode), native reasoning channel detected, load 1.9 s, MLX peak
  2.37 GB after load / 2.56 GB after generate (BF16 baseline 8.4–8.5 GB).

## Pilot (MEASURED; operational validation, not a powered result)

Run `runs/20261008T224538Z-phase1-4-4b-q4-pilot-q4-pilot-dep-4f8e7958df4f`:
`verify` PASS, publication-grade, condition `cond-f5e762d3078f` identical to the
BF16 baseline, presentation/structural task ids identical. Evidence sha256
`47fa617c41da2e44…`; manifest `a4f697e242fed338…`.

| metric | Q4 pilot (10 draws) | BF16 seed-0 pilot (10 draws) |
|---|---|---|
| native reasoning closures | 7/10 | 7/10 |
| success_at_budget | 0.60 | — |
| conditional accuracy given completion | 1.0 | — |
| RMST(2048) | 1195.2 | — |
| median tokens-to-closure | 833 | — |

Per family (Q4): arith 2/2, order 2/2, sm 2/2, path 1/2, logic 0/2.

Pilot-vs-BF16-seed0 comparison contrast: **`weight_representation`** (clean).
RMST reasoning-closure delta **−202.9 tokens** (95% CI [−422, +9.7], 10 clusters)
— the interval includes zero, so the effect is **not distinguishable** at this
size.

## Corrected baseline reference (MEASURED, derived reanalysis)

`runs/phase1_4-reanalysis-4b-baseline.json` (sha256 `78a9e080a8bcb5b9…`): the
frozen 4B baseline raw traces reanalyzed under `phase1.4`.

| endpoint | RMST(2048) | median tokens-to-closure |
|---|---|---|
| historical (total generated tokens) | 1511.9 | 1718 |
| corrected (native think-end index) | 1368.8 | 1569 |

The on-disk `analysis.json` (phase1.3b) is preserved unchanged; raw evidence is
untouched.

## Tests / quality gates

- `ruff check`, `ruff format --check`, `mypy src`, `git diff --check`: pass.
- `pytest`: **231 tests** pass (58 added in Phase 1.4: termination matrix,
  endpoint correction, analytic KM/RMST fixtures, structural-evidence tamper
  cases, comparison negative matrix, conversion/provenance/lineage, spec-vs-
  baseline match).
- All 8 previously valid sealed runs still `verify` PASS; the historical Phase 1.1
  smoke still FAILs as documented.

## Live experiments actually executed

1. Q4 conversion of the pinned 4B source (~9 s; 2.2 GB output).
2. `variant-audit` real load + 32-token generation (load 1.9 s).
3. 10-draw Q4 pilot (5 m 24 s, 10 generations, exit 0).
4. Pilot-vs-BF16-seed0 comparison.

## Claims supported by the evidence

- A controlled MLX affine Q4 artifact can be produced from the pinned source
  revision with verifiable lineage, and it loads and produces valid native
  reasoning traces. (`MEASURED`)
- The Q4 pilot is operationally clean: matching presentations, matching
  condition id, sealed and verifiable evidence. (`MEASURED`)
- The Q4 pilot's reasoning-closure count (7/10) matches the BF16 seed-0 pilot
  (7/10). (`MEASURED`, single seed, descriptive)

## Claims NOT supported by the evidence

- Any claim that Q4 changes reasoning behavior: **NOT ESTIMABLE / NOT YET TESTED**
  at the full 30-draw level.
- Any family-specific effect: with 2 tasks per family, family claims are not
  supported.
- Any causal or policy-transfer claim: **NOT YET TESTED**.
- The pilot delta does not exclude zero; it is not a "significant" drift.

## Statistical uncertainty

The primary design has one execution per stochastic draw, so within-deployment
noise is **not estimable**; uncertainty is the task-clustered bootstrap only
(`ci_only`). The pilot has 10 clusters and is underpowered by design.

## Remaining limitations

Single machine, single runtime, one model family, small procedural tasks, and
family-level censoring (`logic`). See `docs/LIMITATIONS.md`.

## Documentation changes

New: `docs/ARCHITECTURE.md`, `docs/REPRODUCIBILITY.md`, `docs/QUICKSTART.md`,
`docs/ROADMAP.md`, this report. Rewritten: `README.md` (18 sections),
`docs/RESULTS.md` (current-results index, historical smoke preserved).
`METHODOLOGY.md`/`EXPERIMENT_DESIGN.md`/`LIMITATIONS.md` updated for the phase1.4
endpoint and quantization provenance.

## Recommended next experiment

Run the **full 30-draw BF16-vs-controlled-Q4 contrast**
(`specs/phase1_4-4b-q4-calibration.json`) on the frozen protocol and report
task-clustered CIs and censor-aware RMST. This is the single next experiment.
