# PonderScope Phase 1.4 report — controlled quantization drift

**Status.** Implementation complete; the controlled Q4 artifact is
**LIVE VALIDATED**; the 10-draw pilot and the **full 30-draw BF16-vs-Q4 contrast
are MEASURED**. A Phase 1.5 pass corrected two statistical defects (matched-draw
paired RMST; independent EOS event time), added evidence-verification gating, and
regenerated the comparisons. The primary reasoning-behavior effect is **NOT
DISTINGUISHABLE** at this sample size; the speed/memory advantage is large.

**Classification key.** `IMPLEMENTED`, `LIVE VALIDATED`, `MEASURED`, `CORRECTED`,
`INCONCLUSIVE`, `NOT ESTIMABLE`, `NOT YET TESTED`.

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

## Full 30-draw primary contrast (MEASURED)

Run `runs/20261008T225839Z-phase1-4-4b-q4-calibration-q4-full-dep-4f8e7958df4f`:
30 stochastic draws, seeds {0,1,2}, 10 tasks, `verify` PASS, publication-grade.
Comparison `comparisons/20261008T232827Z-dep-4f8e7958df4f-vs-dep-e1c20569173a-sampled.json`:
contrast `weight_representation`, **complete** (30/30 matched, 0 duplicates), both
seals verified.

| primary outcome | delta (Q4 − BF16) | 95% CI | excludes zero |
|---|---|---|---|
| native reasoning closure rate | +0.033 | [0.00, 0.10] | no |
| RMST to reasoning closure (2048) | −117.3 | [−307.9, +63.5] | no |
| success_at_budget | 0.000 | [−0.133, 0.133] | no |
| EOS termination rate | 0.000 | [−0.133, 0.133] | no |
| censoring rate | −0.033 | [−0.10, 0.00] | no |

Secondary: reasoning tokens −117.5 [−307.9, +63.4]; total tokens −134.0; wall time
−79.6 s (≈3.4× faster); 41.7 vs 13.4 tok/s (≈3.1×); repeated-4gram +0.013 (CI
includes 0); MLX peak 2.56 vs 8.5 GB. Family closures: arith 5/6 vs 6/6, order
5/6 vs 6/6, sm 6/6 vs 5/6, path 3/6 vs 2/6, logic 0/6 vs 0/6.

**Conclusion:** the reasoning-behavior effect is **not distinguishable** at this
size; the practical speed/memory advantage is large. A larger independent-task
replication is required before generalizing.

## Phase 1.5 corrections (CORRECTED)

| # | defect | evidence | fix |
|---|---|---|---|
| A | paired RMST used unmatched observations (all seeds/repeats/modes for matched tasks) | pilot-vs-full reasoning RMST −175.7 → −202.9 when restricted | canonical trial key (presentation+task+mode+seed+repeat); matched-draw collapse; trial-population accounting; `require_complete` |
| B | EOS termination used reasoning-closure event time | full-run termination RMST −112.2 → −133.4 | independent `_time_to_termination` (terminal token); competing-event count; documented cause-specific estimand |
| — | `analyze`/`compare` did not gate on seal verification | code path | verify-by-default; `--allow-unverified`/`--allow-confounded` produce labelled exploratory results |

Interpretation versioned `phase1.5`. The historical `analysis.json` files and the
previous comparison file are preserved; new comparisons and
`runs/phase1_5-reanalysis-{baseline,pilot}.json` are added alongside.

## Corrected baseline reference (MEASURED, derived reanalysis)

`runs/phase1_5-reanalysis-baseline.json`: the frozen 4B BF16 baseline raw traces
reanalyzed under phase1.5.

| endpoint | RMST(2048) | median tokens |
|---|---|---|
| reasoning closure (native think-end index) | 1368.8 | 1569 |
| generation termination (EOS terminal token) | 1511.9 | 1718 |

The on-disk `analysis.json` (phase1.3b/1.4) is preserved unchanged; raw evidence
is untouched.

## Tests / quality gates

- `ruff check`, `ruff format --check`, `mypy src`, `git diff --check`: pass.
- `pytest`: **249 tests** pass (Phase 1.4 added 58; Phase 1.5 added 17:
  seed-subset invariance, repeat collapse, 30/30 completeness, duplicate/missing
  accounting, event-time matrix, evidence gating).
- All valid sealed runs still `verify` PASS; the historical Phase 1.1 smoke still
  FAILs as documented.

## Live experiments actually executed

1. Q4 conversion of the pinned 4B source (~9 s; 2.2 GB output).
2. `variant-audit` real load + 32-token generation (load 1.9 s).
3. 10-draw Q4 pilot (5 m 24 s).
4. **Full 30-draw Q4 run (16 m 34 s, exit 0, verify PASS).**
5. Corrected pilot and full comparisons (reanalysis only; no model execution).

## Claims supported by the evidence

- A controlled MLX affine Q4 artifact can be produced from the pinned source
  revision with verifiable lineage and loads with valid native reasoning traces.
  (`LIVE VALIDATED`)
- The full Q4 contrast is complete and publication-grade: matching presentations,
  matching condition id, sealed and verified evidence, 30/30 matched pairs.
  (`MEASURED`)
- Q4 is ≈3.1× faster and uses ≈3.3× less peak memory with no measurable change in
  success-at-budget (0.633 both). (`MEASURED`)
- The native reasoning-closure rate (+0.033) and RMST (−117.3) do not exclude zero.
  (`INCONCLUSIVE` / `NOT ESTIMABLE` at this power)

## Claims NOT supported by the evidence

- Any claim that Q4 materially changes reasoning behavior: **NOT DISTINGUISHABLE**
  at this sample size.
- Any family-specific effect: 2 tasks/family is insufficient.
- Any causal or policy-transfer claim: **NOT YET TESTED**.
- Treating the degenerate `conditional_accuracy` CI [0, 0] as proof of exact
  equivalence: all completed answers were correct in both arms, so the contrast
  is uninformative there.

## Statistical uncertainty

One execution per stochastic draw ⇒ within-deployment noise **not estimable**;
uncertainty is the task-clustered bootstrap only (`ci_only`). 10 task clusters and
30 draws are underpowered for a small effect.

## Remaining limitations

Single machine, single runtime, one model family, small procedural tasks,
family-level censoring (`logic` 0/6 in both arms), and competing EOS-without-close
events treated as noninformative censoring for the primary closure KM. See
`docs/LIMITATIONS.md`.

## Documentation changes

New: `docs/ARCHITECTURE.md`, `docs/REPRODUCIBILITY.md`, `docs/QUICKSTART.md`,
`docs/ROADMAP.md`, this report. Rewritten: `README.md` (18 sections),
`docs/RESULTS.md` (current-results index, historical smoke preserved).
`METHODOLOGY.md`/`EXPERIMENT_DESIGN.md`/`LIMITATIONS.md` updated for the phase1.4
endpoint, phase1.5 matched-draw RMST and competing events, and quantization
provenance.

## Recommended next experiment

**A larger independent-task replication of the same BF16/Q4 contrast.** The
primary effect is not distinguishable at 10 tasks / 30 draws while the
speed/memory advantage is large; the scientifically decisive question is now
whether the small reasoning-behavior effect is real, which requires more
independent tasks (not more seeds or repeats). A targeted secondary follow-up is
a longer-horizon sensitivity run for the `logic` family (censored 0/6 in both
arms at 2048). Early-stopping policy transfer remains **NOT YET TESTED** and is
not justified yet.
