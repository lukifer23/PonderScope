# PonderScope Phase 1.2 — final report

Censoring correctness, clean evidence, and termination calibration.

> **Phase 1.3 annotation (annotated, not erased).** All measured facts in this
> report are preserved unchanged. Three interpretive statements were too strong
> and are superseded by Phase 1.3 (see `PHASE1_3_REPORT.md`):
> (1) "prompt wording is not the cause" → the pp-v1 vs pp-v2 answer-format
> ablation did not produce a detectable large termination improvement in this
> 10-task greedy calibration;
> (2) "long reasoning is caused by looping, not task complexity" → the censored
> trajectories are dominated by severe exact repetition and looping occurs even
> on structurally trivial tasks; structural difficulty alone does not explain
> the observed nontermination;
> (3) "termination-stress model" → **termination stress under the tested greedy
> thinking condition**, until upstream-recommended sampling is measured.
> The 0.8B looping observation is also **not novel**: upstream Qwen
> documentation already warns that 0.8B is unusually prone to thinking loops.

## Starting state

- Starting `main` SHA: `688b54d236395c757560792461caac106f7d64c4` (matched
  `origin/main`; worktree clean). No history was rewritten; no feature branch.
- Final `main` SHA: see the commit that adds this file.

## Commits created (in order)

1. `f5413cf` fix: censor-aware measurement, evidence integrity, identity, and
   calibration scaffolding.
2. `b56612d` docs: correct Phase 1.1 interpretation and document Phase 1.2
   contracts.
3. `f4b53a8` research: reanalyze sealed smoke from immutable raw evidence.
4. `534b010` fix: distinguish sanitize renames from text-weight loss in load audit.
5. `559d526` feat: capture-level selector and cap-prefix invariance qualification.
6. `6a1de7f` research: termination calibration (cap-prefix invariance, pp-v1/pp-v2
   at 2048, 4096 follow-up).
7. (this report) docs: Phase 1.2 final report.

## Confirmed bugs

1. **Censoring scored as incorrectness.** `classify_transitions` appended the
   natural final correctness (`False` for a capped trace with no answer) to
   observed probes. The `order` smoke probes `[F,F,T,T,T]` + censored final were
   reported as `multiple_flips`.
2. **Censored runs treated as accuracy failures / shared answers.** Capped runs
   were reported as accuracy failures; across-seed reported `distinct_answers=1`
   and `accuracy_std=0.0` when every answer was `None`.
3. **Seal ordering.** `seal()` hashed the manifest before its final
   status/timestamp mutation, so `evidence.json.manifest_sha256` did not cover
   the final manifest.
4. **Dirty-code official runs.** `git_dirty=true` runs could be produced without
   an explicit override.
5. **Identity conflation.** `load_audit` was part of artifact identity; source
   artifact and weight variant were not separated; manifest conditions leaked
   `seed: 0`.
6. **Model-load audit no-op check.** `unexpected_non_text = [k for k in non_text
   if not is_non_text(k)]` is empty by construction.
7. **Capability pass rules.** Natural closure collapsed four observables into one
   boolean; forced finalization passed if any single cut worked.
8. **Overhead qualification.** Always minimal→research→digest, 3 repeats, no
   spread reported.

## Fixes

- Censor-aware trajectory: explicit `natural_final_status`
  (`correct`/`incorrect`/`censored`/`unparseable`/`error`); prefix states/flips
  over observed probes only; `observed_probe_stable_from_tokens` separated from
  `stable_sufficient_with_natural_final_tokens`; `harmful_overthinking_observed`
  requires an observed wrong final after an earlier correct probe.
- Separate budget outcomes: `success_at_budget`, `completion_rate`,
  `conditional_accuracy_given_completed` (`None` when nothing completed),
  `answer_observed_accuracy`, `censored_rate`, `error_rate`, `unparseable_rate`.
  Across-seed final-answer diversity/accuracy is `None`/insufficient when no
  answer is observed; token-level across-seed variation is reported separately;
  same-seed ambiguity is detected instead of using `rs[0]`.
- `seal()` finalizes the manifest, writes it atomically, hashes the final
  manifest + raw evidence, and writes `evidence.json` last. Added `verify_run`,
  `ponderscope verify`, and a deterministic `ponderscope bundle`.
- Official runs refuse a dirty tracked worktree unless `--allow-dirty`, which
  records `exploratory=true`, `publication_grade=false`, and a
  `working_tree_diff_sha256`.
- Split `SourceArtifactIdentity` / `WeightVariantIdentity` (schema
  `ponderscope-identity/3`); `load_audit` and `local_path` excluded;
  `derived_from_source_artifact_id` lineage; conditions serialize without seed.
- Load audit computes dropped-key categories and hard-fails on unexpected text
  loss, while accepting sanitize renames only when sanitized output equals the
  used parameter set and counts match 1:1.
- Capability: explicit closure observables and an all-required-cuts forced rule.
- Two capture lanes with cross-capture equivalence checks; counterbalanced
  (rotated) overhead measurement, ≥5 repeats, separate warmup, median + spread.
- Versioned pack registry (`tasks-v1` frozen) and versioned prompt policies.

## Tests and quality gates

- `uv sync --frozen --extra dev`: OK.
- `ruff check`: pass. `ruff format --check`: pass. `mypy src`: pass.
- `pytest`: 116 tests pass, including the new Phase 1.2 regressions (censored
  final creates no flip; censored final cannot be harmful overthinking;
  observed-probe stability separated from natural-final sufficiency;
  success_at_budget/completion_rate/conditional accuracy; all-censored sampled
  variance is insufficient; cross-seed token trajectories still differ; same-seed
  ambiguity detected; seal verification and mutation failure; dirty-worktree
  refusal and override metadata; load_audit excluded from identity; condition
  has no seed; source/variant lineage; rename-vs-loss audit; capture equivalence;
  pack registry and unknown-pack fail-closed; tasks-v1 golden; cap-prefix
  invariance; bundle determinism).
- `git diff --check`: clean.

## Old smoke seal verification

**FAILS.** `verify` on `runs/20261005T221943Z-smoke-v2-live-smoke-2-dep-8b5f653b14f6`:
environment/tasks/traces/probes hashes OK, `manifest.json` mismatch
(`expected=15d7075b…`, `actual=0550c9ad…`). This is the confirmed Phase 1.1
sealing defect; the run is preserved, not rewritten. New runs verify PASS.

## Corrected old order trajectory

Prefix `[False, False, True, True, True]`, natural final censored at 640:
`prefix_state=wrong_to_correct` (flips=1), `natural_final_status=censored`,
`harmful_overthinking_observed=False`, `observed_probe_stable_from_tokens=224`,
`stable_sufficient_with_natural_final_tokens=None`. No `multiple_flips`, no
`correct→wrong`.

## Corrected sampled-noise interpretation

Old budget-640 smoke, sampled condition: 20/20 censored. `success_at_budget=0.0`,
`completion_rate=0.0`, `conditional_accuracy_given_completed=None`,
`answer_observed_accuracy=None`. Across-seed final-answer diversity `None`
(unobserved), observed-answer accuracy std `None` (insufficient). Token-level
across-seed variation is still present (mean first divergence 10.8 tokens).
Greedy: 2/10 completed, both correct, `cond_acc|completed=1.0`, censored 0.8.

## Source/variant/deployment identity

`source_artifact_id` (repo + immutable revision + source weight/tokenizer/
template hashes) → `weight_variant_id` (original or derived; variant weight
hashes; precision/quantization; conversion lineage) → `deployment_id` (variant +
runtime/hardware/OS/device) → `condition_id` (decoding, no seed) → `trial_id`
(task + condition + seed + repeat). For this deployment:
`src-52161236e42a`, `wvar-46d3e044d08b`, `dep-d7a370157187`.

## Clean-worktree enforcement

`tracked_dirty` (porcelain, `--untracked-files=no`) is the official-run gate;
untracked evidence output is ignored but counted. Clean run: `publication_grade=
true`, `exploratory=false`. Dirty override: exploratory run with
`working_tree_diff_sha256`. Tests inject `code_state`, so they never depend on the
real repo.

## Corrected model-load audit result

`ok=true`: 488 source keys, 320 text source, 168 non-text (all allow-listed and
dropped), 320 loaded params; 320 text keys renamed by `sanitize` and accounted for
1:1 (`renames_or_drops_accounted_by_count=true`); 0 unexpected drops, 0 text
weights lost. Categories: `model:473`, `mtp:15`.

## Capture equivalence result

`equivalent=true`: greedy and same-seed sampled minimal vs research lanes are
token-identical.

## Instrumentation overhead result

Counterbalanced (rotated) order, 1 warmup + 5 repeats, decode tok/s median:
minimal 73.1, research 56.1 (+23.3%), digest 54.8 (+25.0%). Minimal is the primary
performance lane; research/digest throughput is not native deployment throughput.

## Cap-prefix invariance result

`all_exact_prefix=true` for 256/512/1024/2048 and for 4096: shorter greedy
generations are exact prefixes of longer ones. Closure and lower-cap censoring
can be derived from one generous run.

## Calibration

- Task count: 10 calibration tasks (2 per family) per prompt policy.
- Prompt policies compared: `pp-v1` (explicit `Answer:` cue) and `pp-v2`
  (minimal neutral instruction); matched structures and exact answers.
- Cap: 2048 greedy + minimal capture; plus a bounded 4096 follow-up.

### Natural closure rate by policy / family

| policy | completed | censored | success_at_budget | cond_acc\|completed |
|---|---|---|---|---|
| pp-v1 | 1/10 (arith) | 9/10 | 0.1 | 1.0 |
| pp-v2 | 1/10 (arith) | 9/10 | 0.1 | 1.0 |

Per family: `arith` 1/2 completed; `path`, `order`, `logic`, `sm` 0/2 completed
(censored at 2048). At 4096, `logic`, `order`, `path`, `sm` are still censored
(one 4096 run of each).

### Closure token distribution

Completed `arith` reasoning tokens: 295 (pp-v1), 385 (pp-v2). Censored
trajectories ran the full cap (2048 / 4096).

### Censoring rate

0.9 for both policies at 2048. Error/unparseable rates 0.

### Repetition behavior

Censored traces show degenerate repetition: `unique_token_ratio` 0.04–0.12 and
`repeated_ngram_fraction_4` 0.61–0.90; one pp-v2 run had `max_token_run=1201`
(a single token repeated 1201 times). Completed arith traces were much healthier
(`unique_token_ratio` 0.32, `repeated_ngram_fraction_4` 0.34–0.35). The censored
trajectories are dominated by severe exact repetition, and looping occurs even on
structurally trivial tasks; structural difficulty alone does not explain the
observed nontermination. (Phase 1.3 correction; the original wording said "long
reasoning is caused by looping, not task complexity".)

### Forced-prefix observations

`order` became forced-correct from 693 tokens; one `arith` from 16 and another
from 109; `path`, `logic`, `sm` were never forced-correct at any observed prefix.
No `harmful_overthinking_observed` in any trajectory.

### Does prompt wording explain the long reasoning?

The pp-v1 vs pp-v2 answer-format ablation **did not produce a detectable large
termination improvement** in this 10-task greedy calibration: both completed
1/10 and censored 0.9, with completion at 295 vs 385 reasoning tokens. This does
not establish that prompt wording is irrelevant in general, only that this
answer-format change did not rescue termination here. Therefore no `tasks-v2` is
justified on this evidence, and `tasks-v1` is unchanged. (Phase 1.3 correction;
the original wording said the explicit `Answer:` cue "is not the cause".)

## Recommended uniform research cap

**2048 tokens.** Most ordinary trajectories do not complete at any reasonable cap
because they loop; the completed arith trajectory finished at 295–385. A uniform
2048 binds compute while preserving genuine nontermination as censoring. The 4096
follow-up showed no recovery, so the cap is not raised further.

## Model classification

**Termination stress under the tested greedy thinking condition.** Under the
greedy, no-penalty thinking condition, `Qwen/Qwen3.5-0.8B`@`2fc06364…` exhibits
highly repetitive long reasoning on trivial tasks and rarely closes naturally. It
is retained because that behavior is scientifically useful. This is **not** a
claim about the model under its upstream-recommended sampling policy, which
Phase 1.3 measures; upstream already warns that 0.8B is unusually prone to
thinking loops. **Recommendation:** if the upstream-recommended condition still
fails to terminate, use one larger same-family model as the primary baseline for
future cross-deployment study.

## Evidence bundle

`ponderscope bundle` on the pp-v1 calibration run produced
`calibration-pp1.tar.gz` (10 members: manifest, seal, environment, tasks, traces,
probes, analysis, summary.md/html, spec.json), deterministic `tar.gz`,
SHA-256 `37975ca2b4226fc289df1ec5921036c59c866f361c1864758ceab3e6bac9635d`,
145,650 bytes. The archive is **not** committed (raw traces stay out of Git) and
is not uploaded.

## Research-gap status

The related-work gap (deployment-conditioned reasoning reproducibility and
stopping-policy transfer) **remains open but narrowed**: Phase 1.2 establishes a
correct instrument and a baseline deployment that is a termination-stress case.
Cross-deployment comparison (e.g. BF16 vs Q4) is still NOT YET TESTED and now
needs a primary baseline model before it is meaningful.

## Decision

**CONDITIONAL GO.** The measurement instrument and evidence integrity now pass
(censor-aware semantics, seal verification, clean-worktree gates, capture
equivalence, load audit). Solid progress requires a primary reasoning model, since
the current deployment mostly does not terminate.

## Exactly ONE next experiment

Add one larger open reasoning model (a quantized 7–14B reasoning model that fits
the M3 Pro) as a second deployment and run the **same** termination-calibration
protocol (calibration split, greedy + minimal, 2048 cap, pp-v1) to establish
whether natural termination is deployment-specific or model-specific. Do not run
the Q4 comparison, the large pilot, or a learned controller in this phase.
