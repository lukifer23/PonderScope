# Phase 1.2 — correctness corrections to Phase 1.1 evidence

This pass re-audited the Phase 1.1 code and the sealed smoke run. It found and
fixed defects in measurement semantics. The sealed smoke is **not rewritten**;
its raw evidence is untouched. Its derived analysis/report were regenerated from
the immutable raw traces and the Phase 1.1 derived files are preserved alongside:

- `analysis.json`, `summary.md`, `summary.html` — corrected (Phase 1.2).
- `analysis.phase1_1.json`, `summary.phase1_1.md`, `summary.phase1_1.html` —
  historical, wrong interpretation, kept for the record.

## 1. Confirmed defect: censoring was scored as incorrectness

`classify_transitions` appended the natural final correctness (derived from
`record["correct"]`, which is `False` for a capped trace with no answer) to the
observed probe sequence. For the `order` task the observed probes were
`[False, False, True, True, True]`, and the capped natural final contributed a
spurious `False`, yielding `multiple_flips`.

There was no observed final incorrect answer. The generated consequence was a
scientifically false `correct→wrong` transition.

**Fix.** Natural final outcome now has explicit categories (correct, incorrect,
censored, unparseable, error). Prefix states/flips are computed over observed
probes only; a censored final contributes no flip and cannot produce
`harmful_overthinking_observed`. Two sufficiency concepts are separated:
`observed_probe_stable_from_tokens` (prefix only) and
`stable_sufficient_with_natural_final_tokens` (requires an observed correct
final).

**Corrected `order` trajectory:** prefix state `wrong_to_correct` (flips=1),
natural final `censored`, harmful overthinking `False`, observed-probe stable
from token 224, natural-final sufficiency `None`. Directional change in state
counts: `{never_correct: 3, wrong_to_correct: 1, multiple_flips: 1}` →
`{never_correct: 3, wrong_to_correct: 2}`.

## 2. Confirmed defect: censored runs scored as accuracy failures / shared answers

The report presented capped runs as accuracy failures, and across-seed analysis
reported `distinct_answers = 1` and `accuracy std = 0.0` when every final answer
was `None` (i.e. treating `None` as one shared answer).

**Fix.** Budget outcomes are now separate metrics: `success_at_budget`,
`completion_rate`, `conditional_accuracy_given_completed` (`None` when no run
completed), `answer_observed_accuracy`, `censored_rate`, `error_rate`,
`unparseable_rate`. Across-seed final-answer diversity/accuracy is reported only
over seeds with an observed answer and is `None`/insufficient otherwise.
Token-level across-seed variation (first divergence, length/repetition spread)
is reported independently and remains valid while censored.

**Corrected sampled interpretation (old budget-640 smoke):** 20/20 sampled
generations censored; `success_at_budget = 0.0`, `completion_rate = 0.0`,
`conditional_accuracy_given_completed = None`. Across-seed final-answer
diversity `None` (unobserved). Token trajectories still differ (mean first
divergence 10.8 tokens). Greedy: 2/10 completed, both correct,
`conditional_accuracy_given_completed = 1.0`, `censored_rate = 0.8`.

## 3. Confirmed defect: evidence seal hashed a pre-final manifest

`RunStore.seal()` hashed the manifest, wrote `evidence.json`, then mutated the
manifest status/timestamp and rewrote it. The seal therefore did not cover the
final manifest.

**Verification of the old run:** `ponderscope verify` on
`runs/20261005T221943Z-smoke-v2-live-smoke-2-dep-8b5f653b14f6` **FAILS**:
`environment.json`, `tasks.jsonl`, `traces.jsonl`, `probes.jsonl` all hash OK,
but `manifest.json` mismatches (`expected=15d7075b…`, `actual=0550c9ad…`). This
is a confirmed Phase 1.1 sealing defect. The run is preserved as historical.

**Fix.** `seal()` now finalizes the manifest, writes it atomically, hashes the
final manifest and raw evidence, and writes `evidence.json` **last**. New runs
verify PASS. `ponderscope verify` and `ponderscope bundle` were added.

## 4. Other corrections

- `load_audit` was included in `ArtifactIdentity.identity_dict()`, so a backend
  audit changed artifact identity. The identity schema was split into
  `SourceArtifactIdentity` and `WeightVariantIdentity`; `load_audit` and
  `local_path` are excluded and `derived_from_source_artifact_id` records
  lineage. Schema version bumped to `ponderscope-identity/3`.
- Manifest condition serialization leaked `seed: 0` for a representative sampled
  condition even though seed is explicitly not part of condition identity; seeds
  now live only in trials.
- The model-load audit's `unexpected_non_text` set was empty by construction;
  the audit now computes dropped-key categories and hard-fails on unexpected
  text-weight loss.
- Capability validation collapsed closure observables into one boolean and
  accepted forced finalization if any single cut worked; both now have explicit
  pass rules.
- Official runs now refuse a dirty tracked worktree unless `--allow-dirty` is
  passed (which records an exploratory, non-publication-grade run).
