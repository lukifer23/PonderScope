# PonderScope Phase 1.3 — decoding validity, censored-termination analysis, and same-family baseline qualification

**Principle.** A decoding pathology is not a model property until the model has
been tested under a defensible decoding policy.

## Starting state

- Starting `main` SHA: `a9018ba46d7ce708f028050a2f5a3e1859117fa1` (matched
  `origin/main`; worktree clean). No history rewritten, no feature branch.
- Final `main` SHA: see the commit that adds this file.
- CI on the starting SHA: green. CI on the Phase 1.3 code commit (`7bb9852`):
  green (`gh run 37392606925`, 45s).

## Commits created (in order)

1. `7bb9852` feat: Phase 1.3 decoding validity, presentation identity,
   censor-aware analysis (code + tests + docs corrections + policy + specs).
2. (this report + research evidence) docs/research: Phase 1.3 Stage A.

## Audit: identity/comparison defects fixed

| # | Defect | Fix |
|---|---|---|
| 1 | `DecodingPolicy` had no logits-processor controls, so the upstream recommendation could not be represented | Added `presence_penalty`/`presence_context_size`, `repetition_penalty`/`repetition_context_size`, `frequency_penalty`/`frequency_context_size`, all participating in `condition_id` when *active*; inactive penalties normalize away so zero-penalty decoding keeps prior identity |
| 2 | `TrialIdentity` keyed only on structural `task_id`, so pp-v1 and pp-v2 could share a `trial_id` | Added explicit `presentation_id` (structural task id + prompt policy + sha256 of rendered prompt); `TrialIdentity` and `trial_id` now include it |
| 3 | `compare_configs` did not protect against prompt/stimulus/horizon/capture/pack mismatches | Strict default contract + explicit `prompt_policy_intentional` / `horizon_intentional` / `capture_intentional` overrides |
| 4 | Cross-run metric `accuracy` behaved as success-at-budget and could read censored trials as 0 | Renamed to `success_at_budget`; added `completion_rate`, `censored_rate`, `error_rate`, `answer_observation_rate`, `conditional_accuracy_given_completed` (`None` when not estimable) |
| 5 | `prefix_invariance` returned `all_exact_prefix=true` for a single cap with zero comparisons | `<2` distinct caps → `status="insufficient_data"`, `exact_prefix=None`; reports `n_comparisons` |
| 6 | Loop analysis was limited to aggregate repetition fractions | Added descriptive loop diagnostics (longest run + decoded piece + span, top repeated motif, rolling-window unique ratio, documented degeneration onset) |

## Upstream 0.8B policy provenance

Authoritative source: Hugging Face model card at the **pinned immutable
revision** `2fc06364715b967f1860aea9cf38778875588b17`
(`https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/2fc06364715b967f1860aea9cf38778875588b17/README.md`).

- README sha256: `87a163af54f32fa608a0f8d3ac67945c53dd2b4a7c96740b3d7fdc28e8458864`
  (61,705 bytes); retrieved `2026-10-05T23:56:55Z`.
- TEXT THINKING profile (verified, structurally recorded):
  `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,
  repetition_penalty=1.0, enable_thinking=true`.
- Upstream warning (verbatim): "In thinking mode, we have observed that when
  using the recommended sampling parameters, Qwen3.5-0.8B is more prone to
  entering thinking loops compared to other Qwen3.5 models, which may prevent it
  from terminating generation properly."
- Recorded as `model_policies/qwen35-08b-thinking-upstream-v1.json` and
  referenced by `specs/calibration-08b-upstream-thinking.json`; it is **not**
  promoted into generic PonderScope defaults.

## MLX penalty semantics / context sizes

- Framework: `mlx-lm==0.32.0`, `mlx==0.32.3`.
- Built with `mlx_lm.sample_utils.make_logits_processors`; processor order
  `repetition → presence → frequency`; `presence_context_size=20`,
  `repetition_context_size=20` (context sizes not specified upstream; recorded as
  explicit choices). Passed to `generate_step(logits_processors=...)`.
- MLX presence penalty is OpenAI-*like* (subtract a constant from logits of
  tokens seen in the last context window); this is labelled
  **`qwen-upstream-profile-on-mlx`**, not claimed bit-for-bit equivalent to any
  OpenAI-compatible server. Recorded per generation in `trace.extra`.

## Corrected 2048↔4096 prefix qualification

`runs/calibration-prefix-2048-4096.phase1_3.json` (sha256
`cfe76fc5e231c61f9c027d659310876de69befb70ecd7abe60563ba090d76eca`):
`caps=[2048,4096]`, `n_comparisons=5`, `status=ok`, `all_exact_prefix=true`.
Each 2048 greedy output is an exact prefix of its matched 4096 output
(`arith` 300/300; `logic`, `order`, `path`, `sm` 2048/4096). The Phase 1.2
4096-only artifact (`checks=[]`) was vacuous and is preserved unchanged; the
corrected semantics now reject a single-cap "validation".

## Historical 1201-token run — decoded loop token

From the preserved pp-v2 calibration trace (`arith c50445f1`, untracked raw
evidence regenerated locally): longest identical-token run = **1,201** tokens of
token id **318**, decoded **`" ("`** (space + open parenthesis), spanning token
indices **847–2047** (to the horizon), in ordinary text context
(`"...The structure is:\n    ( ( ( ..."`). It is **ordinary whitespace +
punctuation**, not a special/control/template token, so no runtime artifact is
implicated. The new loop diagnostics report this class of structure directly.

## Stage A — 0.8B upstream-recommended thinking calibration

Run: `runs/20261006T001705Z-calibration-08b-upstream-thinking-dep-d7a370157187`
(deployment `dep-d7a370157187`, source `src-52161236e42a`, condition
`cond-74a29a115886`; `qwen-upstream-profile-on-mlx`).
Evidence: `evidence.json` sha256 `23e6dae3…`, `manifest.json` `3c1fcbb5…`,
`traces.jsonl` `7f107ade…`; `ponderscope verify` = **PASS**.
Design: tasks-v1 calibration, pp-v1, 10 structural tasks (2/family), 2048
horizon, minimal capture, seeds {0,1,2} × 2 repeats = 60 generations. Greedy was
**not** the primary condition.

### Budget outcomes (60 generations)

| metric | value |
|---|---|
| success_at_budget | 0.067 (4/60) |
| completion_rate | 0.067 |
| censored_rate | 0.933 |
| error_rate | 0.0 |
| answer_observation_rate | 0.067 |
| conditional_accuracy_given_completed | 1.0 |
| reasoning tokens mean / median | 1951.3 / 2048 |

Per family: `arith` 4/12 completed (both repeats of seeds 0 for both arith tasks;
seeds 1–2 censored); `logic`, `order`, `path`, `sm` 0/12 completed (all censored
at 2048). Completed arith reasoning lengths: 280 and 915 tokens.

### Censor-aware survival / RMST

| scope | n | events | censored | median tokens | RMST(2048) |
|---|---|---|---|---|---|
| all | 60 | 4 | 56 | not estimable | 1956.5 |
| arith | 12 | 4 | 8 | not estimable | 1590.7 |
| logic/order/path/sm | 12 each | 0 | 12 each | not estimable | 2048.0 each |

Only 4 closure events exist, so uncertainty is enormous and median is not
estimable. `max_tokens=2048` is an observation horizon, not a stopping threshold.

### Repetition / loop structure

- repeated-4-gram fraction: mean 0.468 (p10 0.331, p90 0.643, max 0.729) —
  lower than the historical greedy censored traces (0.61–0.90).
- unique-token ratio: mean 0.141.
- longest identical-token run: max **3** (vs **1,201** historically): the
  presence penalty eliminated the extreme single-token run.
- dominant repeated motif (n=4) count: mean 25.8, max 72 — repetition now
  manifests as repeated multi-token motifs, not one repeated token.
- 4 generations have a sustained degeneration onset; 0 longest runs are special.

### Cross-seed and same-seed behavior

- Same-seed replay: 30 task/condition/seed groups, token-identical rate **1.0**,
  0 ambiguous seeds, answer agreement 1.0 → deterministic re-execution under the
  new processors.
- Across-seed: mean first token divergence **6.2** tokens; mean reasoning-token
  std across seeds **136.8** → stochastic variation is present.

### Does official sampling materially change the termination conclusion?

**No.** Completion is 4/60 (6.7%) under the upstream-recommended condition versus
1/10 (10%) under the historical greedy condition (different designs; descriptive
reference only). Censoring remains 93.3%, four of five families never close, and
repetition remains severe. The recommended condition removes the extreme
single-token run but does not restore termination.

### Stage A decision

`Qwen3.5-0.8B` thinking mode is classified as a **termination-stress deployment
under both tested conditions** (greedy and upstream-recommended sampling),
consistent with the upstream warning. Per the Phase 1.3 protocol this warrants a
same-family BF16 model-size control (Stage B).

## Small research-capture loop diagnostic

Run: `runs/20261006T004654Z-loop-diagnostic-dep-d7a370157187` (evidence
`acbc1d02…`; verify PASS). Predeclared 3 tasks (arith 7566c88a terminating;
order 061fb724 and sm 39578687 looping), seed 0, research capture, 2048 horizon.

- Capture equivalence under the penalty processors
  (`runs/loop-diagnostic-capture-equivalence.phase1_3.json`, sha256
  `fb6d8d19…`): minimal vs research lanes are **token-identical** for greedy and
  upstream-sampled on all 3 prompts (`equivalent=true`).
- Entropy / chosen logprob (research capture):
  - `arith` (terminating, correct): entropy mean 0.323, median 0.014; chosen
    logprob median 0.0 — highly confident.
  - `order` (looping): entropy mean 0.741, median 0.346, last-200 mean 0.624;
    chosen logprob median −0.125 — the loop remains comparatively stochastic.
  - `sm` (looping): entropy mean 0.573, median 0.036, last-200 mean 0.275;
    chosen logprob median 0.0 — the loop coincides with a late confidence
    collapse.

**Descriptive answer:** the degenerate loop is *not* universally a
collapsed-entropy phenomenon. For `sm` it coincides with low entropy / extreme
confidence in the tail; for `order` it remains comparatively stochastic. This is
a descriptive qualification, not a universal conclusion.

## Stage B — same-family BF16 control (decision)

Stage A still shows severe looping, so a Stage B control is warranted:
`Qwen/Qwen3.5-4B` native BF16 (same Qwen3.5 family; not singled out by the 0.8B
loop warning; ~9.34 GB, may fit the 18 GB M3 Pro), falling back to official
`Qwen/Qwen3.5-2B` BF16 if 4B cannot run safely. **Not executed in this commit**:
per the agreed protocol the ~9 GB download is paused for explicit confirmation.

## Tests / quality gates

- `uv sync --frozen --extra dev`: OK.
- `ruff check` / `ruff format --check` / `mypy src`: pass.
- `pytest`: **148 passed** (32 new Phase 1.3 tests: presentation identity;
  penalty/context condition identity; zero/None penalty preservation; logits
  processors built, passed, and logit-affecting; model-policy serialization and
  provenance; prefix-invariance ≥2-cap semantics; Kaplan-Meier no/all/mixed/tied
  censoring; RMST and event-at-horizon; clustered RMST delta; loop diagnostics +
  special-token reporting; comparison refusals and explicit overrides;
  censor-aware metric names; capture equivalence under penalty processors).
- `git diff --check`: clean.

## Research-gap status

The central deployment-drift question (same source model, native BF16 vs
controlled Q4) remains **NOT YET TESTED**, and now requires a primary baseline.
Phase 1.3 narrows the gap: the instrument now measures decoding-validity,
censor-aware closure survival, and loop structure, and establishes that the
0.8B's nontermination is not merely an artifact of greedy/no-penalty decoding.

## Decision

**CONDITIONAL GO.** The instrument and comparison contracts are corrected and
trustworthy. The 0.8B remains a scientifically useful stress deployment, not a
primary baseline. A same-family BF16 control is required before cross-deployment
work.

## Exactly ONE next experiment

Run the **same-family BF16 control**: load `Qwen/Qwen3.5-4B` (official native
BF16; fall back to official `Qwen/Qwen3.5-2B` BF16 only if 4B cannot run safely
on the 18 GB M3 Pro), resolve an immutable revision, verify tokenizer/template
compatibility, dry-load with a load audit and memory record, then run the
**same** Stage A protocol (tasks-v1 calibration, pp-v1, 2048 horizon, minimal
capture, the model-appropriate upstream-equivalent thinking policy) starting with
one seed. This is a model/scale control, not yet a deployment-drift result; do
**not** run the BF16→Q4 comparison, a learned controller, or the large pilot in
this phase.
