# Phase 1.8 protocol — paired reasoning-prefix recoverability (preregistered)

**Status:** preregistered before any Phase 1.8 probe is executed. Classification
key: `PREREGISTERED`, `MEASURED`, `INCONCLUSIVE`, `NOT ESTIMABLE`,
`NOT YET TESTED`. This is an **exploratory, post-outcome** study: it reuses a
population that has already been measured. It is explicitly **not** a fresh
confirmatory replication.

## Research question

> Do the controlled BF16 and Q4 deployments of the exact same `Qwen/Qwen3.5-4B`
> revision differ in **answer recoverability along matched reasoning
> trajectories**, particularly where natural generations failed to terminate
> within 2048 tokens?

This is distinct from final-answer accuracy and from natural termination: a
correct answer may be recoverable from a prefix even when the natural trajectory
never closes.

## Why saved prefixes, not a fresh greedy study (PREREGISTERED)

The primary BF16/Q4 results were obtained from **sampled** trajectories under a
frozen Qwen thinking policy. The directly relevant next question is whether those
**existing** sampled trajectories already contain reasoning prefixes from which a
correct final answer can be recovered. Replaying forced finalization on the exact
saved prefixes answers that question on the measured population; generating new
greedy trajectories would answer a different, unrelated question and is
explicitly rejected here.

## Source (PREREGISTERED)

Frozen Phase 1.6 independent replication, already measured:

- BF16 `runs/20261009T005354Z-phase1-6-replication-bf16-replication-bf16-dep-e1c20569173a`
- Q4 `runs/20261009T025900Z-phase1-6-replication-q4-replication-q4-dep-4f8e7958df4f`

Both: `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; Q4 =
`models/qwen3.5-4b-mlx-q4-affine-g64` (affine 4-bit/group-64, 4.503 bits/weight);
tasks-v1 `test`, `task_seed=1729`, 30 structural tasks, sampled seeds {0,1}, 2048
horizon, generated-history presence semantics (T=1.0, top_p 0.95, top_k 20,
presence 1.5). The source runs must verify (seal + structural) and their raw
`traces.jsonl` must be present; otherwise the run refuses.

## Intervention (PREREGISTERED)

**Option A — greedy forced finalization.** From a saved reasoning-only prefix,
append the model's validated native closing sequence and read the continuation as
the forced final-answer channel. The continuation is a **fresh deterministic
greedy completion**; it is **not** a continuation under the original sampled
decoding policy and is never described as one.

- A replayed prefix is an **intervention on a frozen historical trace**, not
  naturally observed early termination.
- The replayed prefix was previously *generated output*, so it is declared as
  generated history to the logits processors (the true prompt length is passed,
  not `prompt + prefix + closure`).
- Option B (policy-consistent sampled continuation) is **NOT** implemented: the
  original sampler state at a mid-trace checkpoint is unrecoverable, so a
  "continuation" under the original policy cannot be produced honestly.

## Checkpoints (PREREGISTERED)

`0, 256, 512, 1024, 1536`, plus the **full available reasoning prefix** recorded
as a separately labelled condition. A checkpoint is **eligible** for a trajectory
iff the saved reasoning length is at least that many tokens. A shorter
checkpoint is **never** substituted for a requested one; ineligible checkpoints
are marked `NOT ELIGIBLE` and excluded from that checkpoint's denominator. Every
per-checkpoint denominator is reported. Eligibility (seed 0, from the saved
traces):

| checkpoint | BF16 eligible | Q4 eligible | jointly eligible |
|---|---|---|---|
| 0 | 30/30 | 30/30 | 30/30 |
| 256 | 30/30 | 29/30 | 29/30 |
| 512 | 27/30 | 26/30 | 25/30 |
| 1024 | 18/30 | 18/30 | 15/30 |
| 1536 | 17/30 | 14/30 | 13/30 |
| full prefix | 30/30 | 30/30 | 30/30 |

## Probe horizon qualification (PREREGISTERED)

Before the main run, a sentinel subset of the already-explored population is
probed at continuation budgets **64 / 128 / 256**:

- arithmetic `051fe252328d539c`
- ordering `42328719aee13018`
- logic `0860ee8c753bad15`

EOS completion, scorable-answer rate and censoring are measured per budget. One
primary horizon is chosen from this diagnostic, recorded, and frozen. If 256 is
insufficient on an important family, either 512 is justified and recorded or the
affected outcomes are marked `censored`. **A failed probe is never retried until
it becomes correct.**

## Population (PREREGISTERED)

Primary analysis: **30 structural tasks × seed 0 × {BF16, Q4}** — the same frozen
Phase 1.6 population, matched by source task and source seed. No outcome-based
selection of individual tasks for the primary analysis. Seed 1 is an optional,
separately labelled extension only if measured cost is trivial; it is never used
to rescue an interesting first result. The experiment population is not increased
because results look interesting.

## Outcome taxonomy (PREREGISTERED)

Explicit forced-probe outcomes: `correct`, `scorable_incorrect`, `unparseable`,
`censored`, `error`. Fields recorded per probe: `answer_observed`,
`answer_scorable`, `native_close_injected`, `eos_observed`, generated answer
tokens, `finish_reason`, forced answer text, normalized answer, source checkpoint
index. A probe that reaches its finalization budget with no EOS or complete
scorable answer is `censored` and is **never** an observed wrong answer.
Transition logic uses observed, scorable outcomes only; missing and indeterminate
checkpoints are tracked and are never reinterpreted as wrong.

## Estimands (PREREGISTERED)

For every matched structural task and source seed, at identical eligible
checkpoint lengths, compare BF16 vs Q4:

1. paired recoverable-correct probability (primary exploratory);
2. paired scorable-incorrect probability;
3. unparseable / censored / error probe rates;
4. number eligible per checkpoint and arm, and the jointly eligible denominator;
5. first tested recoverable-correct checkpoint;
6. stability of subsequent observed probe correctness;
7. natural completion / censoring status of the source trajectory;
8. relation between forced recoverability and natural termination.

Uncertainty is a task-clustered paired bootstrap where there are enough
independent tasks; no interval is manufactured when the sample is degenerate. A
correct forced answer is **not** evidence of a deployable stopping policy.
Greedy forced-probe results are never mixed with naturally sampled final-answer
accuracy in one metric.

## Censored-source strata (PREREGISTERED, descriptive)

For `logic` and `path`, and for matched pairs where exactly one deployment
naturally closed, report descriptively whether both deployments can be forced to
a correct answer from an earlier checkpoint, only BF16 can, only Q4 can, neither
can, outcomes fluctuate across later checkpoints, or outcomes are too frequently
unparseable/censored to interpret. These are strata selected *after* prior
outcomes and are **not** independent confirmatory populations. The paired
full-population analysis remains primary.

## Evidence and provenance (PREREGISTERED)

New probe evidence is written to its **own** sealed run directory; the source run
is never appended to or mutated. Each probe record preserves source run id,
condition id, stochastic draw id, trial id, seed, source trace hash, and source
evidence-seal hash; each run records its model/weight-variant identity and its
own intervention decoding, capture and budget. Offline reanalysis regenerates the
paired report from saved evidence without rerunning inference. Deterministic
bundles are built for audit. No raw model weights are committed.

## Execution gates (PREREGISTERED)

- **Gate A — scientific integrity:** all correctness fixes and new tests pass;
  source seals verify; historical evidence unchanged.
- **Gate B — replay correctness:** prompt re-tokenization identity, reasoning-only
  prefix extraction, native-closure validation, generated-history boundary
  semantics, and no final-answer/EOS leakage are all verified before the main run.
- **Gate C — operational pilot:** a short pilot on a couple of tasks exercises
  replay, parsing, provenance, eligibility, sealing and runtime.
- **Gate D — horizon decision:** the sentinel chooses and freezes the primary
  horizon.
- **Gate E — main run:** the predeclared 30 task/seed-0 pairs on both arms.
- **Gate F — authorization:** the bounded exploratory run is authorized; any
  change to the population or checkpoints requires a documented amendment, not
  outcome-driven selection.

## Compute (estimate)

Measured Phase 1.6 rates: BF16 ≈0.061 s/token, Q4 ≈0.021 s/token. With ~150
probes per arm, estimated total wall time at horizon 64/128/256 is ≈17 / 29 / 56
minutes. The exact figure is measured during the pilot and recorded.

## What must NOT be done

- Do not regenerate natural trajectories for the primary experiment.
- Do not describe a replayed prefix as naturally observed early termination.
- Do not call greedy forced finalization a continuation under the sampled policy.
- Do not treat `neither` as "both models answered incorrectly".
- Do not mix greedy forced-probe results with natural sampled accuracy.
- Do not reuse test/1729 as a fresh confirmatory population.
- Do not increase the population because the first results look interesting.

## Horizon decision (MEASURED, pre-declared amendment)

Sentinel diagnostic at checkpoint 256 (the predeclared arith / order / logic
tasks, both arms), budgets 64 / 128 / 256 / 512:

| arm | task | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| BF16 | arith `051fe252` | censored | censored | censored | **correct** (372 tokens) |
| BF16 | order `42328719` | correct | correct | correct | correct |
| BF16 | logic `0860ee8c` | censored | censored | censored | **correct** (317 tokens) |
| Q4 | arith `051fe252` | censored | censored | censored | **correct** (290 tokens) |
| Q4 | order `42328719` | censored | censored | correct | correct |
| Q4 | logic `0860ee8c` | censored | censored | censored | censored |

**Decision: primary probe horizon = 512 tokens.** 256 censors the arithmetic and
logic sentinels in BF16 and all three sentinels in Q4; 512 resolves the
arithmetic and ordering cases and reduces censoring materially. Q4 `logic
0860ee8c` remains censored at 512 and is reported as `censored`, never as a wrong
answer. The horizon is frozen; failed probes are never retried until correct.
Diagnostics: `runs/phase1_8-budget-diagnostic-bf16.json`,
`runs/phase1_8-budget-diagnostic-bf16-512.json`,
`runs/phase1_8-budget-diagnostic-q4.json`. A residual censoring rate is expected
and is a reported outcome, not a defect to be optimized away.

