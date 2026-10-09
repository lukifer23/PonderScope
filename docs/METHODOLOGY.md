# Methodology

PonderScope is measurement software. This document defines what it measures and
how. Terminology is used strictly: **IMPLEMENTED**, **LIVE VALIDATED**,
**MEASURED**, **HYPOTHESIS**, **UNSUPPORTED**, **NOT YET TESTED**.

## The central distinction: model behavior vs deployment behavior

A result observed under one runtime is never described as a property of the
model. Reports identify the deployment as:

> `<repo>@<immutable-revision> (precision, quantization) under <runtime>
> <version> [backend, hardware], <decoding mode>, cfg=<config-id>`

## Deployment identity

Identity is layered, so unrelated provenance can never silently change what is
being measured:

- `source_artifact_id` — upstream repo, immutable revision, source
  weight/tokenizer/chat-template hashes. **Excludes** path, audit, quantization.
- `weight_variant_id` — an actual executable representation: `original` or
  `derived` (quantized/converted), with `variant_weight_files`,
  precision/quantization, conversion tool/version/params, and explicit lineage
  `derived_from_source_artifact_id`. **Excludes** local path and `load_audit`.
- `deployment_id` — weight variant + runtime/version + backend + hardware + OS +
  device; **excludes** decoding.
- `condition_id` — decoding policy (greedy vs sampled; temperature, top-p,
  top-k, min-p, budget, thinking context, and any *active* logits-processor
  penalty with its context size); **excludes** seed. Inactive penalties
  (`None`/`0` additive, `1.0` multiplicative repetition) are normalized away so
  they reproduce prior decoding identity.
- `presentation_id` — structural task id + prompt policy + sha256 of the rendered
  prompt. Structural task ids are wording-independent so the same problem can be
  paired across prompt policies; the presentation id is what actually identifies
  the stimulus, so pp-v1 and pp-v2 cannot collide.
- `trial_id` — presentation id + condition + seed + repeat.

A backend-specific load audit is evidence about how a runtime interpreted an
artifact, not part of the artifact; it can never change `source_artifact_id` or
`weight_variant_id`.

A random seed is trial state, not a deployment property. A run may contain
several decoding conditions and records one condition id per condition; it is
never described by a single decoding-specific id. Canonicalization sorts keys,
drops nulls/empties, rounds floats, and hashes the result.

## Provenance caveat (audited, not asserted)

`Qwen/Qwen3.5-0.8B` is a multimodal checkpoint. PonderScope does **not** merely
assert that `strict=False` drops only vision-tower and MTP weights. A model-load
audit reconstructs the sanitize step, compares source keys to the instantiated
model's parameter keys against an explicit non-text allow-list, and hard-fails if
any text weight is missing or unused. The result is recorded in the run
manifest (`load_audit`).

## Backend contract

A backend executes a `ModelIdentity` and returns a `Trace`. The only real
backend is **MLX / MLX-LM** on Apple Silicon (`backends/mlx_backend.py`). There
are deliberately no placeholder CUDA/llama.cpp/Ollama/vLLM/PyTorch backends.
Future real backends satisfy the same narrow `Backend` protocol.

Capabilities are reported honestly. MLX-LM provides, and PonderScope captures:

- **per-token chosen logprob** (from the normalized logprob vector returned by
  `mlx_lm.generate_step`)
- **Shannon entropy** of the full normalized distribution (nats)
- **top-k** token ids and logprobs
- optional **digest** of the full distribution (for determinism fingerprinting)
- **per-token timing** and **TTFT**
- `finish_reason` (`stop`/`length`/`error`) and EOS/think-end status

Anything MLX-LM does not provide is marked UNSUPPORTED rather than approximated.
The logprob vector from `generate_step` is the **pre-sampler model distribution**
(after logits processors, before temperature/top-p/top-k filtering); top-k is
documented and reported as such, never as the post-sampling distribution.

Capture levels: **minimal** (token ids + termination only; no full-vector cast,
sync, digest, or per-token timing), **research** (chosen logprob, entropy, top-k,
timing), and **digest** (research + full-distribution digest). The **minimal**
lane is the primary performance measurement; research/digest throughput is
instrumentation-affected and never presented as native deployment throughput.
Deterministic greedy and same-seed sampled decoding must be token-identical
across capture lanes; any divergence is reported as an instrumentation effect.
Overhead is qualified by separate warmup and counterbalanced (rotated-order)
repeated runs; median and spread are reported.

## Evidence sealing

`seal()` fully finalizes the manifest, writes it atomically, then hashes the
final manifest and all raw evidence, and writes `evidence.json` **last**. Once the
seal exists, the manifest and raw evidence are immutable. `ponderscope verify`
recomputes every raw hash and the final manifest hash and reports PASS/FAIL
without modifying anything. Official runs refuse a dirty tracked worktree unless
`--allow-dirty` is given, which records an exploratory, non-publication-grade run
with a deterministic hash of the working-tree diff.

## Reasoning channels

The chat template opens the thinking block in the prompt (`<think>\n`). Channels
are split at **token boundaries**: reasoning tokens are those before the first
native think-end token (248069); final tokens are after it. Raw decoded text,
token-level boundaries, and a sanitized semantic final channel are kept distinct,
and decoded special tokens (e.g. `<|im_end|>`) are stripped before scoring so an
EOS marker can never become answer content. PonderScope records:

- **natural termination** — EOS (`<|im_end|>`, token 248046) observed
- **forced-finalization intervention** — the reasoning prefix plus the validated
  native closing sequence `\n</think>\n\n` (every injected token id recorded)
- **capped / censored** — `max_tokens` reached before EOS (no final channel)

Prefix states (over **observed** forced probes only): `initially_correct`,
`wrong_to_correct`, `correct_to_wrong`, `multiple_flips`, `stable_correct`,
`never_correct`. The natural final outcome is a separate category: `correct`,
`incorrect`, `censored` (no final channel observed before the horizon),
`terminated_no_closure` (generation stopped via EOS without ever emitting the
native close), `unparseable`, or `error`.

A censored trajectory has unknown final correctness: it contributes no
correctness transition and cannot be evidence of harmful overthinking. Two
sufficiency concepts are reported separately: `observed_probe_stable_from_tokens`
(earliest probe correct through the last observed probe) and
`stable_sufficient_with_natural_final_tokens` (additionally requires an observed
correct final). `harmful_overthinking_observed` is only True when an **observed**
final is wrong after an earlier correct probe.

Budget outcomes are separate metrics: `success_at_budget`,
`completion_rate`, `conditional_accuracy_given_completed`, `censored_rate`,
`error_rate`, `unparseable_rate`. A capped run is a budget failure, never an
observed wrong answer, and `conditional_accuracy_given_completed` is `None` when
nothing completed.

## Logits processors

Sampling conditions may carry explicit `presence_penalty`, `repetition_penalty`,
and `frequency_penalty` fields with explicit context sizes. They are built with
`mlx_lm.sample_utils.make_logits_processors` (MLX-LM 0.32.0 order:
repetition → presence → frequency) and passed to `generate_step`; the framework,
version, processor order, context sizes, and an honest equivalence label are
recorded per generation. MLX's built-in penalties are an OpenAI-*like*
approximation, so a condition that maps upstream numeric values onto MLX's
20-token prompt-inclusive presence window is labelled
`qwen-upstream-values-mlx-window20`, never claimed bit-for-bit equivalent. The
explicit generated-history presence scope (prompt excluded, full generated
history) is labelled `qwen-generated-history-presence-v1`; the scope is part of
the `DecodingPolicy` condition identity, and the two semantics are never swapped
silently.

## Censor-aware time-to-closure

Termination is a right-censored time-to-event problem with **two distinct
endpoints and two distinct time axes**. The *primary* event is a natural native
think-end/reasoning closure, timed at the think-end token (`reasoning_tokens + 1`)
— never the total generated tokens, which would include the post-closure
final-answer channel. The *secondary* event is generation termination via EOS,
timed independently at the terminal token (`total_tokens`). An observation that
reaches `max_tokens` first is censored at that horizon. The analysis reports
Kaplan-Meier survival (with number at risk, events, and censored), median
time-to-event when estimable, and restricted mean survival time (RMST) up to an
explicit common horizon `tau`. `max_tokens` is an observation horizon, never a
natural stopping threshold. A generation that stops via EOS without ever emitting
the native close is a **competing terminal event** for closure; the primary
closure KM is cause-specific and currently treats it as noninformative censoring
at the terminal token, with the count reported per condition
(`n_competing_terminated_no_closure`) as a documented limitation. The event-time
semantics are versioned `phase1.4`/`phase1.5`; historical definitions are
preserved in earlier derived artifacts.

## Metrics

Token-level, deterministic, no LLM judge:

- reasoning / answer / total token counts (split at the think-end token)
- unique-token ratio; distinct-1/2/4; repeated 4-/8-gram fractions
- longest identical token run (with token id, decoded piece, span, and whether it
  is a special/control token); most-common token count
- most frequent repeated small-n motif; rolling-window unique-token ratio; a
  descriptive degeneration-onset index with a documented rule
- line-level text repetition ratio
- answer extraction success / missing-answer rate
- TTFT, wall time, tokens/sec

## Prefix probes

Sparse forced-finalization probes replay a **reasoning-only** prefix, append the
validated native closing sequence, and read the forced answer. The generated
continuation is parsed as a forced answer continuation, never as a natural trace.
No hand-written answer cue is injected. If the model-native closure cannot be
positively identified and validated, probing **fails closed**
(`UNSUPPORTED`) rather than substituting a cue. **This concept already exists in
prior literature; PonderScope uses it as a measurement primitive, not as its
claimed novelty.** An oracle prefix is never presented as a deployable stopping
method.

## The within-configuration noise floor

A defining feature. Before comparing deployments, three **distinct** quantities
are measured and never collapsed into one standard deviation:

- **Greedy replay variation:** same deterministic condition, repeated; exact
  token identity, first divergence, logprob-digest divergence, answer agreement,
  latency variation.
- **Same-seed sampled replay variation:** same sampler configuration and same
  seed, re-executed; identity/divergence/answer/length/latency.
- **Across-seed stochastic variation:** same policy, different seeds; answer
  diversity and length/accuracy spread.

Distributions are computed for accuracy, reasoning tokens, latency, repetition,
answer flips, and termination failures. Deployment effects are only interpreted
relative to this floor.

## Quantization provenance (derived weight variants)

A derived (quantized) variant is created only by the first-party conversion
workflow, which resolves the exact source revision, hashes the source weights/
tokenizer/template, checks storage, converts, and records the tool version, the
requested parameters, and the **actual** per-module quantization scheme. A
nominal "Q4" artifact is never claimed to be uniformly four-bit unless the loaded
modules confirm it. The derived variant's `weight_variant_id` differs from the
source's while both share the same `source_artifact_id`, and
`derived_from_source_artifact_id` records lineage. Local filesystem paths and the
backend load audit are excluded from scientific identity.

## Frozen task populations

A declared experiment may be frozen with a versioned population lock
(`ponderscope-population-lock/1`) recording the exact task and presentation
identities. When a lock matches a specification's population key, preflight and
the run path verify the freshly generated population against it and **fail
closed** on any mismatch; historical specifications without a lock are
unaffected. A lock change requires a documented preregistration amendment.

## Derived analyses vs raw evidence

Committed `analysis.json` artifacts are compact: counts, aggregates, draw
identities, flags, and per-draw scalar metrics — never raw token arrays, decoded
text, or prompts. Raw traces live in immutable run storage and verified bundles.
Analysis semantics are versioned (`phase1.6`); historical derived artifacts are
preserved rather than rewritten.

## Comparison and statistics

The statistical unit is the **task**, not the generation. Repeated seeds/repeats
from one task are not independent samples, so accuracy CIs and paired deltas use
a **task-clustered** bootstrap that preserves within-task seed/repeat structure;
the estimand is stated explicitly. Paired estimates use only **matched
stochastic draws**: the canonical trial key is
`(presentation_id, task_id, condition_id, mode, seed, repeat)` and the canonical
draw key is `(presentation_id, task_id, condition_id, seed)` — including the
decoding condition so two different conditions never collapse together. Matched
executions collapse to one observation per draw iff token-identical; ambiguous
draws are excluded. Each arm is collapsed independently and then **both arms are
restricted to the intersection of their usable draws**, so a draw ambiguous in
only one arm is excluded from both. Unmatched seeds/repeats and other decoding
conditions cannot enter a paired estimate. A comparison reports its trial
population (expected/observed/matched/unmatched/missing/duplicate per arm) and is
**refused by default when incomplete** (`require_complete`), and refuses when
either run's evidence seal does not verify unless an explicit exploratory
override is given. Standalone per-run primary metrics are draw-level and agree
with the paired code; timing/throughput remain execution-level diagnostics under
`execution_diagnostics`. Every metric's delta is classified
`below_noise` / `comparable` / `clearly_larger` / `ci_only` / `insufficient_data`
using the combined within-deployment noise of **both** compared conditions.

Before any metric is compared, a canonical configuration-difference report lists
identical / changed / missing provenance fields, and the requested contrast is
checked for confounds and **classified by hypothesis** (e.g.
`weight_representation`, `different_source_model`, `runtime_hardware`,
`decoding`, `same_deployment`) rather than by a raw field count. A same-source
precision/quantization change is accepted as one controlled manipulation even
though several model-metadata fields change together; a source-artifact change at
the same repo/revision is refused as inconsistent provenance. The default
contract requires identical structural task
population, identical presentation/stimulus IDs, identical prompt policy,
identical decoding condition, identical task-pack/generator version, a compatible
observation horizon, and a compatible capture lane. A deliberate change to
prompt policy, horizon, capture lane, or decoding requires its explicit
`*_intentional` flag; otherwise the contrast is **refused** so that e.g. pp-v1 vs
pp-v2 cannot masquerade as a clean deployment contrast. Confounded or
under-specified comparisons are refused unless explicitly requested as
exploratory, in which case they are labelled as such. No causal language is used.
Cross-run budget metrics are censor-aware (`success_at_budget`, `completion_rate`,
`censored_rate`, `error_rate`, `answer_observation_rate`, and
`conditional_accuracy_given_completed`, which is `None` when not estimable).

## Policy-transfer contract (future scope, documented now)

Evidence is saved so a later phase can: calibrate a stopping rule on deployment
A; freeze the rule and threshold; evaluate it unchanged on deployment B; measure
accuracy/risk/compute degradation; and compare against recalibration on B.
Candidate baselines: fixed token budget, natural termination, answer stability,
confidence threshold, temporal confidence/stability, a small learned controller.
This is **NOT YET TESTED**.
