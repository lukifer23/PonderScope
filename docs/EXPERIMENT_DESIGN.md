# Experiment design

PonderScope uses a staged design. Expensive runs are only launched after smaller
stages are clean. Task packs are versioned (`tasks-v1`); if generation, metrics,
or the protocol change after seeing outcomes, the protocol is versioned and the
reason recorded.

## Task pack (`tasks-v1`)

Five mechanically scored families with exact ground truth and no LLM judge:

| family | what it tests | answer form |
|---|---|---|
| `arith` | multi-step integer arithmetic (precedence via explicit parens) | integer |
| `path` | weighted shortest path on a small undirected graph | integer |
| `order` | unique satisfying total order from precedence constraints | comma list |
| `logic` | unique satisfying boolean assignment (CNF) | true/false list |
| `sm` | final state of a deterministic finite state machine | state name |

Properties: deterministic from seed; structural difficulty metadata; exact
scorers; no answer leakage at an answer cue; task ids independent of wording;
`train`/`calibration`/`dev`/`test` splits available; a `variant` hook for held-out
structural variants. This pack validates methodology and is **not** a universal
intelligence benchmark.

## Stage 0 — capability smoke

Tiny task set (per family, 1–2 tasks). Proves generation works; reasoning/final
channels parse; scorers work; trace persistence works; the deterministic greedy
diagnostic works; sampled seeds behave as expected; reports regenerate from
saved evidence. Actual traces are inspected by hand. Methodological flaws are
fixed before continuing.

## Stage 1 — greedy determinism

One fixed task, ≥5 greedy repeats with distribution digests enabled. Reports
whether nominally identical greedy calls reproduce exactly, and if not, where
divergence begins. This is a **MEASURED** property of this machine/runtime, not
an assumption.

## Stage 2 — noise floor

Per family, several tasks (default 3–8) with:
- greedy repeats (≥3), and
- a frozen sampled profile: **T=0.6, top_p=0.95, top_k=20**, seeds `{0,1,2}`,
  2 repeats per seed.

The exact sample count is justified by the goal of estimating within-condition
variation rather than producing a single anecdotal run. If the greedy condition
is exactly deterministic, the sampled seeds supply the usable noise floor.

## Stage 3 — bounded pilot

If Stage 2 is clean, a bounded pilot across all five families (default
`5 families × 8 tasks`, greedy + 3 seeds). Enough repeated tasks/seeds to
estimate within-condition variation. Results feed `RESULTS.md`.

## Stage 4 — deployment drift (Gate 5, conditional)

Produce one controlled second deployment on the **same machine**: the same
cached upstream revision converted to a reproducible MLX low-bit variant
(record source revision, exact conversion command/API, scheme, bits, group size,
artifact manifest, hashes, tool versions). Then perform paired comparisons on
matched tasks. If clean provenance cannot be kept, Gate 5 stays **PENDING** and
that is documented rather than forcing the comparison.

Paired comparisons report: accuracy delta; reasoning-token delta; wall-time
delta; reasoning-token distribution; termination-failure delta; repetition
delta; answer-transition delta; prefix-sufficiency delta where measured. Effect
sizes are compared to the within-condition noise and classified
`below_noise` / `comparable` / `clearly_larger`. **No causal claims.**

**Phase 1.4 implementation.** The reproducible MLX low-bit conversion now exists
(`ponderscope convert`), records the actual per-module quantization scheme and
full source→derived lineage, and is validated by a real load audit
(`ponderscope variant-audit`). The controlled Q4 artifact is derived from the
identical pinned BF16 source revision, so BF16 and Q4 share one
`source_artifact_id` and differ only in `weight_variant_id`. The 10-draw pilot is
clean; the full 30-draw contrast is the next experiment. Because the primary
design has one execution per stochastic draw, within-deployment noise is not
estimable and uncertainty is the task-clustered bootstrap only.

## Capture settings

- Capture has three explicit levels: **minimal** (token ids + termination only),
  **research** (chosen logprob, entropy, top-k, per-token timing), and
  **digest** (research + full-distribution digest).
- Reasoning **entropy** and **top-5 logprobs** are on by default for research runs.
- Full-distribution **digests** are opt-in (expensive) and used for the
  determinism diagnostic.
- Full logprob vectors are never persisted; only scalars, top-k, and digests.
- Instrumentation overhead is qualified by warmup + alternating repeated runs of
  the three capture levels; measured overhead is reported rather than assumed.
- The `generate_step` logprob vector is the pre-sampler model distribution; top-k
  is reported as the model's most likely tokens, not the post-sampling set.

## Stage 0b — closure discovery (added after the first live pass)

Because a model may need hundreds of reasoning tokens before natural closure,
Stage 0 first runs a bounded closure-discovery sweep (escalating budget) so the
smoke budget is chosen from measurement, not assumption. The versioned tiny
smoke (`specs/smoke-v2.json`) uses the measured budget and records censoring
explicitly. If difficulty is degenerate (ceiling/floor/censored), version the
task pack (`tasks-v2`) rather than silently editing `tasks-v1`.

## Stage 0c — termination calibration (Phase 1.2)

Before changing task difficulty, determine **why** natural termination is late on
trivial tasks. The calibration experiment uses the `calibration` split only, a
generous bounded horizon (initially 2048 tokens), greedy + minimal capture, and
2–3 tasks per family. It measures natural think-end/EOS rate, reasoning tokens
to closure, censoring rate, repetition, and forced-prefix correctness at fixed
checkpoints. Cap-prefix invariance is qualified first (256/512/1024/2048); if
shorter runs are exact prefixes of the longer run, closure and lower-cap
censoring can be derived from one long run.

A prompt-policy A/B (`pp-v1` explicit answer cue vs `pp-v2` minimal neutral
instruction) uses matched structures and exact scoring with no LLM judge. If
wording materially causes long reasoning, a versioned `tasks-v2` is justified;
if not, the finding stands that this deployment naturally deliberates for
hundreds/thousands of tokens on trivial tasks. Curating tasks to hit an arbitrary
closure rate is explicitly not done.

## Stage 0d — decoding validity and censored-termination analysis (Phase 1.3)

A decoding pathology is not a model property until the model has been tested
under a defensible decoding policy. Stage 0d therefore:

1. records the exact upstream generation recommendation for the pinned 0.8B
   revision as a versioned, structured model-policy profile
   (`model_policies/qwen35-08b-thinking-upstream-v1.json`) rather than silently
   promoting upstream values into PonderScope defaults;
2. reproduces that profile's numeric values on MLX-LM through
   `mlx_lm.sample_utils.make_logits_processors` (presence penalty with an
   explicit context size; this MLX-window mapping is labelled
   `qwen-upstream-values-mlx-window20`, not claimed bit-for-bit) and, separately,
   through PonderScope's explicit generated-history presence processor (prompt
   excluded, full generated history; labelled
   `qwen-generated-history-presence-v1`);
3. treats termination as a right-censored time-to-event problem and reports
   Kaplan-Meier closure survival, median tokens-to-closure, and RMST up to an
   explicit observation horizon (never treating the horizon as a natural stop);
4. adds descriptive loop-structure diagnostics with a documented onset rule;
5. enforces presentation/stimulus identity and a strict comparison contract
   (prompt policy, presentation IDs, task-pack version, horizon, capture lane)
   before any deployment contrast is computed.

Only after this stage is a same-family BF16 model-size control (Stage B)
considered. The central deployment-drift experiment (BF16 vs a controlled Q4
derived from the same source revision) remains **NOT YET TESTED**.

## Task-pack registry and prompt policy

`tasks-v1` is defined by an explicit registry entry (generator version, default
prompt policy, structural parameters, provenance) and is frozen: its task ids
and prompts are byte-reproducible. An unknown pack version fails closed rather
than silently reusing another pack's generator semantics. Prompt policy is a
separate, versioned presentation dimension; it changes wording only, never the
structural task identity or the exact answer.

## Reproducibility

Raw evidence is immutable under `runs/<run-id>/`. `analyze`, `compare`, and
`report` never re-run the model. Raw `traces.jsonl` is excluded from Git;
manifests, environments, analyses, and summaries are committed.
