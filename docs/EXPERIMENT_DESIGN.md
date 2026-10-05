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

## Reproducibility

Raw evidence is immutable under `runs/<run-id>/`. `analyze`, `compare`, and
`report` never re-run the model. Raw `traces.jsonl` is excluded from Git;
manifests, environments, analyses, and summaries are committed.
