# PonderScope

PonderScope is a local-first, reproducible research instrument for measuring
**deployment-conditioned reasoning behavior** in open reasoning models. It asks
whether observed test-time reasoning is a property of the model weights or of
the exact deployment used to execute them — and it refuses to blur the two.

It is not a benchmark, not an intelligence test, and not an early-stopping
algorithm. It is measurement software whose first duty is trustworthy evidence.

The published [MLX Q4 model](https://huggingface.co/Dontbeafed69/Qwen3.5-4B-MLX-4bit-PonderScope)
and [BF16-versus-Q4 research dataset](https://huggingface.co/datasets/Dontbeafed69/PonderScope-Qwen3.5-4B-Quantization-Reasoning)
contain the actual converted weights and all 180 main-study executions, with
verified, sanitized raw evidence. See the [release record](docs/HUGGINGFACE_RELEASE.md)
for hashes, attribution, usability checks, and limits. These are evaluation
artifacts; the evidence does not establish BF16/Q4 equivalence.

## 1. What PonderScope is

A measurement harness that runs a pinned model revision under an explicitly
recorded deployment, generates procedural tasks with exact ground truth, captures
raw traces, seals them immutably, and analyzes reasoning/termination behavior
with censor-aware statistics. Every reported number is derived from saved
evidence; no number is hand-entered.

## 2. Central research questions

1. To what measurable extent is reasoning behavior a property of the model
   weights versus the deployment (precision/quantization, runtime, hardware,
   decoding policy, observation horizon)?
2. Do adaptive-compute / early-stopping policies calibrated on one deployment
   remain valid when the same nominal model is deployed differently?
   *(Documented contract only — NOT YET TESTED.)*

## 3. Why deployment-conditioned reasoning matters

The same checkpoint can be served at different precisions, on different
runtimes, with different samplers. If reasoning length, termination, or accuracy
change under those conditions, then conclusions drawn from one deployment do not
transfer to another. PonderScope makes those differences measurable instead of
assumed.

## 4. What the instrument actually measures

- **Native reasoning closure** — a generation emits the model's native think-end
  token. This is the *primary* endpoint.
- **Generation termination** — the generation stops via EOS. A *secondary*,
  distinct endpoint (an EOS without a native close is not reasoning closure).
- **Right-censoring** — the observation horizon (`max_tokens`) is reached first.
  A capped run is a budget outcome, never an observed wrong answer.
- **Budget outcomes** — `success_at_budget`, `completion_rate`, `censored_rate`,
  `error_rate`, `unparseable_rate`, and conditional accuracy given a completed,
  observed answer.
- **Censor-aware time-to-closure** — Kaplan–Meier closure survival, median
  tokens-to-closure, and RMST up to an explicit horizon. Event time is generated
  tokens *through the native think-end token*.
- **Repeatability / noise** — greedy replay, same-seed sampled replay (with
  ambiguity detection), and across-seed token-level variation, kept separate.
- **Descriptive loop structure** — longest identical-token run, repeated-motif
  counts, rolling-window uniqueness, and a documented degeneration-onset index.
  These are descriptive, not a validated universal loop detector.
- **Prefix probes** — forced finalization from a reasoning-only prefix using the
  model's validated native close. An oracle-style measurement primitive, not a
  deployable policy.

## 5. What it does not measure

- Semantic reasoning quality or correctness beyond exact mechanical scoring.
- Intelligence, general capability, or any universal benchmark score.
- Causal mechanisms. Deployment differences are associations under a protocol.
- Claims about other models, revisions, runtimes, or hardware.
- A deployable early-stopping controller (out of scope for now).

## 6. Current implementation status

- **Runtime:** MLX / MLX-LM on Apple Silicon only. There are deliberately no
  CUDA / llama.cpp / Ollama / vLLM / PyTorch backends.
- **Task pack:** `tasks-v1`, frozen and byte-reproducible, five mechanically
  scored families (`arith`, `path`, `order`, `logic`, `sm`).
- **Evidence:** create-once raw files, atomic writes, a final-manifest seal, and a
  verifier that fails closed on byte-level *and* structural corruption.
- **Deployment identity:** layered `source artifact → weight variant → deployment
  → condition → trial → stochastic draw`.
- **Derived variants:** a first-party `convert` workflow produces a controlled MLX
  quantized artifact with full provenance and a real load audit.
- **Comparison:** explicit contrast classification, task-clustered bootstrap CIs,
  and censor-aware RMST deltas.

## 7. Current status (authoritative)

This section is the single authoritative current-status summary. It keeps three
categories strictly separate: **measured historical results**, **exploratory
Phase 1.8 evidence**, and **unresolved/planned** work.

**Measured historical results (this machine/runtime only):**

- `Qwen/Qwen3.5-4B` BF16 @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` is the
  **primary BF16 baseline** (load audit ok; `logic` censored 6/6 at 2048).
- **Phase 1.4** (10 calibration tasks, 30 draws): the controlled Q4 variant
  (affine 4-bit/group-64, 4.503 bits/weight) is **≈3× faster** and ≈3.3× smaller;
  native reasoning closure +0.033 [0.00, 0.10], RMST(2048) −117.3 [−307.9, +63.5],
  success-at-budget 0.000 — **no primary interval excludes zero**.
- **Phase 1.6** (independent replication, 30 previously unseen `test/1729` tasks,
  60 draws/arm, frozen population lock): closure +0.067 [−0.033, +0.183], RMST
  −48.1 [−171.9, +62.3], success-at-budget +0.050 [−0.017, 0.133]; throughput
  37.0 vs 12.8 tok/s (≈2.9×). **INCONCLUSIVE, not equivalence** — the design is
  underpowered for a ~0.10 closure-rate effect (planning half-width ≈0.165).
- **Phase 1.7** corrected outcome classification (a raw-but-unscorable answer is
  `unparseable`, never wrong) and added offline paired-transition analysis. A
  null result is **not** evidence of equivalence.

**Exploratory Phase 1.8 evidence (post-outcome, NOT confirmatory):** paired
reasoning-prefix recoverability — forced finalization on the *existing* saved
sampled trajectories of the Phase 1.6 population (30 tasks × seed 0 × BF16/Q4) at
identical checkpoints (0/256/512/1024/1536 + full prefix), horizon 512. This asks
whether a correct answer is recoverable from a prefix where natural generation
did not terminate, and whether BF16/Q4 differ. It is an oracle-style intervention
on frozen traces, not a stopping policy. Full protocol:
`docs/PHASE1_8_PROTOCOL.md`; results: `docs/PHASE1_8_REPORT.md`.

**Unresolved / planned:** whether the small closure-rate difference is real at
higher power; whether forced-prefix recoverability motivates a fresh-population
replication, a longer natural horizon, or a first policy-transfer study.
Adaptive-stopping policy transfer remains **NOT YET TESTED**.


## 8. Model / runtime support

- macOS on Apple Silicon; Python 3.12; `mlx-lm==0.32.0`.
- Models are resolved from the local Hugging Face cache at an **immutable
  revision**. PonderScope never silently downloads a different revision.
- Multimodal checkpoints (vision tower / MTP) load with `strict=False`; every
  dropped key is audited and classified, and any unexpected text-weight loss is a
  hard failure.

## 9. Installation requirements

macOS on Apple Silicon, Python 3.12, and [`uv`](https://docs.astral.sh/uv/).
See `docs/QUICKSTART.md`.

```bash
uv sync --frozen --extra dev
uv run ponderscope doctor
```

## 10. Quick-start commands

```bash
ponderscope doctor                       # environment / runtime / model capability
ponderscope generate --pack tasks-v1     # generate and validate a task pack
ponderscope convert --revision <sha> --out models/<artifact> --bits 4
ponderscope variant-audit --path models/<artifact> --revision <sha>
ponderscope preflight --spec specs/<spec>.json   # preconditions, no inference
ponderscope design-precision            # reproducible planning precision (no model data)
ponderscope paired-outcomes --a runs/<id> --b runs/<id>  # offline paired transition analysis
ponderscope run --spec specs/<spec>.json # run a declared deployment/task experiment
ponderscope run --spec specs/<spec>.json --artifact-path models/<artifact>
ponderscope analyze --run runs/<id>      # analyze immutable saved evidence
ponderscope verify --run runs/<id>       # verify raw evidence + final manifest seal
ponderscope bundle --run runs/<id> --output <archive.tar.gz>
ponderscope compare --a runs/<id> --b runs/<id>
ponderscope report --run runs/<id>       # regenerate Markdown/HTML from saved evidence
```

## 11. Experiment workflow

1. Pin the source revision and confirm it is cached.
2. (Optional) `convert` a controlled derived variant; `variant-audit` it.
3. Declare a spec under `specs/` (task pack, split, prompt policy, decoding,
   horizon, seeds).
4. `run` it. Official runs refuse a dirty tracked worktree unless `--allow-dirty`
   (which records an exploratory, non-publication-grade run).
5. `verify`, `analyze`, `report`.
6. `compare` two runs; the contrast is classified and refused if confounded.

See `docs/EXPERIMENT_DESIGN.md`.

## 12. Evidence verification

`ponderscope verify` recomputes every raw-file hash and the final manifest hash,
then applies structural checks (parseable JSONL, unique trial identifiers,
declared conditions, matching deployment/source/variant ids). It is read-only and
fails closed. `ponderscope bundle` refuses to bundle an unverifiable run.

## 13. Reproducing saved analyses

`analyze`, `compare`, and `report` never re-run the model. They read the
immutable evidence under `runs/<id>/`. Raw `traces.jsonl`/`probes.jsonl` are
excluded from Git (they are large and reproducible from the manifest), so
regenerating an analysis requires the local raw evidence; `analyze` refuses when
a sealed run's raw traces are absent rather than emitting an empty analysis.

## 14. Repository structure

```
src/ponderscope/     measurement harness (config, backends, reasoning, tasks,
                     evidence, analysis, conversion)
specs/               versioned experiment specifications
model_policies/      versioned upstream generation-policy profiles
docs/                methodology, design, results, limitations, reports
runs/<run-id>/       immutable evidence + derived analysis (raw traces untracked)
tests/               unit tests; a fake backend exists only inside the test suite
references.bib       machine-readable verified bibliography
```

## 15. Documentation index

- `docs/METHODOLOGY.md` — what is measured and how.
- `docs/EXPERIMENT_DESIGN.md` — staged design and task pack.
- `docs/ARCHITECTURE.md` — identity, lifecycle, evidence, analysis, comparison.
- `docs/REPRODUCIBILITY.md` — hardware/software, caching, conversion, running.
- `docs/QUICKSTART.md` — first commands.
- `docs/RESULTS.md` — current-results index (with historical reports linked).
- [Hugging Face release](docs/HUGGINGFACE_RELEASE.md) — public weights/evidence, exact files and hashes, verification and reuse.
- `docs/LIMITATIONS.md` — scope and caveats.
- `docs/ROADMAP.md` — research questions and next experiments.
- `docs/RELATED_WORK.md`, `references.bib` — literature.
- Phase reports: `PHASE1_2_REPORT.md`, `PHASE1_2_CORRECTION.md`,
  `PHASE1_3_REPORT.md`, `PHASE1_3B_REPORT.md`, `PHASE1_4_REPORT.md`.

## 16. Scientific limitations

Single machine, single runtime, one model family, small procedural tasks, no
causal claims, censoring dominates some families, and small-sample uncertainty.
Full list in `docs/LIMITATIONS.md`.

## 17. Current roadmap

See `docs/ROADMAP.md`. The deployment-drift contrast is **measured but
inconclusive** (Phase 1.4 + Phase 1.6). The current exploratory step is the
bounded **prefix-recoverability** analysis over the existing saved trajectories
(Phase 1.8). Adaptive-stopping policy transfer remains **NOT YET TESTED**.

## 18. Citation and license

MIT (`LICENSE`). If you use PonderScope, cite the repository and the bibliography
in `references.bib`. No DOI is assigned yet.
