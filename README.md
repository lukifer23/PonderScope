# PonderScope

PonderScope is a local-first research instrument for measuring deployment-conditioned reasoning dynamics, reproducibility, and adaptive-compute policy transfer in open reasoning models.

## Research question

To what extent is observed test-time reasoning behavior a property of the model
weights themselves, versus the exact deployment configuration used to execute
them?

A deployment configuration includes model revision, numerical precision,
quantization, inference runtime, hardware, decoding policy, sampling
configuration, and reasoning budget.

A second, later question: do adaptive-compute / early-stopping policies
calibrated on one deployment configuration remain valid when the same nominal
model is deployed under a different precision, quantization, runtime, or
hardware configuration?

PonderScope is **not** an attempt to invent another early-stopping algorithm.
The first priority is to build a trustworthy measurement instrument.

## Status

Work in progress. **Phase 1.3B is complete on `main`** and a primary BF16
baseline now exists. Phase 1.2 corrected measurement semantics that treated a
capped (censored) generation as an observed wrong answer, fixed the evidence-seal
ordering, and split source-/weight-variant identity. Phase 1.3 Stage A measured
the upstream-recommended *numeric* thinking settings. Phase 1.3B corrected the
upstream-policy interpretation, added explicit generated-history presence
semantics, formalized the stochastic-draw/task statistical hierarchy, split
reasoning-closure from EOS termination, fixed run-level deployment descriptions,
and executed the same-family scale control. See `docs/PHASE1_2_REPORT.md`,
`docs/PHASE1_2_CORRECTION.md`, `docs/PHASE1_3_REPORT.md`, and
`docs/PHASE1_3B_REPORT.md`.

Key measured results:

- The old Stage A condition is `qwen-upstream-values-mlx-window20` (upstream
  numeric values mapped through MLX-LM's 20-token prompt-inclusive presence
  window), not a faithful upstream-serving-semantics reproduction. Reanalysed at
  the correct unit, it is 60 executions → 30 stochastic draws → 2.0 closures →
  6.67% completion.
- `Qwen/Qwen3.5-0.8B` is a **termination-stress** deployment under every tested
  policy: greedy, MLX-window upstream values, generated-history T=1.0 (1/30
  closures), and generated-history T=0.6 (0/10). It is retained as a stress
  model.
- `Qwen/Qwen3.5-4B` BF16
  @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` is the **PRIMARY BF16 baseline**:
  native BF16, load audit ok, MLX peak 8.5 GB of 19.3 GB, 20/30 natural
  reasoning closures (66.7%), RMST(2048)=1511.9, median 1718 tokens, and
  token-identical same-seed replay. `logic` remains censored 6/6 at this horizon.
- The central deployment-drift comparison (4B BF16 vs controlled MLX Q4 of the
  same revision) is **NOT YET TESTED** and is the single next experiment.

Terminology is used strictly: `IMPLEMENTED`, `LIVE VALIDATED`, `MEASURED`,
`HYPOTHESIS`, `UNSUPPORTED`, `NOT YET TESTED`.

## Install

Requires macOS on Apple Silicon, Python 3.12, and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run ponderscope doctor
```

## CLI

```bash
ponderscope doctor                       # environment / runtime / model capability
ponderscope generate --pack tasks-v1     # generate and validate a task pack
ponderscope convert --revision <sha> --out models/<artifact> --bits 4   # controlled derived variant
ponderscope variant-audit --path models/<artifact> --revision <sha>     # real load/tokenizer/generation audit
ponderscope run --spec specs/<spec>.json # run a declared deployment/task experiment
ponderscope run --spec specs/<spec>.json --artifact-path models/<artifact>  # run a derived variant
ponderscope analyze --run runs/<id>      # analyze immutable saved evidence
ponderscope verify --run runs/<id>       # verify raw evidence + final manifest seal
ponderscope bundle --run runs/<id> --output <archive.tar.gz>
ponderscope compare --a runs/<id> --b runs/<id>
ponderscope report --run runs/<id>       # regenerate Markdown/HTML from saved evidence
```

## Layout

```
src/ponderscope/    measurement harness
docs/               RELATED_WORK, METHODOLOGY, EXPERIMENT_DESIGN, RESULTS, LIMITATIONS
references.bib      machine-readable verified bibliography
runs/<run-id>/      immutable evidence (manifest, environment, tasks, traces, analysis, evidence seal)
tests/              unit tests; a fake backend exists only inside the test suite
```

Raw model weights and Hugging Face caches are never tracked. Raw traces are
excluded from Git; manifests, environments, analyses, and summaries are kept.
