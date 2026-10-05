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

Work in progress. **Phase 1.2 (censoring correctness, clean evidence, and
termination calibration) is complete on `main`.** Phase 1.1's live capability
gates pass on `Qwen/Qwen3.5-0.8B`@`2fc06364715b967f1860aea9cf38778875588b17`
(MLX-LM 0.32.0, Apple M3 Pro). Phase 1.2 corrected measurement semantics that
treated a capped (censored) generation as an observed wrong answer, fixed the
evidence-seal ordering, and split source-/weight-variant identity. See
`docs/PHASE1_2_REPORT.md` and `docs/PHASE1_2_CORRECTION.md`.

Measured finding: under a generous uniform 2048-token cap, this deployment
remains in the reasoning channel on 9/10 trivial calibration tasks **for both
prompt policies**, and the censored traces loop (repeated-4-gram fraction
0.61–0.90). Failure to terminate is therefore the phenomenon to measure, not a
task-difficulty problem; **no `tasks-v2` is created**. Natural termination is
classified as a **termination-stress** deployment, and a larger open reasoning
model is recommended as the primary baseline before cross-deployment work.

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
ponderscope run --spec specs/<spec>.json # run a declared deployment/task experiment
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
