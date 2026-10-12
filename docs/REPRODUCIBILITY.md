# Reproducibility guide

Real commands, matching the current CLI. Nothing here is aspirational.

## Hardware and software

- macOS on Apple Silicon (validated on an M3 Pro, 19.3 GB unified memory).
- Python 3.12 (`uv`-managed); `mlx-lm==0.32.0`, `mlx==0.32.3` (pinned by
  `uv.lock`).
- Storage: the 4B source snapshot is ~9.3 GB; a derived Q4 artifact is ~2.2 GB.
  The converter checks free space before writing.

## Environment verification

```bash
uv sync --frozen --extra dev
uv run ponderscope doctor            # environment + model load (no generation)
uv run ponderscope doctor --live     # adds a live capability proof
```

## Model download / caching expectations

PonderScope resolves models from the local Hugging Face cache at an **immutable
revision** and never silently downloads a different revision. Populate the cache
out of band, e.g.:

```bash
huggingface-cli download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

Exact revisions used:

- `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`

## Q4 conversion steps

```bash
# Dry run: resolve the source and check storage without converting.
uv run ponderscope convert \
  --source-repo Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --out models/qwen3.5-4b-mlx-q4-affine-g64 \
  --bits 4 --group-size 64 --mode affine --dtype bfloat16 --dry-run

# Real conversion (create-once; refuses to overwrite).
uv run ponderscope convert \
  --source-repo Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --out models/qwen3.5-4b-mlx-q4-affine-g64 \
  --bits 4 --group-size 64 --mode affine --dtype bfloat16

# Validate the derived artifact with a real load + short generation.
uv run ponderscope variant-audit \
  --path models/qwen3.5-4b-mlx-q4-affine-g64 \
  --source-repo Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --out runs/model-load-audit-4b-q4.phase1_4.json
```

The actual quantization scheme is recorded, not assumed: the 4B artifact is
4.503 bits per weight over 248 affine-4bit/group-64 modules, not uniformly 4-bit.

## Running a pilot / the full experiment

```bash
# 10-draw pilot.
uv run ponderscope run \
  --spec specs/phase1_4-4b-q4-pilot.json \
  --artifact-path models/qwen3.5-4b-mlx-q4-affine-g64 \
  --model-repo Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --suffix q4-pilot

# 30-draw primary contrast.
uv run ponderscope run \
  --spec specs/phase1_4-4b-q4-calibration.json \
  --artifact-path models/qwen3.5-4b-mlx-q4-affine-g64 \
  --model-repo Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

Official runs refuse a dirty tracked worktree unless `--allow-dirty` is passed,
which records an exploratory, non-publication-grade run.

## Verify evidence

```bash
uv run ponderscope verify --run runs/<id>
uv run ponderscope verify --run runs/<id> --json
```

`verify` recomputes every raw hash and the final manifest hash, then applies
structural checks (parseable JSONL, unique trial ids, declared conditions,
matching deployment/source/variant ids). It fails closed and never repairs.

## Regenerate reports and compare deployments

```bash
uv run ponderscope analyze --run runs/<id> --report
uv run ponderscope report  --run runs/<id>
uv run ponderscope compare --a runs/<id> --b runs/<id> --mode sampled
uv run ponderscope compare --a runs/<id> --b runs/<id> --mode sampled --require-complete
```

### Saved-prefix forced-finalization replay (Phase 1.8)

```bash
# Forced probes over the saved reasoning prefixes of a sealed generation run.
# Add --artifact-path for the controlled Q4 arm; omit it for BF16.
uv run ponderscope probe-existing \
  --source-run runs/<source-run> --label phase1-8-probe-bf16 \
  --checkpoints 0 256 512 1024 1536 --probe-max-tokens 512 --seeds 0 \
  --model-repo Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

# Offline paired analysis over the two sealed probe runs (no inference).
uv run ponderscope recoverability \
  --bf16-probe runs/<bf16-probe> --q4-probe runs/<q4-probe> \
  --out runs/phase1_8-recoverability.json \
  --md docs/PHASE1_8_RECOVERABILITY.md --html docs/PHASE1_8_RECOVERABILITY.html
```

A probe run is create-once, sealed and verifiable like any other run; it has no
`traces.jsonl` (it produces no natural generations), and `verify` accepts that.
Raw `probes.jsonl` is gitignored exactly like `traces.jsonl`, so full raw probe
evidence is preserved locally and in deterministic bundles rather than committed.
Status of the current Phase 1.8 pass (BF16 arm complete, Q4 arm pending) is in
`runs/phase1_8-status.json`.

`compare` writes `runs/<a>/comparisons/<timestamp>-…json` and prints the contrast
classification and trial population; a confounded, unverified, or incomplete
contrast is refused with reasons. `analyze` and `compare` verify the evidence seal
by default and refuse an unverifiable run; pass `--allow-unverified` (and
`--allow-confounded` for a confounded contrast) only for a labelled
exploratory/forensic analysis. `--require-complete` refuses a comparison whose
matched-trial population is incomplete.

## Known sources of nondeterminism

- Greedy determinism is a same-process, same-machine replay property; it does not
  prove cross-process/machine determinism.
- Same-seed sampled replay is expected to be token-identical under the fixed
  processors; divergent repeats are flagged ambiguous and excluded from primary
  summaries.
- Across-seed variation is real stochastic variation and is reported separately.
- Per-token timing includes capture overhead; the minimal capture lane is the
  primary performance lane.

## Identifying incomplete or invalid runs

- `status.run != "EVIDENCE_COMPLETE"` (e.g. `PARTIAL`, `FAILED`) means no seal.
- `verify` FAIL means the raw evidence or final manifest is inconsistent.
- A sealed run with no local raw traces cannot be analyzed (analyze refuses).

## Frozen populations and compact analyses

A declared experiment may be frozen with `specs/<...>-population-lock.json`.
`preflight` and `run` auto-detect a lock by the specification's population key
and fail closed on any mismatch; `--population-lock` selects one explicitly.
Committed `analysis.json` files are compact summaries (no raw token arrays, text,
or prompts); raw evidence is only in the run directory and verified bundles.
Raw traces are **not guaranteed byte-reproducible** by re-running with the same
seed — GPU nondeterminism and library changes can alter them — so the immutable
run evidence and bundles are the archival record, not a re-run.

## Evidence bundles (external reproducibility)

Raw traces are excluded from Git, so a sealed run's committed hashes and
summaries do not by themselves let an external researcher regenerate every
analysis. Build a deterministic, verified bundle instead:

```bash
uv run ponderscope bundle --run runs/<id> --output <path>.tar.gz
```

`bundle` refuses to bundle a run whose evidence seal does not verify, includes
the final manifest, seal, environment, tasks, raw traces/probes, derived
analysis/summaries, and the exact spec, and is byte-deterministic (sorted
members, zeroed metadata, gzip mtime=0). A local bundle of the Phase 1.4 full
Q4 run was produced at
`models/bundles/q4-full-20261008T225839Z-phase1-4-4b-q4-calibration-q4-full-dep-4f8e7958df4f.tar.gz`
(sha256 `a9327e3d119fd561…`, 985,486 bytes, 10 members). The approved 2026-10-12 [Hugging Face release](HUGGINGFACE_RELEASE.md) publishes
all four main-study bundles as explicitly labeled sanitized derivatives, with
separate source seals, public seals and hashes. Originals remain unchanged locally.

## Interruption and resume

There is **no silent resume**. Raw evidence is create-once and the manifest is
sealed once at completion, so an interrupted process can never append into a
sealed run:

- If the process is interrupted or a generation raises, `run_experiment` marks the
  run `PARTIAL`/`FAILED`, publishes the partial traces/probes, and writes **no
  seal**. `verify` fails on such a run (no evidence seal).
- Starting a new run with the same name/deployment in the same second raises
  `FileExistsError` rather than overwriting.
- Disk exhaustion or a partial write surfaces as a failed/partial run; the
  verifier's structural checks fail closed on truncated JSONL.
- Re-running a failed experiment creates a **new** run directory; completed trials
  are never counted twice. Robust mid-run resumption is not implemented and is
  documented here rather than faked.
- **Probe arms are independent sealed runs.** An interrupted `probe-existing` arm
  is never resumed; it must be rerun in full. A *completed* probe arm is sealed
  and banked, so for the Phase 1.8 BF16/Q4 comparison only the remaining arm need
  be run. The current pass has the BF16 arm complete and the Q4 arm not run
  (`runs/phase1_8-status.json`); the two arms are then joined offline by
  `ponderscope recoverability`.

## Weight-hash cache caveat

Snapshot file hashes are cached by `(path, size, mtime)` under
`~/.cache/ponderscope/` so repeated runs do not re-hash gigabytes of weights. If a
file were replaced with identical size and mtime the cached hash could be stale;
in practice HF cache blobs are content-addressed and immutable. Delete the cache
to force a full re-hash.

## Public model and evidence downloads

The actual standalone [MLX Q4 weights](https://huggingface.co/Dontbeafed69/Qwen3.5-4B-MLX-4bit-PonderScope)
and [calibration/replication dataset](https://huggingface.co/datasets/Dontbeafed69/PonderScope-Qwen3.5-4B-Quantization-Reasoning)
are public. The model card contains tested MLX-LM load/generation commands; the
dataset card contains viewer-compatible split loading and offline evidence
verification instructions. [The release record](HUGGINGFACE_RELEASE.md) lists
the exact files, source and published hash boundaries, privacy redactions, and
`scripts/export_huggingface.py` usage with immutable-source validation. No custom
PonderScope model loader is needed for the published model.
