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
(sha256 `a9327e3d119fd561…`, 985,486 bytes, 10 members). Bundles are **not**
uploaded publicly without explicit approval.

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

## Weight-hash cache caveat

Snapshot file hashes are cached by `(path, size, mtime)` under
`~/.cache/ponderscope/` so repeated runs do not re-hash gigabytes of weights. If a
file were replaced with identical size and mtime the cached hash could be stale;
in practice HF cache blobs are content-addressed and immutable. Delete the cache
to force a full re-hash.
