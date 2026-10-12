# Quickstart

Requires macOS on Apple Silicon, Python 3.12, and
[`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Install
uv sync --frozen --extra dev

# 2. Check the environment and load the pinned model (no generation)
uv run ponderscope doctor

# 3. Generate and validate the frozen task pack
uv run ponderscope generate --pack tasks-v1 --out tasks/tasks-v1.dev.jsonl

# 4. Run a small declared experiment (example: the Q4 pilot)
uv run ponderscope run \
  --spec specs/phase1_4-4b-q4-pilot.json \
  --artifact-path models/qwen3.5-4b-mlx-q4-affine-g64 \
  --model-repo Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

# 5. Verify and report
uv run ponderscope verify --run runs/<id>
uv run ponderscope analyze --run runs/<id> --report

# 6. Compare two deployments (contrast is classified; confounds are refused)
uv run ponderscope compare --a runs/<id> --b runs/<id> --mode sampled
```

The 4B model must already be in the local Hugging Face cache at the pinned
revision. See `docs/REPRODUCIBILITY.md` for conversion and caching.

## Probing saved reasoning prefixes (Phase 1.8)

Replays forced finalization on an existing sealed run's saved reasoning prefixes
(no new natural trajectories), then joins two arms offline:

```bash
# One arm (add --artifact-path ... for the controlled Q4 arm).
uv run ponderscope probe-existing \
  --source-run runs/<source-run> --label phase1-8-probe-bf16 \
  --checkpoints 0 256 512 1024 1536 --probe-max-tokens 512 --seeds 0 \
  --model-repo Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

# Offline paired analysis over the two sealed probe runs.
uv run ponderscope recoverability \
  --bf16-probe runs/<bf16-probe> --q4-probe runs/<q4-probe> \
  --out runs/phase1_8-recoverability.json --md report.md --html report.html
```

The current pass has the BF16 arm complete and the Q4 arm not yet run; see
`runs/phase1_8-status.json` and `docs/PHASE1_8_REPORT.md`.
