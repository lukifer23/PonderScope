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
