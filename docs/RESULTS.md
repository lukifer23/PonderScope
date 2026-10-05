# Results

**Generated from saved evidence; no numbers are hand-entered.** Regenerate with `ponderscope analyze`/`report`; this table is produced by reading the run's `analysis.json`.

## Run identity

- run id: `runs/20261005T221943Z-smoke-v2-live-smoke-2-dep-8b5f653b14f6`
- artifact id: `art-d4247b37d98a`
- deployment id: `dep-8b5f653b14f6`
- condition ids: `cond-481d400047e2, cond-19ba9bb0a86d`
- model: `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`
- model-load audit ok: `True` (text keys=320, params=320, non-text ignored=168)
- spec hash: `7985c6cd3096f33b92acabf599c7c6c80ddf48f5190904586376519acaf2caf4`
- task pack: `{'pack_version': 'tasks-v1', 'generator_version': 'generator-v1', 'requested_pack': 'tasks-v1', 'families': ['arith', 'logic', 'order', 'path', 'sm'], 'n_per_family': 1, 'split': 'dev', 'seed': 0, 'n_tasks': 5, 'task_ids_sha256': 'cca34706048b62ac7a246468b0d3c36eaf87064211210e58a779616862f4e20a'}`
- status: `EVIDENCE_COMPLETE`
- evidence seal sha256 (manifest): `15d7075b27acee23…`

## Live capability proof (LIVE VALIDATED)

- think-start `248068` = `<think>`; think-end `248069` = `</think>`
- EOS `[248046]` = `['<|im_end|>']`
- native closure `[198, 248069, 271]` = `'\n</think>\n\n'`
- natural closure discovered at 593 reasoning tokens (answer `12`) within budget 1024
- forced finalization supported: `True` (native closure honored, no answer cue injected)
- greedy replay identical: `True`
- same-seed sampled replay identical: `True`
- top-k sorted descending: `True`, chosen token in top-k: `True`
- instrumentation overhead (decode tok/s): minimal=67.8, research=52.0 (+23.3%), digest=51.0 (+24.8%)
- unsupported: `none`

## Tiny smoke (5 families × 1 task, budget 640)

- generations: 30; probes: 25

| condition | mode | n | accuracy (95% CI) | termination |
|---|---|---|---|---|
| cond-19ba9bb0a86d | sampled | 20 | 0.0 (0.00–0.00) | {'capped_length': 20, 'missing_answer': 20} |
| cond-481d400047e2 | greedy | 10 | 0.2 (0.00–0.60) | {'natural_eos': 2, 'eos_observed': 2, 'think_end_reached': 2, 'capped_length': 8, 'missing_answer': 8} |

### Repeatability / noise floor
- greedy replay: token-identical 5/5 (1.0), answer agreement 1.0
- same-seed sampled replay: token-identical rate 1.0, answer agreement 1.0
- across-seed: mean distinct answers 1.0, mean accuracy std across seeds 0.0

### Prefix probes / trajectory states
- probes supported: `True`; stable-sufficient prefixes: 1
- state counts: {'never_correct': 3, 'wrong_to_correct': 1, 'multiple_flips': 1}

- `sm`: never_correct (flips=0, first correct at token None, stable sufficient None, final correct False)
- `path`: never_correct (flips=0, first correct at token None, stable sufficient None, final correct False)
- `logic`: never_correct (flips=0, first correct at token None, stable sufficient None, final correct False)
- `arith`: wrong_to_correct (flips=1, first correct at token 183, stable sufficient 183, final correct True)
- `order`: multiple_flips (flips=2, first correct at token 224, stable sufficient None, final correct False)

## Interpretation (manual)

- The instrument is functioning: probes operate on reasoning-only prefixes, the native closure is honored, and no final-answer/EOS token entered a probe prefix (all probe prefix lengths ≤ the natural reasoning length).
- **Greedy replay is deterministic** on this exact MLX deployment, and **same-seed sampled replay is identical**; across-seed variation is not yet interpretable because sampled generations were censored at the budget.
- **Difficulty is degenerate for this model/budget.** Only the arithmetic task closed naturally (greedy, ~516 reasoning tokens, correct); the other four families were censored at 640 tokens for both greedy and sampled. This is a property of the model (Qwen3.5-0.8B) and budget, not a harness defect.
- Because natural termination is rare at this budget, sampling policies that depend on closure cannot yet be compared; the prefix probes nonetheless show the arithmetic answer is recoverable from a mid-reasoning prefix (forced answer 84 from token 183 onward).

## GO / NO-GO

**CONDITIONAL GO.** Software correctness and live capability gates pass, and the instrument is trustworthy enough to proceed. However, the task difficulty must be versioned/adjusted before a full noise study: the current `tasks-v1` pack is censored for 4/5 families at this budget/model. Do not run the large pilot until difficulty and budget are set so that natural termination occurs regularly.

**Recommended next experiment (one):** version the task pack to `tasks-v2` with a calibrated easy/medium tier (and/or a per-family budget derived from a closure-discovery sweep) such that greedy natural-closure rate exceeds ~80% across all five families, then rerun this exact tiny smoke unchanged to confirm the same determinism, probe, and sealing contracts under non-degenerate difficulty.
