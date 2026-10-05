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

_Phase 1.2 correction: budget outcomes, censoring, and trajectory states below
were regenerated from the immutable raw traces. The Phase 1.1 interpretation was
wrong (see `PHASE1_2_CORRECTION.md`)._

| condition | mode | n | success_at_budget | completion_rate | censored_rate | cond_acc\|completed |
|---|---|---|---|---|---|---|
| cond-19ba9bb0a86d | sampled | 20 | 0.0 | 0.0 | 1.0 | — (no completed answers) |
| cond-481d400047e2 | greedy | 10 | 0.2 | 0.2 | 0.8 | 1.0 |

### Repeatability / noise floor
- greedy replay: token-identical 5/5 (1.0), answer agreement 1.0
- same-seed sampled replay: token-identical rate 1.0, ambiguous seeds 0
- across-seed: distinct **observed** answers `—` (unobserved), observed-answer
  accuracy std `—` (insufficient); token-level first divergence across seeds is
  still observable (mean 10.8 tokens)

### Prefix probes / trajectory states (corrected)
- probes supported: `True`
- stable-sufficient WITH observed natural final: 1
- observed-probe stable (natural final unobserved or uncounted): 2
- state counts: {'never_correct': 3, 'wrong_to_correct': 2}

- `sm`: never_correct (flips=0, observed-probe stable None, natural final `censored`)
- `path`: never_correct (flips=0, observed-probe stable None, natural final `censored`)
- `logic`: never_correct (flips=0, observed-probe stable None, natural final `censored`)
- `arith`: wrong_to_correct (flips=1, first correct at token 183, natural-final
  sufficient 183, natural final `correct`)
- `order`: wrong_to_correct (flips=1, first correct at token 224, observed-probe
  stable from 224, natural-final sufficient `None`, natural final `censored`)

## Interpretation (manual, Phase 1.2)

- The instrument is functioning: probes operate on reasoning-only prefixes, the
  native closure is honored, and no final-answer/EOS token entered a probe
  prefix.
- **Greedy replay is deterministic** on this exact MLX deployment, and
  **same-seed sampled replay is identical**. Across-seed final-answer variation
  is **unobserved** because every sampled generation was censored at 640; token
  trajectories nonetheless differ across seeds.
- **Do not call a capped run wrong.** Under this deployment and budget, 8/10
  greedy and 20/20 sampled generations remained in the reasoning channel at the
  640-token observation horizon. `success_at_budget` is 0.2 for greedy and 0.0
  for sampled; `conditional_accuracy_given_completed` is 1.0 for greedy and
  undefined when nothing completed.
- Only the arithmetic task closed naturally (greedy, ~516 reasoning tokens,
  correct). The other four families were censored at 640 for both greedy and
  sampled. This is a property of this deployment and budget, not yet shown to be
  task complexity — see the Phase 1.2 termination calibration.
- The prefix probes show the arithmetic answer is recoverable from a
  mid-reasoning prefix (forced answer 84 from token 183 onward), while `order`
  was forced-correct from token 224 even though its natural generation never
  closed.

## GO / NO-GO

> **Phase 1.1 statement below is historical and was superseded by Phase 1.2.**
> The Phase 1.1 claim that "the task difficulty must be versioned" was made
> before the censoring bug was understood. Phase 1.2 does **not** manufacture an
> easier `tasks-v2`; it treats failure to terminate as a phenomenon to measure.
> See `PHASE1_2_CORRECTION.md` and the Phase 1.2 report for the current decision.

**CONDITIONAL GO (Phase 1.1, historical).** Software correctness and live
capability gates pass, and the instrument is trustworthy enough to proceed.
However, the task difficulty must be versioned/adjusted before a full noise
study: the current `tasks-v1` pack is censored for 4/5 families at this
budget/model. Do not run the large pilot until difficulty and budget are set so
that natural termination occurs regularly.

**Recommended next experiment (Phase 1.1, historical):** version the task pack
to `tasks-v2`. **Phase 1.2 replaces this recommendation with a termination
calibration experiment** (see `EXPERIMENT_DESIGN.md` and the Phase 1.2 report).
