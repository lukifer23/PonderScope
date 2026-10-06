# PonderScope Phase 1.3B — sampler fidelity and primary-baseline completion

**Governing principle.** Technical repeats measure reproducibility. Seeds measure
stochastic behavior. Tasks are the statistical clusters. And upstream parameter
*values* are not upstream serving *semantics* unless we actually reproduce the
semantics.

**Classification key.** `IMPLEMENTED`, `LIVE VALIDATED`, `MEASURED`, `HYPOTHESIS`,
`UNSUPPORTED`, `NOT YET TESTED`.

## Starting state

- Starting `main` SHA: `cbca49c39611003e8cc320b0420ae52076b37978` (matched
  `origin/main`; tracked worktree clean). No history rewritten, no feature branch,
  no Docker, no production mocks.
- Final `main` SHA: see the commit that adds this file.
- This phase did **not** run BF16→Q4, a stopping controller, or dev/test.

## Commits created (in order)

1. `e041118` feat: Phase 1.3B generated-history presence semantics,
   stochastic-draw units, and honest run descriptions (code + tests + profiles +
   specs + doc corrections + preserved-evidence Stage A reanalysis).
2. `947c530` research: Phase 1.3B Stage A2 generated-history calibration,
   horizon sensitivity, and T=0.6 diagnostic.
3. `3a2cc0f` feat: Qwen3.5-4B thinking profile, Stage B specs, and 4B BF16
   load/memory audit.
4. `5b3f904` research: Phase 1.3B Stage B Qwen3.5-4B BF16 scale control and
   model-scale comparison.
5. (this report) docs: Phase 1.3B final report.

CI: green on `e041118` (`gh run 37400486234`). Subsequent pushes trigger the
same `ci` workflow; final result recorded at the end.

## Tests / quality gates

- `uv sync --frozen --extra dev`, `ruff check`, `ruff format --check`, `mypy src`,
  `git diff --check`: pass.
- `pytest`: **173 passed** (Phase 1.3B added 25 tests: generated-history presence
  excludes prompt / persists beyond 20 tokens / set semantics; zero-penalty
  no-op; mlx_window vs generated_history divergence; scope participates in
  condition identity and preserves `cond-74a29a115886`; backend processor
  wiring/labels; stochastic_draw_id repeat-independence; trial id changes with
  repeat; identical-replicate collapse; ambiguous-replicate fail-closed;
  KM/RMST unit and unequal-repeat invariance; reasoning-closure-requires-think-end
  vs EOS-only termination; across-seed support counts; sampled-only run
  description never claims greedy; policy provenance for T=1.0/T=0.6).

## Confirmed Stage A methodological defects (fixed)

| # | Defect | Fix |
|---|---|---|
| 1 | The upstream *numeric* values were mapped to MLX-LM's presence penalty, which examines the last `presence_context_size` positions of the **prompt-inclusive** accumulated history — not OpenAI/vLLM generated-text semantics. The condition was described as a faithful reproduction. | Relabelled `qwen-upstream-values-mlx-window20`; added explicit `presence_scope` and a narrow generated-history processor. |
| 2 | `presence_scope` did not exist, so penalty *semantics* could not participate in condition identity. | `presence_scope` added to `DecodingPolicy`; `mlx_window` normalized away (historical ids preserved), `generated_history` changes identity and drops the irrelevant context size. |
| 3 | The run-level `deployment_description` was built from a dummy **greedy** policy and claimed `greedy … cond-cond-2f336…` although only a sampled condition ran. | Description is now model/weight/runtime/hardware only; conditions are described separately from the actual declared policy. |
| 4 | Same-seed technical repeats were counted as independent survival subjects (60 executions). | Added `stochastic_draw_id`; primary summaries collapse repeats to unique draws (30), flag ambiguities, and expose `n_executions` and `n_unique_draws`. |
| 5 | `_is_closure = think_end OR eos`; an EOS without a native think-end was labelled reasoning closure. | PRIMARY reasoning closure requires the native think-end; EOS is a separate SECONDARY generation-termination endpoint. |
| 6 | `mean_distinct_answers` was reported without its support. | Added `n_task_conditions_total`, `n_task_conditions_with_estimable_answer_diversity`, `proportion_estimable`; no mean without support. |
| 7 | The report attributed the disappearance of the 1,201-token run too strongly to the presence penalty. | Replaced with descriptive wording (see below). |

## Corrected interpretation

- Stage A is **`qwen-upstream-values-mlx-window20`**: the published numeric
  settings mapped through MLX-LM's 20-token prompt-inclusive presence processor.
- "Official sampling did not fix termination" →
  **"the Qwen-recommended numeric values mapped through MLX-LM's 20-token
  prompt-inclusive presence processor did not restore termination."**
- Loop morphology: **"Under the sampled + penalty condition, the maximum
  identical-token run in the measured sample was 3 rather than the 1,201-token
  historical greedy example, while repeated multi-token motifs remained."**
  This is descriptive; greedy→sampling, temperature, RNG trajectory, presence
  penalty, and prompt context all changed together, and no presence-only
  ablation was run.
- Upstream recommends 32,768 output tokens; `max_tokens=2048` is always an
  **observation horizon**, never a claim about absolute nontermination.

## Old Stage A reanalysis (preserved raw evidence)

Raw files and the on-disk `analysis.json` are unchanged. Artifact:
`runs/stage-a-mlx-window20-reanalysis.phase1_3b.json` (sha256 `d307f60db94210d9…`).

| quantity | value |
|---|---|
| executions | 60 |
| unique stochastic draws | 30 |
| ambiguous draws | 0 |
| reasoning-closure events (native think-end) | 2 |
| censored draws | 28 |
| completion proportion | 2/30 = 6.67% |
| RMST(2048), draw-level | 1956.5 |
| generation-termination events (EOS) | 2 |

The four completed executions are two arith draws measured twice.

## Generated-history presence implementation

- `src/ponderscope/backends/processors.py`:
  `make_generated_history_presence_penalty(penalty, prompt_len)`.
- Semantics: prompt tokens never count; a token appearing anywhere in the
  **generated** output receives the penalty once; history is the full generated
  sequence, not a rolling window; `penalty == 0` is a no-op; deterministic.
- Processor order remains `repetition → presence → frequency`. Repetition and
  frequency stay MLX-LM built-ins; upstream `repetition_penalty=1.0` stays a
  no-op (no manufactured effect).
- Labels: `mlx_window` → `qwen-upstream-profile-on-mlx` (historical);
  `generated_history` → `qwen-generated-history-presence-v1`. The two are never
  swapped silently. Both semantics remain available.

## Stage A2 — 0.8B generated-history T=1.0 qualification

Run `runs/20261006T014327Z-calibration-08b-generated-history-api-dep-d7a370157187`
(deployment `dep-d7a370157187`, condition `cond-f5e762d3078f`, scope
`generated_history`); evidence.json sha256 `1b55af4531860848…`, manifest
`02559e2a7ffe1441…`. Design: tasks-v1 calibration, pp-v1, 10 presentations,
2048 horizon, minimal capture, seeds {0,1,2} × **one** primary execution = **30
stochastic draws**. `ponderscope verify` = PASS.

| metric | value |
|---|---|
| reasoning closures (primary) | 1/30 = 3.33% |
| censored | 29/30 = 96.67% |
| success_at_budget | 0.033 |
| conditional accuracy given completion | 1.0 |
| RMST(2048), draw-level | 1987.7 |
| median tokens to closure | not estimable |
| reasoning tokens mean | 1986.0 |
| repeated-4-gram fraction mean | 0.359 |
| longest identical-token run max | 4 |
| cross-seed first-token divergence (mean) | 6.3 |
| cross-seed reasoning-token std (mean) | 87.7 |
| minimal-lane decode throughput (mean) | 62.3 tok/s |

Same-seed technical replay subset (`runs/same-seed-replay-08b-generated-history.phase1_3b.json`,
sha256 `36a12f1e84c836a2…`): arith/order/sm at seed 0, 3 executions each, all
**token-identical** → the custom processor is reproducible. These executions are
not part of N.

Descriptive comparison vs the preserved MLX-window20 Stage A
(`ponderscope compare --decoding-intentional`, 30 matched draws): closure delta
−1 draw; all termination contrasts below the noise floor. Generated-history
semantics do **not** restore termination.

### Bounded horizon sensitivity (triggered)

`runs/horizon-sensitivity-8192.phase1_3b.json` (sha256 `9112ab43ce71de13…`): one
predeclared seed, one **easy** task per family, 2048 vs 8192.

| family | 2048 | 8192 |
|---|---|---|
| arith | closes at 240 | closes at 240 |
| order | censored | closes at 5311 |
| logic | censored | censored at 8192 |
| path | censored | censored at 8192 |
| sm | censored | censored at 8192 |

Mixed: 2048 understates ordinary closure for `order` (closure above 2048), while
`logic`, `path`, and `sm` show persistent nontermination at 8192. The model was
not chased beyond this bounded run.

### T=0.6 benchmark diagnostic

`runs/20261006T021446Z-calibration-08b-generated-history-benchmark-dep-d7a370157187`
(condition `cond-4555bfefa490`; evidence sha256 `790bf5459ad8959f…`): seed 0, 10
presentations, 2048. **0/10 closures, 100% censored, RMST(2048)=2048**,
repeated-4-gram mean 0.504 (higher than T=1.0). T=0.6 is qualitatively the same
nontermination regime; T=0.6 and T=1.0 are **not** collapsed into one "Qwen
recommended" condition.

### 0.8B Stage A2 decision

**TERMINATION-STRESS DEPLOYMENT.** Severe censoring persists across greedy
(historical), MLX-window upstream values, generated-history T=1.0, and
generated-history T=0.6. 0.8B is retained as a stress model.

## Stage B — Qwen3.5-4B native BF16 model/scale control

Downloaded the official checkpoint with explicit authorization.

- Repo/revision: `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
  (immutable commit; official upstream, no community conversion, no
  quantization).
- Checkpoint: snapshot total 9,342,894,355 bytes; weight shards 9,319,828,096
  bytes. Weight file sha256: `26a93f06…` (shard 1), `cb544bd9…` (shard 2).
  `tokenizer.json` `5f9e4d49…`; `chat_template.jinja` `a4aee8af…`; model-card
  README sha256 `1406be1b…` (77,661 bytes).
- Policy profile `model_policies/qwen35-4b-thinking-api-v1.json`
  (`qwen-generated-history-presence-v1`; T=1.0, top_p 0.95, top_k 20, min_p 0,
  presence 1.5, repetition 1.0).

### Memory / load qualification

`runs/model-load-audit-4b.phase1_3b.json` (sha256 `7d471f760762da69…`):

| quantity | value |
|---|---|
| detected precision | bfloat16 |
| load audit | ok (426 text params; 0 dropped text keys; 0 unexpected) |
| load time | 4.4 s |
| MLX peak after load | 8.41 GB |
| MLX peak after 64-token generate | 8.53 GB |
| physical memory | 19.33 GB |
| 4B seed-0 pilot wall/generation (mean) | 112.6–116.9 s |

4B loads and runs safely; no OOM, no fallback to 2B required.

### Stage B results

- Pilot `runs/20261006T022442Z-calibration-4b-generated-history-api-seed0-dep-e1c20569173a`
  (10 draws): 7/10 closures.
- Full `runs/20261006T024435Z-calibration-4b-generated-history-api-dep-e1c20569173a`
  (seeds {0,1,2} × 1 = 30 draws; condition `cond-f5e762d3078f`, shared with A2
  because condition identity excludes the model); evidence sha256
  `31fd272b2e2836fc…`, manifest `701524fe4850ffff…`; verify PASS.
- Replay `runs/same-seed-replay-4b-generated-history.phase1_3b.json`
  (sha256 `4792f5cd674027cb…`): arith/order/sm token-identical.

| metric | 4B BF16 | 0.8B A2 |
|---|---|---|
| unique draws | 30 | 30 |
| reasoning closures | 20 (66.7%) | 1 (3.3%) |
| censored | 10 (33.3%) | 29 (96.7%) |
| success_at_budget | 0.633 | 0.033 |
| conditional accuracy given completion | 1.0 | 1.0 |
| RMST(2048) | 1511.9 | 1987.7 |
| median tokens to closure | 1718 | not estimable |
| reasoning tokens mean | 1368.1 | 1986.0 |
| repeated-4-gram fraction mean | 0.391 | 0.359 |
| longest identical-token run max | 3 | 4 |
| cross-seed reasoning-token std (mean) | 250.6 | 87.7 |
| decode throughput (mean) | 13.4 tok/s | 62.3 tok/s |

4B per family (draws with native closure): arith 6/6, order 6/6, sm 6/6,
path 2/6, **logic 0/6** (logic remains a termination-stress family even at 4B).

## Model-scale comparison (4B vs 0.8B)

`runs/model-scale-comparison-4b-vs-08b.phase1_3b.json` (sha256
`63c1de401f04b234…`). Same presentations, same 2048 horizon, same
generated-history policy semantics, same condition id; **model/scale comparison,
NOT a deployment-drift causal result**.

- Reasoning-closure-rate delta (4B − 0.8B): **+0.633**, task-clustered 95% CI
  [0.367, 0.867].
- RMST delta (4B − 0.8B): **−475.8 tokens**, task-clustered bootstrap 95% CI
  [−700.5, −253.4], excludes zero. Negative means 4B unclosed-reasoning time is
  shorter.
- Both runs have one execution per draw, so the within-(task,seed) noise scale is
  unavailable; the task-clustered bootstrap CI is the reported uncertainty.

## Primary-baseline decision

A primary baseline must have exact provenance, trustworthy native BF16 loading,
manageable M3 Pro memory/runtime, enough natural completions under a defensible
thinking policy, and repeatable execution.

`Qwen/Qwen3.5-4B` BF16 satisfies all five: pinned immutable revision with file/
tokenizer/template/card hashes; load audit ok at bfloat16; MLX peak 8.5 GB of
19.3 GB with ~2 min/generation; 20/30 natural closures and a measurable
time-to-closure (median 1718, RMST 1511.9); same-seed replay token-identical.

**A PRIMARY BF16 BASELINE NOW EXISTS: `Qwen/Qwen3.5-4B` @ `851bf6e8`.**

Residual caveat: the `logic` family is still censored 6/6 at 2048 even at 4B, so
future work should keep logic-specific censoring in view. 0.8B is retained as the
stress model.

## Research-gap status

The central deployment-drift question (same source revision, native BF16 vs the
controlled MLX Q4 derived from that exact revision) remains **NOT YET TESTED**.
Phase 1.3B removes the blocker: it establishes a trustworthy BF16 baseline and
proves the measurement instrument handles the sampler semantics, statistical
units, endpoints, and descriptions correctly.

## Decision

**GO** — with the baseline now frozen for the next (deployment-drift) protocol.

## Exactly ONE next experiment

Run the first central deployment-drift comparison on the frozen baseline:
**`Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, native BF16 vs
a controlled MLX Q4 weight variant derived from that exact revision**, using the
same tasks-v1 calibration presentations, pp-v1, the same generated-history
thinking policy, the same 2048 observation horizon, and unique stochastic draws
as the analysis unit. Do not add a stopping controller, other models, or other
quantizations in that experiment.
