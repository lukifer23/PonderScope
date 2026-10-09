# PonderScope report — 20261009T000607Z-phase1-6-replication-pilot-q4-pilot-dep-4f8e7958df4f

_Generated from saved evidence; no numbers are hand-entered._

## Deployment identity
- description: `Qwen/Qwen3.5-4B@851bf6e806ef (derived, bfloat16 quant=mlx-affine4bit g64) under mlx-lm 0.32.0 [mlx-metal, Apple M3 Pro] src=src-6209ef70e318 wvar=wvar-b7a30758909a dep=dep-4f8e7958df4f`
- source artifact id: `src-6209ef70e318`
- weight variant id: `wvar-b7a30758909a`
- deployment id: `dep-4f8e7958df4f`
- prompt policy: `pp-v1`
- model policy: `qwen35-4b-thinking-api-v1` (label `qwen-generated-history-presence-v1`) recommended=`{'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 1.5, 'repetition_penalty': 1.0, 'enable_thinking': True}`
- condition ids: `cond-f5e762d3078f`
- model repo: `Qwen/Qwen3.5-4B`
- revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- representation: `derived`
- precision: `bfloat16`
- quantization: `mlx-affine` bits=4 group=64
- runtime: `mlx-lm 0.32.0` backend=`mlx-metal`
- hardware: `Apple M3 Pro` os=`Darwin 27.2.0`
- created: `20261009T000607Z`

## Machine / runtime environment
- python: `3.12.12`
- platform: `macOS-27.2-arm64-arm-64bit`
- mlx device: `Device(gpu, 0)`
- memory bytes: `19327352832`
- packages: `{'mlx': '0.32.3', 'mlx-lm': '0.32.0', 'mlx-metal': '0.32.3', 'numpy': '2.5.3', 'transformers': '5.18.0', 'tokenizers': '0.23.2', 'huggingface-hub': '1.33.0', 'safetensors': '0.8.0', 'ponderscope': '0.1.0'}`

## Task population
- n tasks: 5  n generations: 5
- spec: `{'name': 'phase1-6-replication-pilot', 'task_pack': 'tasks-v1', 'prompt_policy': 'pp-v1', 'model_policy': 'qwen35-4b-thinking-api-v1', 'families': ['arith', 'path', 'order', 'logic', 'sm'], 'n_per_family': 1, 'task_seed': 0, 'split': 'dev', 'greedy_repeats': 0, 'sampled_seeds': [0], 'sampled_repeats_per_seed': 1, 'max_tokens': 2048, 'sampled_temperature': 1.0, 'sampled_top_p': 0.95, 'sampled_top_k': 20, 'sampled_min_p': 0.0, 'sampled_presence_penalty': 1.5, 'sampled_presence_context_size': 20, 'sampled_presence_scope': 'generated_history', 'sampled_repetition_penalty': 1.0, 'sampled_repetition_context_size': 20, 'sampled_frequency_penalty': 0.0, 'sampled_frequency_context_size': 20, 'probe': False, 'n_probes': 4, 'probe_max_tokens': 64, 'capture_entropy': True, 'capture_top_k': 5, 'capture_logprob_digest': False, 'capture_level': 'minimal', 'notes': 'Phase 1.6 operational pilot: a dev-split subset (5 families x 1 task, seed 0 = 5 draws) used only to verify generation, scoring, sealing, and comparison completeness on both deployments. It is deliberately NOT the frozen held-out test population and its outcomes are never used to tune the replication.'}`

## Configurations
| condition_id | mode | exec | draws | success_at_budget (95% CI) | completion_rate | censored_rate | conditional_acc_given_completed | reasoning tokens mean | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| cond-f5e762d3078f | sampled | 5 | 5 | 0.6 [0.2, 1] | 0.6 | 0.4 | 1 | 1338 | 42.07 |

## Reasoning-length distribution

### cond-f5e762d3078f (sampled)

```
     465-544         2 ########################################
     544-623         0 
     623-702         0 
     702-782         0 
     782-861         0 
     861-940         0 
     940-1019        0 
    1019-1098        0 
    1098-1177        0 
    1177-1256        0 
    1256-1336        0 
    1336-1415        0 
    1415-1494        0 
    1494-1573        0 
    1573-1652        1 ####################
    1652-1731        0 
    1731-1811        0 
    1811-1890        0 
    1890-1969        0 
    1969-2048        2 ########################################
```
- {"n": 5, "mean": 1337.6, "median": 1607.0, "std": 792.4350446566583, "min": 465.0, "max": 2048.0, "p10": 487.0, "p90": 2048.0}

## Latency / throughput
- cond-f5e762d3078f: wall_ms {"n": 5, "mean": 34030.21441639867, "median": 43109.07895799755, "std": 18038.99250988113, "min": 13091.303499997593, "max": 49728.4489579979, "p10": 14206.423016398912, "p90": 49174.3247247985}; tok/s {"n": 5, "mean": 42.07498445952211, "median": 42.3638197444608, "std": 0.7184821814064716, "min": 41.1836693665994, "max": 42.77648898752542, "p10": 41.285445936457464, "p90": 42.7110267556274}; ttft_ms {"n": 5, "mean": 251.91333319817204, "median": 239.0857499995036, "std": 62.32190225411566, "min": 177.5072079981328, "max": 340.18529199965997, "p10": 194.27832479850622, "p90": 317.452541597595}

## Repetition
- cond-f5e762d3078f: unique_token_ratio {"n": 5, "mean": 0.2589152819262173, "median": 0.21728515625, "std": 0.07010110946795876, "min": 0.2099609375, "max": 0.3723404255319149, "p10": 0.21111537578252587, "p90": 0.3362613981762918}; repeated_ngram_fraction_4 {"n": 5, "mean": 0.3095631395167365, "median": 0.3095354523227384, "std": 0.12253858126193495, "min": 0.14198473282442747, "max": 0.4880195599022005, "p10": 0.20440089355462057, "p90": 0.41691206309508255}; text_repeat_ratio {"n": 5, "mean": 0.0353304186140007, "median": 0.02985074626865669, "std": 0.02807744660374379, "min": 0.0, "max": 0.07407407407407407, "p10": 0.009090909090909084, "p90": 0.06444444444444446}

## Repeatability / noise floor

### 1. Greedy replay (same deterministic condition, repeated)
- no repeated greedy conditions in this run

### 2. Same-seed sampled replay (same seed + sampler, re-executed)
- no repeated same-seed sampled conditions in this run

### 3. Across-seed stochastic variation (fixed sampler policy, different seeds)
- fewer than two seeds per task/condition

### 4. Across-seed token variation (valid even when all answers are censored)
- fewer than two seeds per task/condition

## Censoring / termination (do not read a capped run as a wrong answer)

- capped_length: 2
- eos_observed: 3
- missing_answer: 2
- natural_eos: 3
- status:censored: 2
- status:correct: 3
- think_end_reached: 3

### Budget outcomes per condition
| condition_id | success_at_budget | completion_rate | censored_rate | error_rate | unparseable_rate | cond_acc|completed |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 0.6 | 0.6 | 0.4 | 0 | 0 | 1 |

## Family reconciliation (closures vs EOS are distinct)

| condition | family | n | native closures | EOS | censored | no-closure | unparseable | errors | observed answers | correct answers |
|---|---|---|---|---|---|---|---|---|---|---|
| cond-f5e762d3078f | **ALL** | 5 | 3 | 3 | 2 | 0 | 0 | 0 | 3 | 3 |
| cond-f5e762d3078f | arith | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| cond-f5e762d3078f | logic | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| cond-f5e762d3078f | order | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| cond-f5e762d3078f | path | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| cond-f5e762d3078f | sm | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

## Censor-aware time-to-closure (Kaplan-Meier / RMST)

- observation horizon tau: 2048.0 tokens (an observation horizon, not a natural stopping threshold)
- event: PRIMARY: native think-end reasoning closure observed
- secondary event: SECONDARY: EOS observed (generation termination), timed at the terminal token, independent of the reasoning-closure time
- analysis unit: unique stochastic draws (same-seed technical repeats collapsed)

| condition | exec | draws (used) | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 5 | 5 (5) | 3 | 2 | 1608 | 1338 |

Secondary endpoint (generation termination via EOS):

| condition | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|
| cond-f5e762d3078f | 3 | 2 | 1837 | 1430 |

## Loop-structure diagnostics (descriptive)

Descriptive loop structure with an explicitly documented onset rule; not a validated universal loop detector.

| condition | n | longest_run median (max) | special longest runs | with degeneration onset |
|---|---|---|---|---|
| cond-f5e762d3078f | 5 | 2 (3) | 0 | 0 |

## Prefix probes / trajectory states

- no probes in this run

## Limitations
- Single model revision on a single machine; results do not generalize to other weights or hardware.
- Greedy determinism is a same-process, same-machine replay test.
- Where the reasoning channel was never closed, the trajectory is censored, not a natural stop.
- No causal claims: deployment differences are associations under this protocol.
