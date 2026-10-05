# PonderScope report — 20261005T221943Z-smoke-v2-live-smoke-2-dep-8b5f653b14f6

_Generated from saved evidence; no numbers are hand-entered._

## Deployment identity
- description: `Qwen/Qwen3.5-0.8B@2fc06364715b (bfloat16) under mlx-lm 0.32.0 [mlx-metal, Apple M3 Pro] greedy art=art-d4247b37d98a dep=dep-8b5f653b14f6 cond=cond-481d400047e2`
- artifact id: `art-d4247b37d98a`
- deployment id: `dep-8b5f653b14f6`
- condition ids: `cond-481d400047e2, cond-19ba9bb0a86d`
- model repo: `Qwen/Qwen3.5-0.8B`
- revision: `2fc06364715b967f1860aea9cf38778875588b17`
- precision: `bfloat16`
- quantization: `None` bits=None group=None
- runtime: `mlx-lm 0.32.0` backend=`mlx-metal`
- hardware: `Apple M3 Pro` os=`Darwin 27.2.0`
- created: `20261005T221943Z`

## Machine / runtime environment
- python: `3.12.12`
- platform: `macOS-27.2-arm64-arm-64bit`
- mlx device: `Device(gpu, 0)`
- memory bytes: `19327352832`
- packages: `{'mlx': '0.32.3', 'mlx-lm': '0.32.0', 'mlx-metal': '0.32.3', 'numpy': '2.5.3', 'transformers': '5.18.0', 'tokenizers': '0.23.2', 'huggingface-hub': '1.33.0', 'safetensors': '0.8.0', 'ponderscope': '0.1.0'}`

## Task population
- n tasks: 5  n generations: 30
- spec: `{'name': 'smoke-v2', 'task_pack': 'tasks-v1', 'families': ['arith', 'path', 'order', 'logic', 'sm'], 'n_per_family': 1, 'task_seed': 0, 'split': 'dev', 'greedy_repeats': 2, 'sampled_seeds': [0, 1], 'sampled_repeats_per_seed': 2, 'max_tokens': 640, 'sampled_temperature': 0.6, 'sampled_top_p': 0.95, 'sampled_top_k': 20, 'sampled_min_p': 0.0, 'probe': True, 'n_probes': 4, 'probe_max_tokens': 64, 'capture_entropy': True, 'capture_top_k': 5, 'capture_logprob_digest': False, 'notes': 'Versioned tiny smoke after closure discovery (this model can need ~600 reasoning tokens before natural closure). 5 families x 1 task; 2 greedy repeats; 2 seeds x 2 repeats for same-seed replay and across-seed variation; forced probes. Budget 640 to bound runtime; censoring is recorded, not hidden.'}`

## Configurations
| condition_id | mode | n | accuracy (95% CI) | reasoning tokens mean | total tokens mean | wall ms mean | tok/s |
|---|---|---|---|---|---|---|---|
| cond-19ba9bb0a86d | sampled | 20 | 0 [0, 0] | 640 | 640 | 1.224e+04 | 52.37 |
| cond-481d400047e2 | greedy | 10 | 0.2 [0, 0.6] | 615.2 | 616.8 | 1.152e+04 | 53.52 |

## Reasoning-length distribution

### cond-19ba9bb0a86d (sampled)

```
     640-640         0 
     640-640         0 
     640-640         0 
     640-640         0 
     640-640         0 
     640-640         0 
     640-640         0 
     640-640         0 
     640-640         0 
     640-640         0 
     640-640        20 ########################################
     640-640         0 
     640-640         0 
     640-640         0 
     640-640         0 
     640-640         0 
     640-640         0 
     640-640         0 
     640-640         0 
     640-640         0 
```
- {"n": 20, "mean": 640.0, "median": 640.0, "std": 0.0, "min": 640.0, "max": 640.0, "p10": 640.0, "p90": 640.0}

### cond-481d400047e2 (greedy)

```
     516-522         2 ##########
     522-528         0 
     528-535         0 
     535-541         0 
     541-547         0 
     547-553         0 
     553-559         0 
     559-566         0 
     566-572         0 
     572-578         0 
     578-584         0 
     584-590         0 
     590-597         0 
     597-603         0 
     603-609         0 
     609-615         0 
     615-621         0 
     621-628         0 
     628-634         0 
     634-640         8 ########################################
```
- {"n": 10, "mean": 615.2, "median": 640.0, "std": 52.282990648117206, "min": 516.0, "max": 640.0, "p10": 516.0, "p90": 640.0}

## Latency / throughput
- cond-19ba9bb0a86d: wall_ms {"n": 20, "mean": 12238.489560299786, "median": 12247.595270499005, "std": 473.31239457298824, "min": 11427.879292008583, "max": 13087.461290997453, "p10": 11660.88215859636, "p90": 12813.71324169304}; tok/s {"n": 20, "mean": 52.368108161831046, "median": 52.25650986738407, "std": 2.017709414893342, "min": 48.901768323871984, "max": 56.00339167456436, "p10": 49.94882775761828, "p90": 54.88435563794892}; ttft_ms {"n": 20, "mean": 61.03124380024383, "median": 61.185625003417954, "std": 10.504013242454352, "min": 43.4095420059748, "max": 83.07583299756516, "p10": 45.20523690589471, "p90": 74.03664141020272}
- cond-481d400047e2: wall_ms {"n": 10, "mean": 11520.75980429654, "median": 11528.075583992177, "std": 591.4983146979507, "min": 10183.167916999082, "max": 12413.804124997114, "p10": 11139.203341705434, "p90": 12060.978699993575}; tok/s {"n": 10, "mean": 53.51991645360037, "median": 54.28813784000415, "std": 3.0236383223017738, "min": 46.59670846717895, "max": 56.5392380934145, "p10": 50.971388756499515, "p90": 56.235668205772065}; ttft_ms {"n": 10, "mean": 91.4242166953045, "median": 62.19639549817657, "std": 90.73969883058177, "min": 46.267124998848885, "max": 348.2941669935826, "p10": 55.4929872014327, "p90": 103.59504169464334}

## Repetition
- cond-19ba9bb0a86d: unique_token_ratio {"n": 20, "mean": 0.22296874999999994, "median": 0.234375, "std": 0.04498977634228633, "min": 0.1546875, "max": 0.2921875, "p10": 0.16171875, "p90": 0.27109375}; repeated_ngram_fraction_4 {"n": 20, "mean": 0.4337519623233909, "median": 0.4277864992150706, "std": 0.1362264360673718, "min": 0.24175824175824176, "max": 0.6828885400313972, "p10": 0.2869701726844584, "p90": 0.6249607535321823}; text_repeat_ratio {"n": 20, "mean": 0.04219602358382372, "median": 0.024390243902439046, "std": 0.049485659178519395, "min": 0.0, "max": 0.13793103448275867, "p10": 0.0, "p90": 0.0986987638256344}
- cond-481d400047e2: unique_token_ratio {"n": 10, "mean": 0.18239503816793892, "median": 0.1625, "std": 0.04423824401547891, "min": 0.1421875, "max": 0.25572519083969464, "p10": 0.1421875, "p90": 0.25572519083969464}; repeated_ngram_fraction_4 {"n": 10, "mean": 0.5829358467142947, "median": 0.6232339089481946, "std": 0.14942190047848047, "min": 0.4065934065934066, "max": 0.7551020408163265, "p10": 0.4065934065934066, "p90": 0.7551020408163265}; text_repeat_ratio {"n": 10, "mean": 0.14145842143031048, "median": 0.10344827586206895, "std": 0.1372507387453179, "min": 0.021739130434782594, "max": 0.37777777777777777, "p10": 0.021739130434782594, "p90": 0.37777777777777777}

## Repeatability / noise floor

### 1. Greedy replay (same deterministic condition, repeated)
- task/condition pairs with repeats: 5
- token-identical replays: 5 (1)
- answer-agreement rate: 1
- mean within-task reasoning-token std: 0
- mean within-task wall-ms std: 226.7

### 2. Same-seed sampled replay (same seed + sampler, re-executed)
- task/condition/seed groups: 10
- token-identical rate: 1
- answer-agreement rate: 1
- mean within-group reasoning-token std: 0

### 3. Across-seed stochastic variation (fixed sampler policy, different seeds)
- task/condition groups: 5
- mean distinct answers across seeds: 1
- mean accuracy std across seeds: 0
- mean reasoning-token std across seeds: 0

## Censoring / termination problems

- natural_eos: 2
- capped_length: 28
- think_end_reached: 2
- missing_answer: 28

## Prefix probes / trajectory states

- n probes: 25
- stable-sufficient prefixes observed: 1
- multiple_flips: 1
- never_correct: 3
- wrong_to_correct: 1

## Limitations
- Single model revision on a single machine; results do not generalize to other weights or hardware.
- Greedy determinism is a same-process, same-machine replay test.
- Where the reasoning channel was never closed, the trajectory is censored, not a natural stop.
- No causal claims: deployment differences are associations under this protocol.
