# PonderScope report — 20261009T005354Z-phase1-6-replication-bf16-replication-bf16-dep-e1c20569173a

_Generated from saved evidence; no numbers are hand-entered._

## Deployment identity
- description: `Qwen/Qwen3.5-4B@851bf6e806ef (original, bfloat16) under mlx-lm 0.32.0 [mlx-metal, Apple M3 Pro] src=src-6209ef70e318 wvar=wvar-9b6f9b40b789 dep=dep-e1c20569173a`
- source artifact id: `src-6209ef70e318`
- weight variant id: `wvar-9b6f9b40b789`
- deployment id: `dep-e1c20569173a`
- prompt policy: `pp-v1`
- model policy: `qwen35-4b-thinking-api-v1` (label `qwen-generated-history-presence-v1`) recommended=`{'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 1.5, 'repetition_penalty': 1.0, 'enable_thinking': True}`
- condition ids: `cond-f5e762d3078f`
- model repo: `Qwen/Qwen3.5-4B`
- revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- representation: `original`
- precision: `bfloat16`
- quantization: `None` bits=None group=None
- runtime: `mlx-lm 0.32.0` backend=`mlx-metal`
- hardware: `Apple M3 Pro` os=`Darwin 27.2.0`
- created: `20261009T005354Z`

## Machine / runtime environment
- python: `3.12.12`
- platform: `macOS-27.2-arm64-arm-64bit`
- mlx device: `Device(gpu, 0)`
- memory bytes: `19327352832`
- packages: `{'mlx': '0.32.3', 'mlx-lm': '0.32.0', 'mlx-metal': '0.32.3', 'numpy': '2.5.3', 'transformers': '5.18.0', 'tokenizers': '0.23.2', 'huggingface-hub': '1.33.0', 'safetensors': '0.8.0', 'ponderscope': '0.1.0'}`

## Task population
- n tasks: 30  n generations: 60
- spec: `{'name': 'phase1-6-replication-bf16', 'task_pack': 'tasks-v1', 'prompt_policy': 'pp-v1', 'model_policy': 'qwen35-4b-thinking-api-v1', 'families': ['arith', 'path', 'order', 'logic', 'sm'], 'n_per_family': 6, 'task_seed': 1729, 'split': 'test', 'greedy_repeats': 0, 'sampled_seeds': [0, 1], 'sampled_repeats_per_seed': 1, 'max_tokens': 2048, 'sampled_temperature': 1.0, 'sampled_top_p': 0.95, 'sampled_top_k': 20, 'sampled_min_p': 0.0, 'sampled_presence_penalty': 1.5, 'sampled_presence_context_size': 20, 'sampled_presence_scope': 'generated_history', 'sampled_repetition_penalty': 1.0, 'sampled_repetition_context_size': 20, 'sampled_frequency_penalty': 0.0, 'sampled_frequency_context_size': 20, 'probe': False, 'n_probes': 4, 'probe_max_tokens': 64, 'capture_entropy': True, 'capture_top_k': 5, 'capture_logprob_digest': False, 'capture_level': 'minimal', 'notes': 'Phase 1.6 independent replication (BF16 arm). Qwen3.5-4B native BF16 at the pinned revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, on the held-out test split (task_seed=1729, 5 families x 6 tasks = 30 previously unseen tasks, seeds {0,1} = 60 stochastic draws, 2048 horizon). Identical controlled dimensions to the Q4 arm; only the weight representation (selected via --artifact-path) differs. Preregistered before inspecting outcomes.'}`

## Configurations
| condition_id | mode | exec | draws | success_at_budget (95% CI) | completion_rate | censored_rate | conditional_acc_given_completed | reasoning tokens mean | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| cond-f5e762d3078f | sampled | 60 | 60 | 0.4333 [0.2667, 0.6] | 0.4333 | 0.55 | 1 | 1467 | 12.81 |

## Reasoning-length distribution

### cond-f5e762d3078f (sampled)

```
     292-380         4 #####
     380-468         1 #
     468-555         3 ####
     555-643         1 #
     643-731         4 #####
     731-819         3 ####
     819-907         1 #
     907-994         5 ######
     994-1082        3 ####
    1082-1170        1 #
    1170-1258        0 
    1258-1346        0 
    1346-1433        0 
    1433-1521        0 
    1521-1609        0 
    1609-1697        0 
    1697-1785        0 
    1785-1872        1 #
    1872-1960        0 
    1960-2048       33 ########################################
```
- {"n": 60, "mean": 1466.6166666666666, "median": 2048.0, "std": 684.9633193610126, "min": 292.0, "max": 2048.0, "p10": 475.7, "p90": 2048.0}

## Latency / throughput
- cond-f5e762d3078f: wall_ms {"n": 60, "mean": 124075.17053260053, "median": 146476.58281249824, "std": 50277.58471176822, "min": 28572.998250005185, "max": 257570.9326669894, "p10": 41878.751691601065, "p90": 172516.87424549746}; tok/s {"n": 60, "mean": 12.809853906808112, "median": 13.36457881310739, "std": 1.4005550879144157, "min": 7.864652720746778, "max": 14.410794849250271, "p10": 11.273644739054848, "p90": 14.013709100274202}; ttft_ms {"n": 60, "mean": 1342.3322784338477, "median": 403.1062079993717, "std": 6976.368788066472, "min": 256.05866700061597, "max": 54458.74037499743, "p10": 331.5340284992999, "p90": 521.5709034993779}

## Repetition
- cond-f5e762d3078f: unique_token_ratio {"n": 60, "mean": 0.19587053096710497, "median": 0.20068359375, "std": 0.054671571728140245, "min": 0.09608785175017158, "max": 0.3629441624365482, "p10": 0.130712890625, "p90": 0.246502569373073}; repeated_ngram_fraction_4 {"n": 60, "mean": 0.4178343941782232, "median": 0.40260931651863024, "std": 0.15448317455961652, "min": 0.13043478260869565, "max": 0.7461103253182461, "p10": 0.24078239608801957, "p90": 0.6454834234190033}; text_repeat_ratio {"n": 60, "mean": 0.049772699219578144, "median": 0.03228580362726702, "std": 0.05812247753473946, "min": 0.0, "max": 0.2761904761904762, "p10": 0.0, "p90": 0.1328842720337608}

## Repeatability / noise floor

### 1. Greedy replay (same deterministic condition, repeated)
- no repeated greedy conditions in this run

### 2. Same-seed sampled replay (same seed + sampler, re-executed)
- no repeated same-seed sampled conditions in this run

### 3. Across-seed stochastic variation (fixed sampler policy, different seeds)
- task/condition groups: 30
- estimable answer diversity: 15 / 30 (proportion 0.5)
- mean distinct observed answers across seeds: 1 (support: 15 tasks)
- mean observed-answer accuracy std across seeds: 0
- any ambiguous seeds (divergent same-seed repeats): False
- note: final-answer diversity/accuracy is defined only over seeds with an observed answer; `—` means unobserved, not zero. No mean is shown without its support count.

### 4. Across-seed token variation (valid even when all answers are censored)
- task/condition groups: 30
- mean first token divergence across seeds: 6.733
- mean reasoning-token std across seeds: 105.3

## Censoring / termination (do not read a capped run as a wrong answer)

- capped_length: 34
- eos_observed: 26
- missing_answer: 33
- natural_eos: 26
- status:censored: 33
- status:correct: 26
- status:incorrect: 1
- think_end_reached: 27

### Budget outcomes per condition
| condition_id | success_at_budget | completion_rate | censored_rate | error_rate | unparseable_rate | cond_acc|completed |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 0.4333 | 0.4333 | 0.55 | 0 | 0 | 1 |

## Family reconciliation (closures vs EOS are distinct)

| condition | family | n | native closures | EOS | censored | no-closure | unparseable | errors | observed answers | correct answers |
|---|---|---|---|---|---|---|---|---|---|---|
| cond-f5e762d3078f | **ALL** | 60 | 27 | 26 | 33 | 0 | 0 | 0 | 26 | 26 |
| cond-f5e762d3078f | arith | 12 | 9 | 9 | 3 | 0 | 0 | 0 | 9 | 9 |
| cond-f5e762d3078f | logic | 12 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 |
| cond-f5e762d3078f | order | 12 | 10 | 10 | 2 | 0 | 0 | 0 | 10 | 10 |
| cond-f5e762d3078f | path | 12 | 1 | 1 | 11 | 0 | 0 | 0 | 1 | 1 |
| cond-f5e762d3078f | sm | 12 | 7 | 6 | 5 | 0 | 0 | 0 | 6 | 6 |

## Censor-aware time-to-closure (Kaplan-Meier / RMST)

- observation horizon tau: 2048.0 tokens (an observation horizon, not a natural stopping threshold)
- event: PRIMARY: native think-end reasoning closure observed
- secondary event: SECONDARY: EOS observed (generation termination), timed at the terminal token, independent of the reasoning-closure time
- analysis unit: unique stochastic draws (same-seed technical repeats collapsed)

| condition | exec | draws (used) | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 60 | 60 (60) | 27 | 33 | — | 1467 |

Secondary endpoint (generation termination via EOS):

| condition | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|
| cond-f5e762d3078f | 26 | 34 | — | 1564 |

### cond-f5e762d3078f by family
| family | n | events | censored | median | RMST |
|---|---|---|---|---|---|
| arith | 12 | 9 | 3 | 653 | 943.8 |
| logic | 12 | 0 | 12 | — | 2048 |
| order | 12 | 10 | 2 | 735 | 961.2 |
| path | 12 | 1 | 11 | — | 1957 |
| sm | 12 | 7 | 5 | 1019 | 1426 |

## Loop-structure diagnostics (descriptive)

Descriptive loop structure with an explicitly documented onset rule; not a validated universal loop detector.

| condition | n | longest_run median (max) | special longest runs | with degeneration onset |
|---|---|---|---|---|
| cond-f5e762d3078f | 60 | 1 (5) | 0 | 2 |

## Prefix probes / trajectory states

- no probes in this run

## Limitations
- Single model revision on a single machine; results do not generalize to other weights or hardware.
- Greedy determinism is a same-process, same-machine replay test.
- Where the reasoning channel was never closed, the trajectory is censored, not a natural stop.
- No causal claims: deployment differences are associations under this protocol.
