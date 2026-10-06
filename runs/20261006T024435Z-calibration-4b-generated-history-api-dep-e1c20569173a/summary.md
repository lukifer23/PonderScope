# PonderScope report — 20261006T024435Z-calibration-4b-generated-history-api-dep-e1c20569173a

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
- created: `20261006T024435Z`

## Machine / runtime environment
- python: `3.12.12`
- platform: `macOS-27.2-arm64-arm-64bit`
- mlx device: `Device(gpu, 0)`
- memory bytes: `19327352832`
- packages: `{'mlx': '0.32.3', 'mlx-lm': '0.32.0', 'mlx-metal': '0.32.3', 'numpy': '2.5.3', 'transformers': '5.18.0', 'tokenizers': '0.23.2', 'huggingface-hub': '1.33.0', 'safetensors': '0.8.0', 'ponderscope': '0.1.0'}`

## Task population
- n tasks: 10  n generations: 30
- spec: `{'name': 'calibration-4b-generated-history-api', 'task_pack': 'tasks-v1', 'prompt_policy': 'pp-v1', 'model_policy': 'qwen35-4b-thinking-api-v1', 'families': ['arith', 'path', 'order', 'logic', 'sm'], 'n_per_family': 2, 'task_seed': 0, 'split': 'calibration', 'greedy_repeats': 0, 'sampled_seeds': [0, 1, 2], 'sampled_repeats_per_seed': 1, 'max_tokens': 2048, 'sampled_temperature': 1.0, 'sampled_top_p': 0.95, 'sampled_top_k': 20, 'sampled_min_p': 0.0, 'sampled_presence_penalty': 1.5, 'sampled_presence_context_size': 20, 'sampled_presence_scope': 'generated_history', 'sampled_repetition_penalty': 1.0, 'sampled_repetition_context_size': 20, 'sampled_frequency_penalty': 0.0, 'sampled_frequency_context_size': 20, 'probe': False, 'n_probes': 4, 'probe_max_tokens': 64, 'capture_entropy': True, 'capture_top_k': 5, 'capture_logprob_digest': False, 'capture_level': 'minimal', 'notes': 'Stage B model/scale control: Qwen3.5-4B native BF16, same tasks-v1 calibration presentations and pp-v1, same generated-history thinking semantics as Stage A2, 2048 horizon, seeds {0,1,2} x ONE primary execution = 30 stochastic draws. Not the central deployment-drift experiment.'}`

## Configurations
| condition_id | mode | exec | draws | success_at_budget (95% CI) | completion_rate | censored_rate | conditional_acc_given_completed | reasoning tokens mean | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| cond-f5e762d3078f | sampled | 30 | 30 | 0.6333 [0.3667, 0.8667] | 0.6333 | 0.3333 | 1 | 1368 | 13.42 |

## Reasoning-length distribution

### cond-f5e762d3078f (sampled)

```
     380-463         2 ########
     463-547         2 ########
     547-630         1 ####
     630-714         2 ########
     714-797         0 
     797-880         1 ####
     880-964         3 ############
     964-1047        2 ########
    1047-1131        0 
    1131-1214        0 
    1214-1297        0 
    1297-1381        1 ####
    1381-1464        1 ####
    1464-1548        0 
    1548-1631        3 ############
    1631-1714        0 
    1714-1798        1 ####
    1798-1881        1 ####
    1881-1965        0 
    1965-2048       10 ########################################
```
- {"n": 30, "mean": 1368.1, "median": 1513.5, "std": 622.5477879455215, "min": 380.0, "max": 2048.0, "p10": 520.1, "p90": 2048.0}

## Latency / throughput
- cond-f5e762d3078f: wall_ms {"n": 30, "mean": 112625.31173070165, "median": 125431.11637549737, "std": 42966.74720620062, "min": 34855.894457999966, "max": 159576.54316700064, "p10": 47655.69132530509, "p90": 157259.0280670047}; tok/s {"n": 30, "mean": 13.421247756035505, "median": 13.575240104702376, "std": 0.4157175251922901, "min": 12.448091171705725, "max": 13.939735542142131, "p10": 12.828395467283896, "p90": 13.872227790110037}; ttft_ms {"n": 30, "mean": 416.6259223682573, "median": 366.3137914991239, "std": 170.62582438508758, "min": 257.5093750056112, "max": 1181.304874990019, "p10": 263.6071789136622, "p90": 573.1080506098806}

## Repetition
- cond-f5e762d3078f: unique_token_ratio {"n": 30, "mean": 0.21500033122692036, "median": 0.2140539329218885, "std": 0.06119062378694919, "min": 0.11813904861474124, "max": 0.337152209492635, "p10": 0.14909150823583744, "p90": 0.2965128660159716}; repeated_ngram_fraction_4 {"n": 30, "mean": 0.3912788721094123, "median": 0.3282050403819675, "std": 0.16118204503928688, "min": 0.15625, "max": 0.7373233582709892, "p10": 0.22998248700320173, "p90": 0.6167194000640456}; text_repeat_ratio {"n": 30, "mean": 0.049860965938031354, "median": 0.030117753623188415, "std": 0.053101933099819504, "min": 0.0, "max": 0.1728395061728395, "p10": 0.0, "p90": 0.14638554216867475}

## Repeatability / noise floor

### 1. Greedy replay (same deterministic condition, repeated)
- no repeated greedy conditions in this run

### 2. Same-seed sampled replay (same seed + sampler, re-executed)
- no repeated same-seed sampled conditions in this run

### 3. Across-seed stochastic variation (fixed sampler policy, different seeds)
- task/condition groups: 10
- estimable answer diversity: 7 / 10 (proportion 0.7)
- mean distinct observed answers across seeds: 1 (support: 7 tasks)
- mean observed-answer accuracy std across seeds: 0
- any ambiguous seeds (divergent same-seed repeats): False
- note: final-answer diversity/accuracy is defined only over seeds with an observed answer; `—` means unobserved, not zero. No mean is shown without its support count.

### 4. Across-seed token variation (valid even when all answers are censored)
- task/condition groups: 10
- mean first token divergence across seeds: 8.3
- mean reasoning-token std across seeds: 250.6

## Censoring / termination (do not read a capped run as a wrong answer)

- capped_length: 11
- eos_observed: 19
- missing_answer: 10
- natural_eos: 19
- status:censored: 10
- status:correct: 19
- status:incorrect: 1
- think_end_reached: 20

### Budget outcomes per condition
| condition_id | success_at_budget | completion_rate | censored_rate | error_rate | unparseable_rate | cond_acc|completed |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 0.6333 | 0.6333 | 0.3333 | 0 | 0 | 1 |

## Censor-aware time-to-closure (Kaplan-Meier / RMST)

- observation horizon tau: 2048.0 tokens (an observation horizon, not a natural stopping threshold)
- event: PRIMARY: native think-end reasoning closure observed
- secondary event: SECONDARY: EOS observed (generation termination)
- analysis unit: unique stochastic draws (same-seed technical repeats collapsed)

| condition | exec | draws (used) | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 30 | 30 (30) | 20 | 10 | 1718 | 1512 |

Secondary endpoint (generation termination via EOS):

| condition | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|
| cond-f5e762d3078f | 19 | 11 | 1718 | 1512 |

### cond-f5e762d3078f by family
| family | n | events | censored | median | RMST |
|---|---|---|---|---|---|
| arith | 6 | 6 | 0 | 589 | 880 |
| logic | 6 | 0 | 6 | — | 2048 |
| order | 6 | 6 | 0 | 1157 | 1233 |
| path | 6 | 2 | 4 | — | 1778 |
| sm | 6 | 6 | 0 | 1289 | 1620 |

## Loop-structure diagnostics (descriptive)

Descriptive loop structure with an explicitly documented onset rule; not a validated universal loop detector.

| condition | n | longest_run median (max) | special longest runs | with degeneration onset |
|---|---|---|---|---|
| cond-f5e762d3078f | 30 | 1 (3) | 0 | 1 |

## Prefix probes / trajectory states

- no probes in this run

## Limitations
- Single model revision on a single machine; results do not generalize to other weights or hardware.
- Greedy determinism is a same-process, same-machine replay test.
- Where the reasoning channel was never closed, the trajectory is censored, not a natural stop.
- No causal claims: deployment differences are associations under this protocol.
