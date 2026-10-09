# PonderScope report — 20261008T235847Z-phase1-6-replication-pilot-bf16-pilot-dep-e1c20569173a

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
- created: `20261008T235847Z`

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
| cond-f5e762d3078f | sampled | 5 | 5 | 0.8 [0.4, 1] | 0.8 | 0.2 | 1 | 1050 | 14.04 |

## Reasoning-length distribution

### cond-f5e762d3078f (sampled)

```
     458-538         1 ########################################
     538-617         0 
     617-696         0 
     696-776         0 
     776-856         1 ########################################
     856-935         1 ########################################
     935-1014        1 ########################################
    1014-1094        0 
    1094-1174        0 
    1174-1253        0 
    1253-1332        0 
    1332-1412        0 
    1412-1492        0 
    1492-1571        0 
    1571-1650        0 
    1650-1730        0 
    1730-1810        0 
    1810-1889        0 
    1889-1968        0 
    1968-2048        1 ########################################
```
- {"n": 5, "mean": 1049.6, "median": 930.0, "std": 595.003613434406, "min": 458.0, "max": 2048.0, "p10": 604.0, "p90": 1624.4}

## Latency / throughput
- cond-f5e762d3078f: wall_ms {"n": 5, "mean": 86471.17535839934, "median": 84455.75475000078, "std": 37933.38971167362, "min": 38115.136708001955, "max": 142443.2295000006, "p10": 51734.428524799296, "p90": 123537.29353359959}; tok/s {"n": 5, "mean": 14.043137887556941, "median": 14.010181941388424, "std": 0.28647347841643606, "min": 13.711095614286418, "max": 14.37765773205803, "p10": 13.758548018095595, "p90": 14.341405649731808}; ttft_ms {"n": 5, "mean": 677.0283417994506, "median": 410.3631670004688, "std": 633.5928497415817, "min": 340.8080839944887, "max": 1807.086665998213, "p10": 348.6620335970656, "p90": 1270.8283332001884}

## Repetition
- cond-f5e762d3078f: unique_token_ratio {"n": 5, "mean": 0.22221340709305099, "median": 0.21746575342465754, "std": 0.04404926699000034, "min": 0.16015325670498085, "max": 0.28277153558052437, "p10": 0.18222192492502537, "p90": 0.2638035463483146}; repeated_ngram_fraction_4 {"n": 5, "mean": 0.37947210998870623, "median": 0.3389830508474576, "std": 0.13001954665476725, "min": 0.29377431906614787, "max": 0.6075268817204301, "p10": 0.29694918801670583, "p90": 0.5066620517790392}; text_repeat_ratio {"n": 5, "mean": 0.04249509700389222, "median": 0.02777777777777779, "std": 0.046793357590838365, "min": 0.007633587786259555, "max": 0.12222222222222223, "p10": 0.009125607217210275, "p90": 0.09072463768115943}

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

- capped_length: 1
- eos_observed: 4
- missing_answer: 1
- natural_eos: 4
- status:censored: 1
- status:correct: 4
- think_end_reached: 4

### Budget outcomes per condition
| condition_id | success_at_budget | completion_rate | censored_rate | error_rate | unparseable_rate | cond_acc|completed |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 0.8 | 0.8 | 0.2 | 0 | 0 | 1 |

## Family reconciliation (closures vs EOS are distinct)

| condition | family | n | native closures | EOS | censored | no-closure | unparseable | errors | observed answers | correct answers |
|---|---|---|---|---|---|---|---|---|---|---|
| cond-f5e762d3078f | **ALL** | 5 | 4 | 4 | 1 | 0 | 0 | 0 | 4 | 4 |
| cond-f5e762d3078f | arith | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| cond-f5e762d3078f | logic | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| cond-f5e762d3078f | order | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| cond-f5e762d3078f | path | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| cond-f5e762d3078f | sm | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |

## Censor-aware time-to-closure (Kaplan-Meier / RMST)

- observation horizon tau: 2048.0 tokens (an observation horizon, not a natural stopping threshold)
- event: PRIMARY: native think-end reasoning closure observed
- secondary event: SECONDARY: EOS observed (generation termination), timed at the terminal token, independent of the reasoning-closure time
- analysis unit: unique stochastic draws (same-seed technical repeats collapsed)

| condition | exec | draws (used) | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 5 | 5 (5) | 4 | 1 | 931 | 1050 |

Secondary endpoint (generation termination via EOS):

| condition | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|
| cond-f5e762d3078f | 4 | 1 | 1168 | 1217 |

## Loop-structure diagnostics (descriptive)

Descriptive loop structure with an explicitly documented onset rule; not a validated universal loop detector.

| condition | n | longest_run median (max) | special longest runs | with degeneration onset |
|---|---|---|---|---|
| cond-f5e762d3078f | 5 | 1 (3) | 0 | 0 |

## Prefix probes / trajectory states

- no probes in this run

## Limitations
- Single model revision on a single machine; results do not generalize to other weights or hardware.
- Greedy determinism is a same-process, same-machine replay test.
- Where the reasoning channel was never closed, the trajectory is censored, not a natural stop.
- No causal claims: deployment differences are associations under this protocol.
