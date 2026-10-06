"""Phase 1.3B tests: penalty semantics/scope, stochastic-draw units, endpoint
semantics, estimability support, and honest run-level descriptions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ponderscope.analysis.analyze import (
    _is_generation_termination,
    _is_reasoning_closure,
    across_seed_variation,
)
from ponderscope.analysis.draws import (
    collapse_to_stochastic_draws,
    draw_summary,
)
from ponderscope.analysis.survival import rmst_delta_clustered, survival_summary
from ponderscope.backends.mlx_backend import MlxBackend
from ponderscope.backends.processors import make_generated_history_presence_penalty
from ponderscope.config.identity import DecodingPolicy, TrialIdentity
from ponderscope.config.policies import load_model_policy
from ponderscope.config.schema import ExperimentSpec
from ponderscope.experiment import run_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN = {"version": "test", "git_sha": "0" * 40, "tracked_dirty": False}

HISTORICAL_MLX_WINDOW_COND = "cond-74a29a115886"


def _hist_policy(**overrides) -> DecodingPolicy:
    base = {
        "mode": "sampled",
        "max_tokens": 2048,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "presence_context_size": 20,
        "repetition_penalty": 1.0,
        "repetition_context_size": 20,
        "frequency_penalty": 0.0,
        "frequency_context_size": 20,
        "context": {"enable_thinking": True},
    }
    base.update(overrides)
    return DecodingPolicy(**base)


def _sampled(**overrides) -> DecodingPolicy:
    base = {
        "mode": "sampled",
        "max_tokens": 100,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    }
    base.update(overrides)
    return DecodingPolicy(**base)


def _record(
    *,
    task_id: str,
    presentation_id: str,
    condition_id: str,
    mode: str,
    seed: int | None,
    repeat: int,
    token_ids: list[int],
    answer: str | None = None,
    correct: bool = False,
    think_end: bool = False,
    eos: bool = False,
    stochastic_draw_id: str | None = None,
) -> dict:
    rec: dict = {
        "task_id": task_id,
        "presentation_id": presentation_id,
        "family": "arith",
        "condition_id": condition_id,
        "condition": {"mode": mode, "seed": seed, "repeat": repeat},
        "trace": {"token_ids": token_ids},
        "answer_normalized": answer,
        "correct": correct,
        "termination": {
            "think_end_reached": think_end,
            "terminated_by_eos": eos,
            "capped": not eos and not think_end,
            "missing_answer": answer is None,
        },
        "total_tokens": len(token_ids),
    }
    if stochastic_draw_id is not None:
        rec["stochastic_draw_id"] = stochastic_draw_id
    return rec


# --------------------------------------------------------------------------- #
# Generated-history presence semantics
# --------------------------------------------------------------------------- #
def test_generated_history_presence_excludes_prompt_tokens():
    import mlx.core as mx

    proc = make_generated_history_presence_penalty(1.5, prompt_len=2)
    # tokens: prompt [1, 2], generated [5]
    logits = mx.zeros((1, 10))
    out = proc(mx.array([1, 2, 5]), logits)
    mx.eval(out)
    values = out.tolist()[0]
    assert values[1] == 0.0  # prompt token: not penalized
    assert values[2] == 0.0  # prompt token: not penalized
    assert values[5] == -1.5  # generated token: penalized


def test_generated_history_presence_persists_beyond_window():
    import mlx.core as mx

    tokens = [1, 2, 5] + list(range(100, 130))  # 5 is 30 positions back
    out = make_generated_history_presence_penalty(1.5, 2)(mx.array(tokens), mx.zeros((1, 200)))
    mx.eval(out)
    assert out.tolist()[0][5] == -1.5


def test_mlx_window_loses_penalty_after_leaving_window():
    import mlx.core as mx
    from mlx_lm.sample_utils import make_presence_penalty

    tokens = [1, 2, 5] + list(range(100, 130))
    out = make_presence_penalty(1.5, 20)(mx.array(tokens), mx.zeros((1, 200)))
    mx.eval(out)
    values = out.tolist()[0]
    assert values[5] == 0.0  # outside the 20-token window
    assert values[129] == -1.5  # inside the window


def test_generated_history_penalty_scales_without_duplicate_double_count():
    import mlx.core as mx

    # token 5 appears three times; presence penalty is applied once (set semantics)
    out = make_generated_history_presence_penalty(1.5, 0)(mx.array([5, 5, 5]), mx.zeros((1, 10)))
    mx.eval(out)
    assert out.tolist()[0][5] == -1.5


def test_generated_history_zero_penalty_is_noop():
    import mlx.core as mx

    out = make_generated_history_presence_penalty(0.0, 2)(mx.array([1, 2, 5]), mx.zeros((1, 10)))
    mx.eval(out)
    assert all(abs(v) < 1e-9 for v in out.tolist()[0])


# --------------------------------------------------------------------------- #
# Presence scope participates in condition identity
# --------------------------------------------------------------------------- #
def test_presence_scope_changes_condition_id():
    mlx = _sampled(seed=0, presence_penalty=1.5, presence_context_size=20)
    gen = _sampled(
        seed=0,
        presence_penalty=1.5,
        presence_context_size=20,
        presence_scope="generated_history",
    )
    assert mlx.condition_id != gen.condition_id


def test_mlx_window_scope_preserves_historical_condition_id():
    assert _hist_policy().condition_id == HISTORICAL_MLX_WINDOW_COND


def test_generated_history_ignores_context_size_in_identity():
    a = _sampled(presence_penalty=1.5, presence_context_size=20, presence_scope="generated_history")
    b = _sampled(presence_penalty=1.5, presence_context_size=99, presence_scope="generated_history")
    assert a.condition_id == b.condition_id


def test_zero_penalty_preserves_old_identity_across_scopes():
    base = _sampled(seed=0)
    zero_mlx = _sampled(seed=0, presence_penalty=0.0, presence_context_size=20)
    zero_gen = _sampled(
        seed=0,
        presence_penalty=0.0,
        presence_context_size=20,
        presence_scope="generated_history",
    )
    assert base.condition_id == zero_mlx.condition_id == zero_gen.condition_id


def test_backend_builds_generated_history_processor():
    backend = MlxBackend()
    procs, meta = backend._prepare_logits_processors(
        _sampled(
            presence_penalty=1.5,
            presence_context_size=20,
            presence_scope="generated_history",
            repetition_penalty=1.0,
        ),
        prompt_len=7,
    )
    assert len(procs) == 1
    assert meta["presence_semantics"] == "generated_history"
    assert meta["presence_scope"] == "generated_history"
    assert meta["equivalence_label"] == "qwen-generated-history-presence-v1"
    assert meta["base_prompt_tokens"] == 7


def test_backend_mlx_window_processor_label_unchanged():
    backend = MlxBackend()
    procs, meta = backend._prepare_logits_processors(
        _sampled(presence_penalty=1.5, presence_context_size=20)
    )
    assert len(procs) == 1
    assert meta["presence_semantics"] == "mlx_window"
    assert meta["equivalence_label"] == "qwen-upstream-profile-on-mlx"


# --------------------------------------------------------------------------- #
# Model-policy profiles (T=1.0 API and T=0.6 benchmark)
# --------------------------------------------------------------------------- #
def test_api_v2_profile_provenance_and_scope():
    profile = load_model_policy(
        "qwen35-08b-thinking-api-v2", policy_dir=REPO_ROOT / "model_policies"
    )
    rec = profile.sampling_kwargs()
    assert rec["temperature"] == 1.0
    assert rec["presence_penalty"] == 1.5
    assert rec["repetition_penalty"] == 1.0
    assert profile.mlx_mapping["presence_scope"] == "generated_history"
    assert profile.condition_label == "qwen-generated-history-presence-v1"
    assert profile.provenance["readme_sha256"]


def test_benchmark_profile_is_t06_not_collapsed_with_api():
    api = load_model_policy("qwen35-08b-thinking-api-v2", policy_dir=REPO_ROOT / "model_policies")
    bench = load_model_policy(
        "qwen35-08b-thinking-benchmark-v1", policy_dir=REPO_ROOT / "model_policies"
    )
    assert api.sampling_kwargs()["temperature"] == 1.0
    assert bench.sampling_kwargs()["temperature"] == 0.6
    assert bench.sampling_kwargs()["presence_penalty"] == 1.5
    assert bench.mlx_mapping["presence_scope"] == "generated_history"
    assert "benchmark_note" in bench.raw["quotes"]


# --------------------------------------------------------------------------- #
# Stochastic-draw hierarchy
# --------------------------------------------------------------------------- #
def test_stochastic_draw_id_stable_across_repeats_but_trial_changes():
    cond = _sampled(seed=0, presence_penalty=1.5, presence_scope="generated_history")
    t0 = TrialIdentity("t", cond, 0, 0, presentation_id="pres-x")
    t1 = TrialIdentity("t", cond, 0, 1, presentation_id="pres-x")
    assert t0.stochastic_draw_id == t1.stochastic_draw_id
    assert t0.trial_id != t1.trial_id


def test_stochastic_draw_id_differs_by_seed_and_presentation():
    cond = _sampled(seed=0, presence_penalty=1.5, presence_scope="generated_history")
    a = TrialIdentity("t", cond, 0, 0, presentation_id="pres-a")
    b = TrialIdentity("t", cond, 1, 0, presentation_id="pres-a")
    c = TrialIdentity("t", cond, 0, 0, presentation_id="pres-b")
    assert len({a.stochastic_draw_id, b.stochastic_draw_id, c.stochastic_draw_id}) == 3


def _draw_records(identical: bool) -> list[dict]:
    first = _record(
        task_id="t1",
        presentation_id="pres-a",
        condition_id="cond-x",
        mode="sampled",
        seed=0,
        repeat=0,
        token_ids=[1, 2, 3, 4],
        answer="7",
        correct=True,
        think_end=True,
        eos=True,
    )
    second = dict(first)
    second["condition"] = {"mode": "sampled", "seed": 0, "repeat": 1}
    second["trace"] = {"token_ids": [1, 2, 3, 9] if not identical else [1, 2, 3, 4]}
    return [first, second]


def test_identical_technical_replicates_collapse_to_one_draw():
    draws = collapse_to_stochastic_draws(_draw_records(identical=True))
    assert len(draws) == 1
    assert draws[0]["n_executions"] == 2
    assert draws[0]["token_identical"] is True
    assert draws[0]["ambiguous"] is False
    assert draws[0]["record"] is not None


def test_ambiguous_technical_replicates_flagged_and_excluded():
    draws = collapse_to_stochastic_draws(_draw_records(identical=False))
    assert len(draws) == 1
    draw = draws[0]
    assert draw["ambiguous"] is True
    assert draw["record"] is None  # fail closed: never silently select repeat 0
    summary = draw_summary(draws)
    assert summary["n_unique_draws"] == 1
    assert summary["n_primary_draws"] == 0
    assert summary["n_ambiguous_draws"] == 1


def test_km_rmst_ignore_duplicate_technical_repeats():
    records = _draw_records(identical=True)
    draws = collapse_to_stochastic_draws(records)
    reps = [d["record"] for d in draws if d["record"] is not None]
    s = survival_summary([r["total_tokens"] for r in reps], [True] * len(reps), tau=10)
    assert s["n_at_risk_initial"] == 1  # two executions -> one draw


def test_unequal_technical_repeats_do_not_change_rmst_delta():
    a = {"t1": [(10.0, True)], "t2": [(20.0, False)]}
    b = {"t1": [(10.0, True)] * 3, "t2": [(20.0, False)] * 3}
    d = rmst_delta_clustered(a, b, tau=20, n_resamples=200, seed=0)
    assert d["point"] == 0.0


# --------------------------------------------------------------------------- #
# Survival endpoint semantics
# --------------------------------------------------------------------------- #
def test_reasoning_closure_requires_native_think_end():
    eos_only = _record(
        task_id="t",
        presentation_id="p",
        condition_id="c",
        mode="sampled",
        seed=0,
        repeat=0,
        token_ids=[1, 2],
        think_end=False,
        eos=True,
    )
    assert _is_reasoning_closure(eos_only) is False
    assert _is_generation_termination(eos_only) is True


def test_native_think_end_is_reasoning_closure():
    closed = _record(
        task_id="t",
        presentation_id="p",
        condition_id="c",
        mode="sampled",
        seed=0,
        repeat=0,
        token_ids=[1, 2],
        think_end=True,
        eos=True,
    )
    assert _is_reasoning_closure(closed) is True


# --------------------------------------------------------------------------- #
# Across-seed estimability support
# --------------------------------------------------------------------------- #
def test_across_seed_estimability_reports_support_counts():
    records = []
    for task, answers in (("t1", ["7", "8"]), ("t2", [None, None])):
        for seed, answer in enumerate(answers):
            records.append(
                _record(
                    task_id=task,
                    presentation_id=f"pres-{task}",
                    condition_id="cond-x",
                    mode="sampled",
                    seed=seed,
                    repeat=0,
                    token_ids=[seed],
                    answer=answer,
                    correct=answer is not None,
                    think_end=answer is not None,
                    eos=answer is not None,
                )
            )
    out = across_seed_variation(records)
    assert out["n_task_conditions_total"] == 2
    assert out["n_task_conditions_with_estimable_answer_diversity"] == 1
    assert out["proportion_estimable"] == 0.5


# --------------------------------------------------------------------------- #
# Run-level deployment description
# --------------------------------------------------------------------------- #
def test_sampled_only_run_description_never_claims_greedy(tmp_path, fake_backend, monkeypatch):
    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda name: fake_backend)
    spec = ExperimentSpec(
        name="sampled-only",
        task_pack="tasks-v1",
        families=["arith"],
        n_per_family=1,
        greedy_repeats=0,
        sampled_seeds=[0, 1],
        sampled_repeats_per_seed=1,
        sampled_presence_penalty=1.5,
        sampled_presence_scope="generated_history",
        max_tokens=16,
    )
    result = run_experiment(
        spec,
        model_repo="fake/model",
        model_revision="deadbeef",
        runs_dir=tmp_path,
        code_state=CLEAN,
    )
    manifest = result.store.manifest
    assert "greedy" not in manifest["deployment_description"]
    conditions = manifest["conditions"]
    assert len(conditions) == 1
    assert "sampled" in conditions[0]["description"]
    # the description must match the condition actually declared/executed
    recorded = {r["condition_id"] for r in result.store.read_traces()}
    assert recorded == {conditions[0]["condition_id"]}


def test_run_records_stochastic_draw_id(tmp_path, fake_backend, monkeypatch):
    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda name: fake_backend)
    spec = ExperimentSpec(
        name="draws",
        task_pack="tasks-v1",
        families=["arith"],
        n_per_family=1,
        greedy_repeats=0,
        sampled_seeds=[0],
        sampled_repeats_per_seed=1,
        max_tokens=16,
    )
    result = run_experiment(
        spec,
        model_repo="fake/model",
        model_revision="deadbeef",
        runs_dir=tmp_path,
        code_state=CLEAN,
    )
    traces = result.store.read_traces()
    assert traces
    assert all(r.get("stochastic_draw_id", "").startswith("draw-") for r in traces)
    assert len({r["stochastic_draw_id"] for r in traces}) == len(traces)


def test_spec_files_declare_generated_history_scope():
    import json

    for name in (
        "calibration-08b-generated-history-api",
        "calibration-08b-generated-history-benchmark",
    ):
        data = json.loads((REPO_ROOT / "specs" / f"{name}.json").read_text())
        assert data["sampled_presence_scope"] == "generated_history"
    with pytest.raises(ValueError):
        DecodingPolicy(mode="sampled", max_tokens=1, presence_scope="bogus")
