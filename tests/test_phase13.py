"""Phase 1.3 tests: presentation identity, logits-processor conditions,
censor-aware comparison, survival, loop diagnostics, and model-policy profiles.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ponderscope.analysis.analyze import prefix_invariance
from ponderscope.analysis.compare import _METRICS, compare_configs
from ponderscope.analysis.survival import (
    kaplan_meier,
    rmst,
    rmst_delta_clustered,
    survival_summary,
)
from ponderscope.backends.mlx_backend import MlxBackend
from ponderscope.config.identity import (
    DecodingPolicy,
    TrialIdentity,
    presentation_id,
)
from ponderscope.config.policies import load_model_policy
from ponderscope.config.schema import ExperimentSpec
from ponderscope.experiment import run_experiment, verify_capture_equivalence
from ponderscope.reasoning.metrics import compute_loop_diagnostics
from ponderscope.tasks import generate_pack, generate_pack_metadata, make_task

CLEAN = {"version": "test", "git_sha": "0" * 40, "tracked_dirty": False}
REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Presentation / stimulus identity
# --------------------------------------------------------------------------- #
def test_prompt_policies_share_structure_but_not_presentation():
    p1 = make_task("arith", 0, prompt_policy="pp-v1")
    p2 = make_task("arith", 0, prompt_policy="pp-v2")
    assert p1.task_id == p2.task_id  # structural identity is wording-independent
    pres1 = presentation_id(p1.task_id, p1.prompt_policy, p1.prompt)
    pres2 = presentation_id(p2.task_id, p2.prompt_policy, p2.prompt)
    assert pres1 != pres2
    assert pres1.startswith("pres-")


def test_identical_rendered_prompt_reproduces_presentation_id():
    p = make_task("logic", 1, prompt_policy="pp-v1")
    a = presentation_id(p.task_id, "pp-v1", p.prompt)
    b = presentation_id(p.task_id, "pp-v1", p.prompt)
    assert a == b
    c = presentation_id(p.task_id, "pp-v1", p.prompt + " ")
    assert a != c


def test_trial_id_changes_with_presentation():
    cond = DecodingPolicy(mode="greedy", max_tokens=8)
    a = TrialIdentity("t", cond, None, 0, presentation_id="pres-a")
    b = TrialIdentity("t", cond, None, 0, presentation_id="pres-b")
    assert a.trial_id != b.trial_id
    # legacy evidence with no presentation id still reproduces its historical id
    legacy = TrialIdentity("t", cond, None, 0)
    assert "presentation_id" not in legacy.to_dict() or legacy.to_dict()["presentation_id"] is None


def test_task_pack_metadata_records_structural_and_presentation_hashes():
    tasks = generate_pack(n_per_family=1, split="calibration")
    meta = generate_pack_metadata(
        tasks,
        families=None,
        n_per_family=1,
        split="calibration",
        seed=0,
        pack="tasks-v1",
    )
    assert meta["task_ids_sha256"] == meta["structural_task_ids_sha256"]
    assert len(meta["presentation_ids"]) == len(tasks)
    assert meta["presentation_ids_sha256"]


# --------------------------------------------------------------------------- #
# Condition identity for logits processors
# --------------------------------------------------------------------------- #
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


def test_zero_and_none_penalties_preserve_prior_condition_identity():
    base = _sampled(seed=0)
    zero = _sampled(
        seed=0,
        presence_penalty=0.0,
        presence_context_size=20,
        repetition_penalty=1.0,
        repetition_context_size=20,
        frequency_penalty=0.0,
        frequency_context_size=20,
    )
    assert base.condition_id == zero.condition_id


def test_penalty_changes_condition_identity():
    base = _sampled(seed=0)
    pen = _sampled(seed=0, presence_penalty=1.5, presence_context_size=20)
    assert base.condition_id != pen.condition_id


def test_penalty_context_size_changes_condition_identity():
    p20 = _sampled(seed=0, presence_penalty=1.5, presence_context_size=20)
    p40 = _sampled(seed=0, presence_penalty=1.5, presence_context_size=40)
    assert p20.condition_id != p40.condition_id


def test_seed_does_not_change_condition_identity():
    a = _sampled(seed=0, presence_penalty=1.5)
    b = _sampled(seed=7, presence_penalty=1.5)
    assert a.condition_id == b.condition_id
    assert a.to_dict()["seed"] == 0


def test_backend_builds_processors_and_records_semantics():
    backend = MlxBackend()
    procs, meta = backend._prepare_logits_processors(
        _sampled(presence_penalty=1.5, presence_context_size=20, repetition_penalty=1.0)
    )
    assert len(procs) >= 1
    assert meta["presence_penalty"] == 1.5
    assert meta["framework"] == "mlx-lm"
    assert meta["equivalence_label"] == "qwen-upstream-profile-on-mlx"
    assert meta["processor_order"] == [
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    ]


def test_backend_zero_penalties_yield_no_processors():
    backend = MlxBackend()
    procs, meta = backend._prepare_logits_processors(_sampled())
    assert procs == []
    assert meta["n_processors"] == 0


def test_presence_penalty_processor_affects_logits():
    import mlx.core as mx
    from mlx_lm.sample_utils import make_logits_processors

    procs = make_logits_processors(presence_penalty=1.5, presence_context_size=20)
    logits = mx.zeros((1, 5))
    out = procs[0](mx.array([2, 2, 3]), logits)
    mx.eval(out)
    values = out.tolist()[0]
    assert values[2] == -1.5
    assert values[3] == -1.5
    assert values[0] == 0.0


# --------------------------------------------------------------------------- #
# Model-policy profile provenance
# --------------------------------------------------------------------------- #
def test_upstream_profile_loads_with_provenance():
    profile = load_model_policy(
        "qwen35-08b-thinking-upstream-v1", policy_dir=REPO_ROOT / "model_policies"
    )
    assert profile.repo_id == "Qwen/Qwen3.5-0.8B"
    assert profile.revision == "2fc06364715b967f1860aea9cf38778875588b17"
    rec = profile.sampling_kwargs()
    assert rec == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "enable_thinking": True,
    }
    assert profile.provenance["readme_sha256"]
    assert profile.condition_label == "qwen-upstream-profile-on-mlx"
    assert "thinking loops" in profile.raw["quotes"]["loop_warning"]
    assert profile.to_dict()["profile_id"] == "qwen35-08b-thinking-upstream-v1"


def test_run_refuses_model_policy_revision_mismatch(tmp_path, fake_backend, monkeypatch):
    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda name: fake_backend)
    spec = ExperimentSpec(
        name="policy",
        task_pack="tasks-v1",
        families=["arith"],
        n_per_family=1,
        max_tokens=8,
        model_policy="qwen35-08b-thinking-upstream-v1",
    )
    with pytest.raises(ValueError, match="not"):
        run_experiment(
            spec,
            model_repo="fake/model",
            model_revision="deadbeef",
            runs_dir=tmp_path,
            code_state=CLEAN,
        )


# --------------------------------------------------------------------------- #
# Cap-prefix invariance
# --------------------------------------------------------------------------- #
def test_prefix_invariance_single_cap_is_insufficient_data():
    out = prefix_invariance({4096: [1, 2, 3]})
    assert out["exact_prefix"] is None
    assert out["status"] == "insufficient_data"
    assert out["n_comparisons"] == 0


def test_prefix_invariance_two_caps_reports_comparisons():
    out = prefix_invariance({2048: [1, 2, 3], 4096: [1, 2, 3, 4]})
    assert out["exact_prefix"] is True
    assert out["status"] == "ok"
    assert out["n_comparisons"] == 1


def test_prefix_invariance_detects_divergence():
    out = prefix_invariance({2048: [1, 2, 9], 4096: [1, 2, 3]})
    assert out["exact_prefix"] is False
    assert out["status"] == "divergent"


def test_run_cap_prefix_invariance_marks_insufficient(fake_backend):
    from ponderscope.calibration import run_cap_prefix_invariance

    prompts = [{"task_id": "t1", "family": "arith", "token_ids": [10, 20]}]
    out = run_cap_prefix_invariance(fake_backend, prompts, [4096])
    assert out["status"] == "insufficient_data"
    assert out["all_exact_prefix"] is None
    assert out["n_comparisons"] == 0


# --------------------------------------------------------------------------- #
# Survival analysis
# --------------------------------------------------------------------------- #
def test_kaplan_meier_no_censoring():
    km = kaplan_meier([10, 20, 30, 40], [True, True, True, True])
    assert km["n_events"] == 4 and km["n_censored"] == 0
    assert km["survival"][-1] == 0.0
    assert km["median"] == 20  # S drops to 0.5 at t=20


def test_kaplan_meier_all_censored():
    km = kaplan_meier([100, 100, 100], [False, False, False])
    assert km["n_events"] == 0 and km["n_censored"] == 3
    assert km["survival"] == [1.0]
    assert km["median"] is None
    assert rmst([100, 100, 100], [False, False, False], 100) == 100.0


def test_kaplan_meier_mixed_and_tied():
    km = kaplan_meier([10, 10, 20, 30], [True, False, True, False])
    assert km["n_events"] == 2 and km["n_censored"] == 2
    assert km["at_risk"][0] == 4
    assert km["events"][0] == 1 and km["censored"][0] == 1


def test_rmst_and_median_event_exactly_at_horizon():
    km = kaplan_meier([50], [True])
    assert km["median"] == 50
    assert rmst([50], [True], 50) == 50.0  # S=1 on [0,50)
    assert rmst([50], [True], 100) == 50.0


def test_survival_summary_reports_uncertainty_note():
    s = survival_summary([100], [True], tau=100)
    assert s["n_events"] == 1
    assert "uncertainty is enormous" in s["note"]


def test_rmst_delta_clustered_identical_is_zero():
    a = {"t1": [(10.0, True)], "t2": [(20.0, False)]}
    d = rmst_delta_clustered(a, a, tau=20, n_resamples=200, seed=0)
    assert d["point"] == 0.0
    assert d["excludes_zero"] is False


# --------------------------------------------------------------------------- #
# Loop diagnostics
# --------------------------------------------------------------------------- #
def test_longest_run_and_motif_detected():
    ids = [1, 2, 3] + [7] * 10 + [4, 5, 6]
    diag = compute_loop_diagnostics(ids, decode=lambda t: "x", motif_n=2)
    assert diag.longest_run_length == 10
    assert diag.longest_run_token_id == 7
    assert diag.longest_run_start == 3
    assert diag.longest_run_end == 12
    assert diag.top_motif_count >= 2


def test_special_token_loop_reported():
    ids = [1, 2] + [99] * 6
    diag = compute_loop_diagnostics(ids, decode=lambda t: "<|im_end|>", special_token_ids={99})
    assert diag.longest_run_is_special is True
    assert diag.longest_run_token_piece == "<|im_end|>"


def test_degeneration_onset_descriptive():
    ids = [1, 2, 3, 4] + [5, 6, 5, 6] * 40
    diag = compute_loop_diagnostics(ids, window=16, threshold=0.5)
    assert diag.degeneration_onset_index is not None
    assert "descriptive" in diag.onset_rule


# --------------------------------------------------------------------------- #
# Censor-aware comparison
# --------------------------------------------------------------------------- #
def _run(tmp_path, fake_backend, monkeypatch, *, name, prompt_policy="pp-v1", spec_override=None):
    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda backend_name: fake_backend)
    spec = ExperimentSpec(
        name=name,
        task_pack="tasks-v1",
        prompt_policy=prompt_policy,
        families=["arith"],
        n_per_family=2,
        greedy_repeats=1,
        max_tokens=16,
    )
    if spec_override:
        spec = replace(spec, **spec_override)
    return run_experiment(
        spec,
        model_repo="fake/model",
        model_revision="deadbeef",
        runs_dir=tmp_path,
        code_state=CLEAN,
    )


def test_compare_uses_censor_aware_metric_names(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, name="a")
    b = _run(tmp_path, fake_backend, monkeypatch, name="b")
    cmp = compare_configs(a.store, b.store, mode="greedy")
    assert cmp["refused"] is False
    for name in (
        "success_at_budget",
        "completion_rate",
        "censored_rate",
        "error_rate",
        "answer_observation_rate",
        "conditional_accuracy_given_completed",
    ):
        assert name in cmp["metrics"]
    assert "accuracy" not in cmp["metrics"]


def test_compare_refuses_prompt_policy_mismatch_by_default(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, name="a", prompt_policy="pp-v1")
    b = _run(tmp_path, fake_backend, monkeypatch, name="b", prompt_policy="pp-v2")
    cmp = compare_configs(a.store, b.store, mode="greedy")
    assert cmp["refused"] is True
    assert any("presentation" in r or "prompt policy" in r for r in cmp["validity"]["reasons"])


def test_compare_allows_prompt_policy_with_explicit_override(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, name="a", prompt_policy="pp-v1")
    b = _run(tmp_path, fake_backend, monkeypatch, name="b", prompt_policy="pp-v2")
    cmp = compare_configs(a.store, b.store, mode="greedy", prompt_policy_intentional=True)
    assert cmp["refused"] is False
    assert cmp["validity"]["presentation_mismatch"] is True


def test_compare_refuses_horizon_mismatch_unless_intentional(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, name="a")
    b = _run(tmp_path, fake_backend, monkeypatch, name="b")
    b.store.manifest["spec"] = {**b.store.manifest["spec"], "max_tokens": 32}
    refused = compare_configs(a.store, b.store, mode="greedy")
    assert refused["refused"] is True
    assert any("horizon" in r for r in refused["validity"]["reasons"])
    allowed = compare_configs(a.store, b.store, mode="greedy", horizon_intentional=True)
    assert allowed["refused"] is False


def test_conditional_accuracy_metric_is_none_when_censored():
    censored = {
        "correct": False,
        "answer_normalized": None,
        "termination": {"terminated_by_eos": False, "missing_answer": True},
    }
    assert _METRICS["conditional_accuracy_given_completed"](censored) is None
    completed = {
        "correct": True,
        "answer_normalized": "1",
        "termination": {"terminated_by_eos": True, "missing_answer": False},
    }
    assert _METRICS["conditional_accuracy_given_completed"](completed) == 1.0


# --------------------------------------------------------------------------- #
# Capture equivalence under penalty processors
# --------------------------------------------------------------------------- #
def test_capture_equivalence_under_penalty_processors(fake_backend):
    sampled = _sampled(
        max_tokens=16,
        seed=7,
        presence_penalty=1.5,
        presence_context_size=20,
        repetition_penalty=1.0,
    )
    out = verify_capture_equivalence(fake_backend, [10, 20, 30], sampled=sampled)
    assert out["equivalent"] is True
    assert out["sampled_same_seed"]["identical"] is True
