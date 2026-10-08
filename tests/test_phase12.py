"""Phase 1.2 regression tests: censoring semantics, policy, capture, packs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ponderscope.analysis.analyze import (
    _budget_metrics,
    across_seed_variation,
    cross_seed_token_variation,
    prefix_invariance,
)
from ponderscope.backends.audit import audit_model_load
from ponderscope.backends.base import CaptureSpec
from ponderscope.config.identity import (
    DecodingPolicy,
    Deployment,
    RuntimeIdentity,
    SourceArtifactIdentity,
    WeightVariantIdentity,
)
from ponderscope.config.schema import ExperimentSpec
from ponderscope.evidence.bundle import bundle_run
from ponderscope.evidence.run import RunStore
from ponderscope.experiment import run_experiment, verify_capture_equivalence
from ponderscope.tasks import (
    generate_pack,
    generate_pack_metadata,
    get_pack,
    make_task,
    normalize,
    score,
)

CLEAN = {"version": "test", "git_sha": "0" * 40, "tracked_dirty": False}


# --------------------------------------------------------------------------- #
# Budget metric semantics
# --------------------------------------------------------------------------- #
def _rec(
    *,
    task_id="t",
    family="arith",
    mode="greedy",
    seed=None,
    repeat=0,
    correct=False,
    answer=None,
    tokens=(1, 2, 3),
    reasoning_tokens=3,
    eos=True,
    capped=False,
    think_end=True,
):
    status = "correct" if correct else ("incorrect" if answer is not None else "censored")
    if answer is None and not capped:
        status = "unparseable"
    return {
        "task_id": task_id,
        "family": family,
        "condition": {"mode": mode, "seed": seed, "repeat": repeat},
        "condition_id": f"cond-{mode}",
        "correct": correct,
        "answer_normalized": answer,
        "natural_final_status": status,
        "reasoning_tokens": reasoning_tokens,
        "trace": {"token_ids": list(tokens), "wall_ms": 1.0, "steps": []},
        "termination": {
            "terminated_by_eos": eos,
            "capped": capped,
            "think_end_reached": think_end,
            "eos_observed": eos,
            "missing_answer": answer is None,
            "finish_reason": "stop" if eos else "length",
            "error_type": None,
        },
        "metrics": {},
    }


def test_budget_metrics_separate_success_completion_and_conditional():
    records = [
        _rec(correct=True, answer="1"),
        _rec(correct=True, answer="1"),
        _rec(correct=False, answer=None, capped=True, eos=False, think_end=False),
        _rec(correct=False, answer=None, capped=True, eos=False, think_end=False),
    ]
    o = _budget_metrics(records)
    assert o["success_at_budget"] == 0.5
    assert o["completion_rate"] == 0.5
    assert o["conditional_accuracy_given_completed"] == 1.0
    assert o["answer_observed_accuracy"] == 1.0
    assert o["censored_rate"] == 0.5
    assert o["answer_observation_rate"] == 0.5


def test_budget_metrics_all_censored_has_no_conditional_accuracy():
    records = [
        _rec(answer=None, capped=True, eos=False, think_end=False),
        _rec(answer=None, capped=True, eos=False, think_end=False),
    ]
    o = _budget_metrics(records)
    assert o["success_at_budget"] == 0.0
    assert o["completion_rate"] == 0.0
    assert o["conditional_accuracy_given_completed"] is None
    assert o["answer_observed_accuracy"] is None
    assert o["censored_rate"] == 1.0


def test_all_censored_sampled_variance_is_insufficient_not_zero():
    records = []
    for seed, seq in ((0, (1, 2, 3)), (1, (1, 2, 3, 4, 5))):
        for rep in range(2):
            records.append(
                _rec(
                    mode="sampled",
                    seed=seed,
                    repeat=rep,
                    answer=None,
                    capped=True,
                    eos=False,
                    think_end=False,
                    tokens=seq,
                    reasoning_tokens=len(seq),
                )
            )
    across = across_seed_variation(records)
    assert across["available"] is True
    assert across["mean_distinct_answers"] is None  # unobserved, not "1"
    assert across["mean_accuracy_std_across_seeds"] is None
    # token-level variation is still measurable while censored
    tokens = cross_seed_token_variation(records)
    assert tokens["available"] is True
    assert tokens["mean_reasoning_tokens_std_across_seeds"] > 0
    assert tokens["mean_first_token_divergence"] == 3


def test_same_seed_ambiguity_detected_and_not_silently_rs0():
    records = [
        _rec(mode="sampled", seed=0, repeat=0, answer="1", tokens=(1, 2)),
        _rec(mode="sampled", seed=0, repeat=1, answer="2", tokens=(9, 9)),
        _rec(mode="sampled", seed=1, repeat=0, answer="1", tokens=(1, 2)),
        _rec(mode="sampled", seed=1, repeat=1, answer="1", tokens=(1, 2)),
    ]
    across = across_seed_variation(records)
    per = across["per_task"][0]
    assert per["n_ambiguous_seeds"] == 1
    assert across["any_ambiguous_seeds"] is True
    # the ambiguous seed's answer must not be counted as the canonical "1"
    assert per["n_seeds_with_observed_answer"] == 1


# --------------------------------------------------------------------------- #
# Cap-prefix invariance
# --------------------------------------------------------------------------- #
def test_prefix_invariance_true_when_shorter_is_prefix():
    out = prefix_invariance({256: [1, 2, 3], 512: [1, 2, 3, 4], 1024: [1, 2, 3, 4, 5]})
    assert out["exact_prefix"] is True
    assert len(out["checks"]) == 2


def test_prefix_invariance_false_when_diverging():
    out = prefix_invariance({256: [1, 2, 9], 512: [1, 2, 3, 4]})
    assert out["exact_prefix"] is False


# --------------------------------------------------------------------------- #
# Worktree policy
# --------------------------------------------------------------------------- #
def test_official_run_refuses_dirty_tracked_worktree(tmp_path, fake_backend, monkeypatch):
    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda name: fake_backend)
    spec = ExperimentSpec(name="dirty", task_pack="tasks-v1", families=["arith"], n_per_family=1)
    with pytest.raises(RuntimeError, match="dirty"):
        run_experiment(
            spec,
            model_repo="fake/model",
            model_revision="deadbeef",
            runs_dir=tmp_path,
            code_state={"version": "x", "git_sha": "0" * 40, "tracked_dirty": True},
        )


def test_exploratory_dirty_override_records_metadata(tmp_path, fake_backend, monkeypatch):
    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda name: fake_backend)
    spec = ExperimentSpec(name="explore", task_pack="tasks-v1", families=["arith"], n_per_family=1)
    result = run_experiment(
        spec,
        model_repo="fake/model",
        model_revision="deadbeef",
        runs_dir=tmp_path,
        allow_dirty=True,
        code_state={
            "version": "x",
            "git_sha": "0" * 40,
            "tracked_dirty": True,
            "working_tree_diff_sha256": "d" * 64,
        },
    )
    code = result.store.manifest["code"]
    assert code["exploratory"] is True
    assert code["publication_grade"] is False
    assert code["working_tree_diff_sha256"] == "d" * 64


# --------------------------------------------------------------------------- #
# Load audit
# --------------------------------------------------------------------------- #
class _FakeModel:
    def __init__(self, params, drop):
        self._params = params
        self._drop = drop

    def sanitize(self, weights):
        return {k: v for k, v in weights.items() if k not in self._drop}

    def parameters(self):
        return self._params


def _snapshot(tmp_path: Path, keys):
    import mlx.core as mx

    snap = tmp_path / "snap"
    snap.mkdir()
    arrays = {k: mx.zeros((2, 2)) for k in keys}
    mx.save_safetensors(str(snap / "model.safetensors"), arrays)
    return snap


def test_load_audit_allows_non_text_drop(tmp_path):
    keys = ["model.embed_tokens.weight", "model.visual.patch.weight", "mtp.foo"]
    snap = _snapshot(tmp_path, keys)
    import mlx.core as mx

    params = {"model.embed_tokens.weight": mx.zeros((2, 2))}
    model = _FakeModel(params, {"model.visual.patch.weight", "mtp.foo"})
    audit = audit_model_load(model, snap)
    assert audit["ok"] is True
    assert audit["n_dropped_allowed_non_text"] == 2
    assert audit["n_dropped_unexpected"] == 0
    assert audit["n_text_source_lost"] == 0


def test_load_audit_accepts_renamed_text_keys(tmp_path):
    keys = ["model.language_model.embed_tokens.weight", "model.visual.patch.weight"]
    snap = _snapshot(tmp_path, keys)

    class _RenameModel(_FakeModel):
        def sanitize(self, weights):
            return {"embed_tokens.weight": weights["model.language_model.embed_tokens.weight"]}

        def parameters(self):
            import mlx.core as mx

            return {"embed_tokens.weight": mx.zeros((2, 2))}

    audit = audit_model_load(_RenameModel({}, set()), snap)
    assert audit["ok"] is True
    assert audit["n_renamed_or_dropped_text"] == 1
    assert audit["renames_or_drops_accounted_by_count"] is True


def test_load_audit_hard_fails_on_unexpected_text_loss(tmp_path):
    keys = ["model.embed_tokens.weight", "model.visual.patch.weight"]
    snap = _snapshot(tmp_path, keys)
    model = _FakeModel({}, {"model.embed_tokens.weight", "model.visual.patch.weight"})
    audit = audit_model_load(model, snap)
    assert audit["ok"] is False
    assert "model.embed_tokens.weight" in audit["dropped_unexpected_keys"]
    assert "model.embed_tokens.weight" in audit["text_source_keys_lost"]
    assert audit["dropped_key_categories"].get("model", 0) == 2


# --------------------------------------------------------------------------- #
# Capture equivalence
# --------------------------------------------------------------------------- #
def test_capture_equivalence_fake_backend(fake_backend):
    out = verify_capture_equivalence(fake_backend, [10, 20, 30])
    assert out["equivalent"] is True
    assert out["greedy"]["identical"] is True
    assert out["sampled_same_seed"]["identical"] is True


def test_capture_equivalence_detects_instrumentation_effect(fake_backend):
    original = fake_backend.generate

    def differing(prompt_ids, decoding, capture):
        trace = original(prompt_ids, decoding, capture)
        if capture.entropy:  # research lane mutates the sequence
            trace.token_ids = trace.token_ids[:-1]
        return trace

    fake_backend.generate = differing  # type: ignore[method-assign]
    out = verify_capture_equivalence(fake_backend, [10, 20, 30])
    assert out["equivalent"] is False
    assert out["greedy"]["identical"] is False


# --------------------------------------------------------------------------- #
# Pack registry + prompt policy
# --------------------------------------------------------------------------- #
def test_unknown_pack_fails_closed():
    with pytest.raises(ValueError, match="unknown task pack"):
        get_pack("tasks-v9")
    with pytest.raises(ValueError, match="unknown task pack"):
        generate_pack(n_per_family=1, pack="tasks-v9")


def test_tasks_v1_is_frozen():
    tasks = {t.family: t for t in generate_pack(n_per_family=1, seed=0, split="dev")}
    assert tasks["arith"].task_id == "def4c48a5ef9a105"
    assert hashlib.sha256(tasks["arith"].prompt.encode()).hexdigest().startswith("1d4e4a25")
    assert tasks["order"].task_id == "f9e021d67855a0af"
    assert tasks["logic"].task_id == "bef299025f0673a9"
    meta = generate_pack_metadata(
        list(tasks.values()),
        families=None,
        n_per_family=1,
        split="dev",
        seed=0,
        pack="tasks-v1",
    )
    assert meta["task_ids_sha256"] == (
        "cca34706048b62ac7a246468b0d3c36eaf87064211210e58a779616862f4e20a"
    )
    assert meta["generator_version"] == "generator-v1"
    assert meta["prompt_policy"] == "pp-v1"


def test_prompt_policy_changes_wording_not_structure_or_id():
    p0 = make_task("arith", 0, prompt_policy="pp-v1")
    p1 = make_task("arith", 0, prompt_policy="pp-v2")
    assert p0.task_id == p1.task_id
    assert p0.answer == p1.answer
    assert p0.structural == p1.structural
    assert p0.prompt != p1.prompt
    assert "Answer:" in p0.prompt
    assert "Answer:" not in p1.prompt
    assert normalize("arith", f"Answer: {p1.answer}") == p1.answer
    assert score("arith", p1.answer, p1.answer)


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #
def _deployment() -> Deployment:
    return Deployment(
        model=WeightVariantIdentity(
            source=SourceArtifactIdentity(repo_id="fake/model", revision="deadbeef"),
            representation="original",
            precision="float32",
        ),
        runtime=RuntimeIdentity(
            runtime="fake",
            runtime_version="0",
            backend="fake",
            hardware="h",
            os="o",
            python_version="3.12",
        ),
        decoding=DecodingPolicy(mode="greedy", max_tokens=8),
    )


def _sealed(tmp_path: Path, name="bundle"):
    spec = ExperimentSpec(name=name, task_pack="tasks-v1")
    store = RunStore.create(
        _deployment(),
        spec,
        runs_dir=tmp_path,
        created_utc="20260101T001000Z",
        code_state=CLEAN,
    )
    store.write_tasks([{"task_id": "x"}])
    with store.open_traces() as w:
        w.append(
            {
                "task_id": "x",
                "condition_id": store.condition_ids[0],
                "deployment_id": store.deployment_id,
                "source_artifact_id": store.source_artifact_id,
                "weight_variant_id": store.weight_variant_id,
                "trial_id": "trial-1",
                "condition": {"mode": "greedy", "seed": None, "repeat": 0},
                "termination": {"think_end_reached": True, "terminated_by_eos": True},
                "trace": {"token_ids": [1, 2, 3]},
            }
        )
    with store.open_probes() as w:
        w.append({"task_id": "x"})
    store.seal()
    return store


def test_bundle_is_deterministic_and_requires_valid_seal(tmp_path):
    store = _sealed(tmp_path)
    a = bundle_run(store.path, tmp_path / "a.tar.gz")
    b = bundle_run(store.path, tmp_path / "b.tar.gz")
    assert a["sha256"] == b["sha256"]
    assert "manifest.json" in a["members"] and "evidence.json" in a["members"]
    assert "traces.jsonl" in a["members"]

    # tamper -> refuse
    manifest = store.path / "manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(RuntimeError, match="seal does not verify"):
        bundle_run(store.path, tmp_path / "c.tar.gz")


def test_cap_prefix_invariance_helper_with_backend(fake_backend):
    from ponderscope.calibration import run_cap_prefix_invariance

    prompts = [{"task_id": "t1", "family": "arith", "token_ids": [10, 20, 30]}]
    out = run_cap_prefix_invariance(fake_backend, prompts, [128, 256, 512])
    assert out["all_exact_prefix"] is True
    assert out["per_task"][0]["lengths"] == {"128": 23, "256": 23, "512": 23}


def test_capture_level_minimal_via_spec(fake_backend, tmp_path, monkeypatch):
    from ponderscope.experiment import run_experiment

    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda name: fake_backend)
    spec = ExperimentSpec(
        name="minimal",
        task_pack="tasks-v1",
        families=["arith"],
        n_per_family=1,
        max_tokens=8,
        capture_level="minimal",
    )
    result = run_experiment(
        spec,
        model_repo="fake/model",
        model_revision="deadbeef",
        runs_dir=tmp_path,
        code_state=CLEAN,
    )
    rec = result.store.read_traces()[0]
    capture = rec["trace"]["extra"]["capture"]
    assert capture["entropy"] is False
    assert capture["per_token_timing"] is False
    assert capture["chosen_logprob"] is False


def test_capture_spec_minimal_vs_research_fields():
    assert CaptureSpec.minimal().entropy is False
    assert CaptureSpec.research(top_k=5).top_k == 5
