"""Phase 1.4 comparison-contract tests: explicit contrast classification, RMST
reporting, and a negative matrix of confounded/incomplete comparisons.
"""

from __future__ import annotations

from pathlib import Path

from ponderscope.analysis import compare_configs
from ponderscope.analysis.compare import classify_contrast
from ponderscope.config.schema import ExperimentSpec
from ponderscope.experiment import run_experiment

CLEAN = {"version": "test", "git_sha": "0" * 40, "tracked_dirty": False}


def _run(tmp_path: Path, fake_backend, monkeypatch, name="t", seeds=(0, 1), task_seed=0):
    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda backend_name: fake_backend)
    spec = ExperimentSpec(
        name=name,
        task_pack="tasks-v1",
        families=["arith"],
        n_per_family=2,
        greedy_repeats=1,
        sampled_seeds=list(seeds),
        max_tokens=16,
        task_seed=task_seed,
    )
    return run_experiment(
        spec,
        model_repo="fake/model",
        model_revision="deadbeef",
        runs_dir=tmp_path,
        code_state=CLEAN,
    )


def _derived_variant_manifest(store) -> dict:
    wv = store.manifest["weight_variant"]
    return {
        **wv,
        "representation": "derived",
        "precision": "4bit",
        "quantization": "mlx-affine",
        "quantization_bits": 4,
        "quantization_group_size": 64,
        "variant_weight_files": {"model.safetensors": "d" * 64},
        "conversion": {"tool": "mlx-lm", "command": "convert -q --q-bits 4"},
    }


def test_classify_contrast_weight_representation():
    diff = {
        "changed": {
            "model.representation": {},
            "model.precision": {},
            "model.quantization": {},
            "model.quantization_bits": {},
            "model.quantization_group_size": {},
            "model.variant_weight_files": {},
            "model.conversion": {},
        }
    }
    label, fields = classify_contrast(
        diff,
        model_changed=sorted(diff["changed"]),
        runtime_changed=[],
        decoding_changed=[],
    )
    assert label == "weight_representation"
    assert "representation" in fields and "conversion" in fields


def test_classify_contrast_source_change_is_different_model():
    diff = {"changed": {"model.revision": {}, "model.weight_files": {}}}
    label, fields = classify_contrast(
        diff,
        model_changed=sorted(diff["changed"]),
        runtime_changed=[],
        decoding_changed=[],
    )
    assert label == "different_source_model"
    assert "revision" in fields


def test_bf16_to_q4_same_source_is_clean_weight_representation(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, name="bf16")
    b = _run(tmp_path, fake_backend, monkeypatch, name="q4")
    b.store.manifest["weight_variant"] = _derived_variant_manifest(b.store)
    b.store.manifest["artifact"] = b.store.manifest["weight_variant"]
    cmp = compare_configs(a.store, b.store, mode="sampled")
    assert cmp["refused"] is False, cmp["validity"]["reasons"]
    assert cmp["contrast"] == "weight_representation"
    # Several metadata fields changed together and that is expected, not a confound.
    assert set(cmp["validity"]["contrast_changed_fields"]) >= {
        "representation",
        "precision",
        "quantization",
        "quantization_bits",
        "quantization_group_size",
        "variant_weight_files",
        "conversion",
    }


def test_source_change_at_same_revision_is_refused(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, name="src-a")
    b = _run(tmp_path, fake_backend, monkeypatch, name="src-b")
    # Same repo+revision but different source weight hashes: inconsistent provenance.
    b.store.manifest["source_artifact"] = {
        **b.store.manifest["source_artifact"],
        "weight_files": {"fake.safetensors": "f" * 64},
    }
    cmp = compare_configs(a.store, b.store, mode="sampled")
    assert cmp["refused"] is True
    assert any("inconsistent provenance" in r for r in cmp["validity"]["reasons"])


def test_different_source_model_is_labelled_not_confounded(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, name="m-a")
    b = _run(tmp_path, fake_backend, monkeypatch, name="m-b")
    b.store.manifest["source_artifact"] = {
        **b.store.manifest["source_artifact"],
        "repo_id": "fake/other-model",
        "revision": "cafebabe",
    }
    cmp = compare_configs(a.store, b.store, mode="sampled")
    assert cmp["refused"] is False
    assert cmp["contrast"] == "different_source_model"


def test_prompt_policy_change_requires_intent(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, name="pp-a")
    b = _run(tmp_path, fake_backend, monkeypatch, name="pp-b")
    b.store.manifest["spec"] = {**b.store.manifest["spec"], "prompt_policy": "pp-v2"}
    cmp = compare_configs(a.store, b.store, mode="sampled")
    assert cmp["refused"] is True
    assert any("prompt policy" in r for r in cmp["validity"]["reasons"])


def test_horizon_change_requires_intent(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, name="h-a")
    b = _run(tmp_path, fake_backend, monkeypatch, name="h-b")
    b.store.manifest["spec"] = {**b.store.manifest["spec"], "max_tokens": 32}
    refused = compare_configs(a.store, b.store, mode="sampled")
    assert refused["refused"] is True
    assert any("horizon" in r for r in refused["validity"]["reasons"])
    allowed = compare_configs(a.store, b.store, mode="sampled", horizon_intentional=True)
    assert allowed["refused"] is False


def test_capture_lane_change_requires_intent(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, name="c-a")
    b = _run(tmp_path, fake_backend, monkeypatch, name="c-b")
    b.store.manifest["spec"] = {**b.store.manifest["spec"], "capture_level": "minimal"}
    cmp = compare_configs(a.store, b.store, mode="sampled")
    assert cmp["refused"] is True
    assert any("capture" in r for r in cmp["validity"]["reasons"])


def test_task_population_mismatch_refused(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, name="p-a", task_seed=0)
    b = _run(tmp_path, fake_backend, monkeypatch, name="p-b", task_seed=99)
    cmp = compare_configs(a.store, b.store, mode="sampled")
    assert cmp["refused"] is True
    assert any("task populations differ" in r for r in cmp["validity"]["reasons"])


def test_compare_reports_clustered_rmst(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, name="r-a")
    b = _run(tmp_path, fake_backend, monkeypatch, name="r-b")
    cmp = compare_configs(a.store, b.store, mode="sampled", n_resamples=200)
    endpoints = cmp["rmst"]["endpoints"]
    assert "reasoning_closure" in endpoints
    assert endpoints["reasoning_closure"]["n_clusters"] >= 1
    # Identical runs -> zero RMST difference.
    assert endpoints["reasoning_closure"]["mean"] == 0.0
