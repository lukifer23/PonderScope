from __future__ import annotations

from pathlib import Path

from ponderscope.config.identity import (
    DecodingPolicy,
    Deployment,
    RuntimeIdentity,
    SourceArtifactIdentity,
    TrialIdentity,
    WeightVariantIdentity,
    canonical_json,
    configuration_id,
    sha256_file,
)


def _source(**overrides) -> SourceArtifactIdentity:
    base = {
        "repo_id": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "weight_files": {"model.safetensors": "a" * 64},
        "tokenizer_files": {"tokenizer.json": "b" * 64},
        "chat_template_sha256": "c" * 64,
    }
    base.update(overrides)
    return SourceArtifactIdentity(**base)


def _variant(**overrides) -> WeightVariantIdentity:
    base = {
        "source": overrides.pop("source", _source()),
        "representation": "original",
        "variant_weight_files": {"model.safetensors": "a" * 64},
        "precision": "bfloat16",
        "quantization": None,
    }
    base.update(overrides)
    return WeightVariantIdentity(**base)


def _runtime(**overrides) -> RuntimeIdentity:
    base = {
        "runtime": "mlx-lm",
        "runtime_version": "0.32.0",
        "backend": "mlx-metal",
        "hardware": "Apple M3 Pro",
        "os": "macOS",
        "python_version": "3.12",
    }
    base.update(overrides)
    return RuntimeIdentity(**base)


def _deployment(**overrides) -> Deployment:
    return Deployment(
        model=overrides.get("model", _variant()),
        runtime=overrides.get("runtime", _runtime()),
        decoding=overrides.get("decoding", DecodingPolicy(mode="greedy", max_tokens=512)),
        label=overrides.get("label"),
    )


def test_configuration_id_stable_across_key_order():
    assert configuration_id({"b": 1, "a": 2}) == configuration_id({"a": 2, "b": 1})


def test_source_artifact_id_changes_with_revision_and_weights():
    assert _source().source_artifact_id != _source(revision="other").source_artifact_id
    assert (
        _source().source_artifact_id
        != _source(weight_files={"model.safetensors": "z" * 64}).source_artifact_id
    )


def test_load_audit_does_not_change_artifact_identity():
    a = _variant(load_audit={"ok": True, "n_loaded_params": 320})
    b = _variant(load_audit={"ok": True, "n_loaded_params": 999})
    assert a.weight_variant_id == b.weight_variant_id
    assert a.source.source_artifact_id == b.source.source_artifact_id


def test_weight_variant_id_changes_with_quantization_and_variant_weights():
    a = _variant()
    b = _variant(
        representation="derived",
        variant_weight_files={"model.safetensors": "q" * 64},
        precision="float16",
        quantization="mlx-4bit",
        quantization_bits=4,
        quantization_group_size=64,
    )
    assert a.weight_variant_id != b.weight_variant_id
    assert a.source.source_artifact_id == b.source.source_artifact_id


def test_derived_variant_lineage_points_at_source():
    a = _variant()
    b = _variant(representation="derived", variant_weight_files={"model.safetensors": "q" * 64})
    assert a.derived_from_source_artifact_id is None
    assert b.derived_from_source_artifact_id == b.source.source_artifact_id


def test_label_does_not_change_any_identity():
    a = _deployment(label="run-one")
    b = _deployment(label="totally-different-name")
    assert a.artifact_id == b.artifact_id
    assert a.deployment_id == b.deployment_id
    assert a.condition_id == b.condition_id


def test_local_path_does_not_change_artifact_identity():
    a = _deployment(model=_variant(local_path="/cache/a"))
    b = _deployment(model=_variant(local_path="/somewhere/else"))
    assert a.artifact_id == b.artifact_id
    assert a.deployment_id == b.deployment_id


def test_condition_identity_has_no_seed():
    d = DecodingPolicy(mode="sampled", max_tokens=256, temperature=0.6, seed=0)
    assert "seed" not in d.identity_dict()
    assert "seed" in d.to_dict()


def test_seed_does_not_change_deployment_or_condition_identity():
    a = _deployment(
        decoding=DecodingPolicy(mode="sampled", max_tokens=256, temperature=0.6, seed=0)
    )
    b = _deployment(
        decoding=DecodingPolicy(mode="sampled", max_tokens=256, temperature=0.6, seed=12345)
    )
    assert a.deployment_id == b.deployment_id
    assert a.condition_id == b.condition_id
    assert (
        TrialIdentity("t", a.decoding, 0, 0).trial_id
        != TrialIdentity("t", b.decoding, 12345, 0).trial_id
    )


def test_temperature_changes_condition_identity_only():
    a = _deployment(
        decoding=DecodingPolicy(mode="sampled", max_tokens=256, temperature=0.6, seed=0)
    )
    b = _deployment(
        decoding=DecodingPolicy(mode="sampled", max_tokens=256, temperature=1.1, seed=0)
    )
    assert a.condition_id != b.condition_id
    assert a.deployment_id == b.deployment_id
    assert a.artifact_id == b.artifact_id


def test_runtime_or_hardware_changes_deployment_identity():
    a = _deployment()
    b = _deployment(runtime=_runtime(hardware="Apple M4 Max"))
    c = _deployment(runtime=_runtime(runtime_version="0.33.0"))
    assert a.deployment_id != b.deployment_id
    assert a.deployment_id != c.deployment_id
    assert a.artifact_id == b.artifact_id


def test_trial_identity_distinguishes_seed_and_repeat():
    cond = DecodingPolicy(mode="sampled", max_tokens=256, temperature=0.6, seed=0)
    t1 = TrialIdentity("task", cond, 0, 0)
    t2 = TrialIdentity("task", cond, 0, 1)
    t3 = TrialIdentity("task", cond, 1, 0)
    assert len({t1.trial_id, t2.trial_id, t3.trial_id}) == 3


def test_canonical_json_drops_empty_and_none():
    a = canonical_json({"x": None, "y": {}, "z": [], "keep": 1})
    b = canonical_json({"keep": 1})
    assert a == b


def test_sha256_file(tmp_path: Path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    assert sha256_file(str(p)) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
