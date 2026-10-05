from __future__ import annotations

from pathlib import Path

from ponderscope.config.identity import (
    DecodingPolicy,
    Deployment,
    ModelIdentity,
    RuntimeIdentity,
    canonical_json,
    configuration_id,
    sha256_file,
)


def _deployment(**overrides) -> Deployment:
    model = ModelIdentity(
        repo_id="Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
        precision="bfloat16",
    )
    runtime = RuntimeIdentity(
        runtime="mlx-lm",
        runtime_version="0.32.0",
        backend="mlx-metal",
        hardware="Apple M3 Pro",
        os="macOS",
        python_version="3.12",
    )
    decoding = DecodingPolicy(mode="greedy", max_tokens=512)
    return Deployment(
        model=overrides.get("model", model),
        runtime=overrides.get("runtime", runtime),
        decoding=overrides.get("decoding", decoding),
    )


def test_configuration_id_stable_across_key_order():
    assert configuration_id({"b": 1, "a": 2}) == configuration_id({"a": 2, "b": 1})


def test_configuration_id_changes_with_revision():
    a = _deployment()
    b = _deployment(
        model=ModelIdentity(repo_id="Qwen/Qwen3.5-0.8B", revision="other", precision="bfloat16")
    )
    assert a.config_id != b.config_id


def test_configuration_id_changes_with_quantization():
    a = _deployment()
    b = _deployment(
        model=ModelIdentity(
            repo_id="Qwen/Qwen3.5-0.8B",
            revision="2fc06364715b967f1860aea9cf38778875588b17",
            precision="bfloat16",
            quantization="mlx-4bit",
            quantization_bits=4,
            quantization_group_size=64,
        )
    )
    assert a.config_id != b.config_id


def test_configuration_id_changes_with_decoding():
    a = _deployment()
    b = _deployment(
        decoding=DecodingPolicy(mode="sampled", max_tokens=512, seed=0, temperature=0.6)
    )
    assert a.config_id != b.config_id


def test_canonical_json_drops_empty_and_none():
    a = canonical_json({"x": None, "y": {}, "z": [], "keep": 1})
    b = canonical_json({"keep": 1})
    assert a == b


def test_sha256_file(tmp_path: Path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    assert sha256_file(str(p)) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
