"""Phase 1.4 conversion/provenance tests.

These exercise the first-party conversion *bookkeeping* (source identity,
storage guard, provenance validation, lineage, path-independence) without
running a multi-gigabyte conversion. The real conversion is validated separately
by a live load audit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ponderscope.backends.mlx_backend import hash_snapshot_files
from ponderscope.cli import build_parser, main
from ponderscope.config.identity import SourceArtifactIdentity
from ponderscope.conversion import (
    VARIANT_SCHEMA,
    ConversionError,
    check_storage,
    convert_q4_variant,
    load_variant_provenance,
    source_identity_from_snapshot,
    variant_identity,
    verify_variant_artifact,
)


def _fake_snapshot(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"source-weights")
    (root / "tokenizer.json").write_text('{"v": 1}')
    (root / "chat_template.jinja").write_text("{{ messages }}")
    return root


def _provenance(snapshot: Path, artifact: Path) -> dict:
    source = source_identity_from_snapshot("fake/model", "deadbeef", snapshot)
    weights, tokenizers = hash_snapshot_files(artifact)
    return {
        "schema": VARIANT_SCHEMA,
        "created_utc": "20260101T000000Z",
        "source_artifact": source.to_dict(),
        "representation": "derived",
        "precision": "4bit",
        "quantization": "mlx-affine",
        "quantization_bits": 4,
        "quantization_group_size": 64,
        "quantization_params": {
            "quantized": True,
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "layer_schemes": [{"bits": 4, "group_size": 64, "mode": "affine", "layers": 3}],
        },
        "conversion": {
            "tool": "mlx-lm",
            "tool_version": "0.32.0",
            "function": "mlx_lm.convert.convert",
            "params": {"q_bits": 4, "q_group_size": 64, "q_mode": "affine", "dtype": "bfloat16"},
        },
        "variant_weight_files": weights,
        "variant_tokenizer_files": tokenizers,
        "chat_template_sha256": source.chat_template_sha256,
        "think_end_token_id": 248069,
        "eos_token_ids": [248046],
        "local": {"source_path": str(snapshot), "output_path": str(artifact)},
    }


def test_source_identity_from_snapshot(tmp_path: Path):
    snap = _fake_snapshot(tmp_path / "snap")
    src = source_identity_from_snapshot("fake/model", "deadbeef", snap)
    assert src.repo_id == "fake/model"
    assert "model.safetensors" in src.weight_files
    assert "tokenizer.json" in src.tokenizer_files
    assert src.chat_template_sha256 is not None
    assert src.source_artifact_id.startswith("src-")


def test_check_storage_insufficient_raises(tmp_path: Path, monkeypatch):
    snap = _fake_snapshot(tmp_path / "snap")
    out = tmp_path / "out"

    class Usage:
        free = 1

    monkeypatch.setattr("ponderscope.conversion.shutil.disk_usage", lambda _p: Usage())
    with pytest.raises(ConversionError, match="insufficient free space"):
        check_storage(snap, out)


def test_check_storage_ok(tmp_path: Path):
    snap = _fake_snapshot(tmp_path / "snap")
    report = check_storage(snap, tmp_path / "out")
    assert report["source_bytes"] > 0
    assert report["free_bytes"] >= report["required_bytes"]


def test_convert_refuses_existing_output(tmp_path: Path):
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(ConversionError, match="refusing to overwrite"):
        convert_q4_variant("Qwen/Qwen3.5-4B", "deadbeef", out)


def test_provenance_fail_closed_missing_and_bad_schema(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    with pytest.raises(ConversionError, match="no derived-variant provenance"):
        load_variant_provenance(artifact)
    (artifact / "ponderscope_variant.json").write_text("{bad")
    with pytest.raises(ConversionError, match="invalid provenance JSON"):
        load_variant_provenance(artifact)
    (artifact / "ponderscope_variant.json").write_text(json.dumps({"schema": "other"}))
    with pytest.raises(ConversionError, match="unsupported variant provenance schema"):
        load_variant_provenance(artifact)


def test_verify_variant_artifact_detects_mismatch(tmp_path: Path):
    snap = _fake_snapshot(tmp_path / "snap")
    artifact = _fake_snapshot(tmp_path / "artifact")
    prov = _provenance(snap, artifact)
    assert verify_variant_artifact(artifact, prov)["ok"] is True
    (artifact / "model.safetensors").write_bytes(b"tampered")
    report = verify_variant_artifact(artifact, prov)
    assert report["ok"] is False
    assert any("weight-file hashes" in e for e in report["errors"])


def test_variant_identity_lineage_and_path_independence(tmp_path: Path):
    snap = _fake_snapshot(tmp_path / "snap")
    artifact_a = _fake_snapshot(tmp_path / "a")
    artifact_b = _fake_snapshot(tmp_path / "b")
    prov = _provenance(snap, artifact_a)
    va = variant_identity(prov, local_path=str(artifact_a))
    vb = variant_identity(prov, local_path=str(artifact_b))
    assert va.representation == "derived"
    assert va.derived_from_source_artifact_id == va.source.source_artifact_id
    # Local filesystem paths never change scientific identity.
    assert va.weight_variant_id == vb.weight_variant_id


def test_bf16_and_q4_share_source_but_differ_as_variants(tmp_path: Path):
    snap = _fake_snapshot(tmp_path / "snap")
    artifact = _fake_snapshot(tmp_path / "q4")
    prov = _provenance(snap, artifact)
    q4 = variant_identity(prov, local_path=str(artifact))
    bf16 = type(q4)(
        source=SourceArtifactIdentity(**prov["source_artifact"]),
        representation="original",
        variant_weight_files=prov["source_artifact"]["weight_files"],
        precision="bfloat16",
    )
    assert q4.source.source_artifact_id == bf16.source.source_artifact_id
    assert q4.weight_variant_id != bf16.weight_variant_id
    assert q4.derived_from_source_artifact_id == bf16.source.source_artifact_id
    assert bf16.derived_from_source_artifact_id is None


def test_conversion_field_excludes_local_paths(tmp_path: Path):
    snap = _fake_snapshot(tmp_path / "snap")
    artifact = _fake_snapshot(tmp_path / "artifact")
    prov = _provenance(snap, artifact)
    v = variant_identity(prov, local_path=str(artifact))
    assert "source_path" not in v.conversion
    assert "output_path" not in v.conversion
    assert v.conversion["tool"] == "mlx-lm"


def test_convert_cli_help_path(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["convert", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "ponderscope convert" in out
    assert "--dry-run" in out


def test_run_cli_accepts_artifact_path():
    parser = build_parser()
    args = parser.parse_args(["run", "--name", "x", "--artifact-path", "/tmp/q4"])
    assert args.artifact_path == "/tmp/q4"


def test_convert_cli_dry_run_reports_without_converting(tmp_path, monkeypatch, capsys):
    snap = _fake_snapshot(tmp_path / "snap")
    monkeypatch.setattr(
        "ponderscope.backends.mlx_backend.resolve_local_snapshot", lambda repo, rev: snap
    )
    rc = main(
        [
            "convert",
            "--source-repo",
            "fake/model",
            "--revision",
            "deadbeef",
            "--out",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "convert dry-run" in out
    assert not (tmp_path / "out").exists()


def test_run_with_artifact_path_records_derived_variant(tmp_path, fake_backend, monkeypatch):
    from ponderscope.config.schema import ExperimentSpec
    from ponderscope.experiment import run_experiment

    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda name: fake_backend)
    spec = ExperimentSpec(
        name="q4",
        task_pack="tasks-v1",
        families=["arith"],
        n_per_family=1,
        sampled_seeds=[0],
        max_tokens=16,
    )
    result = run_experiment(
        spec,
        model_repo="fake/model",
        model_revision="deadbeef",
        artifact_path="/tmp/q4",
        runs_dir=tmp_path,
        code_state={"version": "test", "git_sha": "0" * 40, "tracked_dirty": False},
    )
    manifest = result.store.manifest
    assert manifest["weight_variant"]["representation"] == "derived"
    assert manifest["weight_variant"]["quantization_bits"] == 4
    assert manifest["source_artifact_id"].startswith("src-")
    assert manifest["weight_variant_id"].startswith("wvar-")
