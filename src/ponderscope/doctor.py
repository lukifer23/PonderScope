"""`ponderscope doctor`: environment, runtime, and model capability report."""

from __future__ import annotations

import json
from typing import Any

from .backends import get_backend
from .config.identity import DecodingPolicy, Deployment, SourceArtifactIdentity
from .evidence.environment import capture_environment

DEFAULT_REPO = "Qwen/Qwen3.5-0.8B"
DEFAULT_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def run_doctor(
    repo_id: str = DEFAULT_REPO,
    revision: str = DEFAULT_REVISION,
    backend_name: str = "mlx",
    live: bool = False,
    overhead: bool = False,
) -> dict[str, Any]:
    backend = get_backend(backend_name)
    caps = backend.capabilities()
    report: dict[str, Any] = {
        "environment": capture_environment(),
        "capabilities": caps.to_dict(),
    }

    request = SourceArtifactIdentity(repo_id=repo_id, revision=revision)
    loaded = backend.load(request)
    report["model"] = loaded.to_dict()
    report["source_artifact_id"] = loaded.source.source_artifact_id
    report["weight_variant_id"] = loaded.weight_variant_id

    runtime = backend.runtime_identity()
    decoding = DecodingPolicy(mode="greedy", max_tokens=64)
    deployment = Deployment(model=loaded, runtime=runtime, decoding=decoding, label="doctor")
    report["deployment"] = deployment.to_dict()
    report["artifact_id"] = deployment.artifact_id
    report["deployment_id"] = deployment.deployment_id
    report["condition_id"] = deployment.condition_id
    report["deployment_description"] = deployment.describe()

    if live:
        from .capability import validate_live

        report["capability"] = validate_live(backend, overhead=overhead)
    return report


def format_doctor(report: dict[str, Any]) -> str:
    lines: list[str] = []
    env = report["environment"]
    caps = report["capabilities"]
    model = report["model"]
    lines.append("PonderScope doctor")
    lines.append("=" * 60)
    lines.append("Environment")
    lines.append(f"  python        {env['python_version']}")
    lines.append(f"  platform      {env['platform']}")
    lines.append(f"  mlx device    {env.get('mlx_default_device')}")
    lines.append(f"  memory        {env.get('physical_memory_bytes')}")
    lines.append("Runtime")
    lines.append(f"  runtime       {caps['runtime']} {caps['runtime_version']}")
    lines.append(f"  backend       {caps['name']}")
    lines.append("Capabilities")
    for key in (
        "greedy",
        "seeded_sampling",
        "logprobs",
        "entropy",
        "full_logprob_digest",
        "per_token_timing",
        "ttft",
        "prefix_probe",
    ):
        lines.append(f"  {key:<22} {caps[key]}")
    source = model.get("source", model)
    lines.append("Model")
    lines.append(f"  repo          {source.get('repo_id')}")
    lines.append(f"  revision      {source.get('revision')}")
    lines.append(f"  local path    {model.get('local_path')}")
    lines.append(f"  precision     {model.get('precision')}")
    lines.append(
        f"  quantization  {model['quantization']} "
        f"bits={model['quantization_bits']} group={model['quantization_group_size']}"
    )
    wf = model.get("variant_weight_files") or model.get("source", {}).get("weight_files", {})
    lines.append(f"  representation {model.get('representation')}")
    lines.append(f"  weight files  {len(wf)}")
    for name, digest in wf.items():
        lines.append(f"    {name}  sha256:{digest[:16]}…")
    lines.append("Deployment")
    lines.append(f"  source art id {report.get('source_artifact_id', report['artifact_id'])}")
    lines.append(f"  weight var id {report.get('weight_variant_id', report['artifact_id'])}")
    lines.append(f"  deployment id {report['deployment_id']}")
    lines.append(f"  condition id  {report['condition_id']}")
    lines.append(f"  description   {report['deployment_description']}")
    if "capability" in report:
        cap = report["capability"]
        c = cap["capabilities"]
        lines.append("Live capability proof")
        lines.append(f"  prompt tail    {json.dumps(c['thinking_prompt_tail'])}")
        lines.append(f"  think start    {c['think_start']['id']} {c['think_start']['decoded']!r}")
        lines.append(f"  think end      {c['think_end']['id']} {c['think_end']['decoded']!r}")
        lines.append(f"  eos            {c['eos']['ids']} {c['eos']['decoded']}")
        cl = c["native_closure"]
        lines.append(
            f"  closure        supported={cl.get('supported')} ids={cl.get('ids')} "
            f"decoded={cl.get('decoded')!r}"
        )
        nc = c["natural_closure"]
        lines.append(
            f"  natural close  supported={nc['supported']} tokens={nc.get('reasoning_tokens')} "
            f"answer={nc.get('answer')!r} think_end={nc.get('think_end_observed')} "
            f"final_channel={nc.get('final_answer_channel_observed')} "
            f"eos={nc.get('eos_observed')} parseable={nc.get('parseable_answer_observed')}"
        )
        ff = c["forced_finalization"]
        lines.append(
            f"  forced probe   supported={ff.get('supported')} "
            f"required_cuts={ff.get('n_required_cuts')} honored={ff.get('n_honored')} "
            f"all_close={ff.get('all_forced_close')} all_final_mode="
            f"{ff.get('all_entered_final_answer_mode')}"
        )
        lines.append(
            f"  greedy replay  identical={c['greedy']['replay_identical']} "
            f"finish={c['greedy']['finish_reason']}"
        )
        td = c["token_distribution"]
        lines.append(
            f"  distribution   entropy={td.get('entropy_nats')} top_sorted={td.get('top_sorted_descending')} "
            f"chosen_in_top={td.get('chosen_in_top')}"
        )
        lines.append(f"  same-seed      identical={c['same_seed_replay']['identical_token_ids']}")
        if "capture_equivalence" in c:
            ce = c["capture_equivalence"]
            lines.append(
                f"  capture-equiv  equivalent={ce['equivalent']} "
                f"greedy={ce['greedy']['identical']} sampled={ce['sampled_same_seed']['identical']}"
            )
        if "instrumentation_overhead" in c:
            o = c["instrumentation_overhead"]["modes"]
            for name in ("minimal", "research", "digest"):
                m = o[name]
                frac = m.get("decode_overhead_fraction")
                frac_txt = f"{frac * 100:+.1f}%" if frac is not None else "n/a"
                lines.append(
                    f"  overhead {name:<8} end2end={m['output_tokens_per_sec_median']:.1f}tok/s "
                    f"decode={m['decode_tokens_per_sec_median']:.1f}tok/s overhead={frac_txt}"
                )
        lines.append(f"  UNSUPPORTED    {cap['unsupported'] or 'none'}")
    return "\n".join(lines)
