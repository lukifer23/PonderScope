"""`ponderscope doctor`: environment, runtime, and model capability report."""

from __future__ import annotations

import json
from typing import Any

from .backends import get_backend
from .config.identity import DecodingPolicy, Deployment, ModelIdentity
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

    declared = ModelIdentity(repo_id=repo_id, revision=revision, precision="unknown")
    loaded = backend.load(declared)
    report["model"] = loaded.to_dict()

    runtime = backend.runtime_identity()
    decoding = DecodingPolicy(mode="greedy", max_tokens=64)
    deployment = Deployment(model=loaded, runtime=runtime, decoding=decoding, label="doctor")
    report["deployment"] = deployment.to_dict()
    report["config_id"] = deployment.config_id
    report["deployment_description"] = deployment.describe()

    if live:
        messages = [
            {
                "role": "user",
                "content": "State the capital of France in one word after the tag 'Answer:'.",
            }
        ]
        prompt_ids = backend.tokenize_prompt(messages, enable_thinking=True)
        trace = backend.generate(
            prompt_ids,
            DecodingPolicy(mode="greedy", max_tokens=64),
            capture=_default_capture(),
        )
        report["live"] = {
            "prompt_tokens": trace.prompt_tokens,
            "generated_tokens": trace.generated_tokens,
            "finish_reason": trace.finish_reason,
            "terminated_by_eos": trace.terminated_by_eos,
            "think_end_reached": trace.think_end_reached,
            "think_end_token_id": getattr(backend, "think_end_token_id", None),
            "ttft_ms": trace.ttft_ms,
            "tokens_per_sec": trace.tokens_per_sec,
            "text_head": trace.text[:400],
        }
    if overhead:
        from .experiment import measure_capture_overhead

        messages = [{"role": "user", "content": "Continue counting: 1, 2, 3, 4, 5,"}]
        prompt_ids = backend.tokenize_prompt(messages, enable_thinking=True)
        report["overhead"] = measure_capture_overhead(
            backend, prompt_ids, DecodingPolicy(mode="greedy", max_tokens=128)
        )
    return report


def _default_capture():
    from .backends import CaptureSpec

    return CaptureSpec(entropy=False, top_k=3, logprob_digest=False)


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
    lines.append("Model")
    lines.append(f"  repo          {model['repo_id']}")
    lines.append(f"  revision      {model['revision']}")
    lines.append(f"  local path    {model['local_path']}")
    lines.append(f"  precision     {model['precision']}")
    lines.append(
        f"  quantization  {model['quantization']} "
        f"bits={model['quantization_bits']} group={model['quantization_group_size']}"
    )
    wf = model["weight_files"]
    lines.append(f"  weight files  {len(wf)}")
    for name, digest in wf.items():
        lines.append(f"    {name}  sha256:{digest[:16]}…")
    lines.append("Deployment")
    lines.append(f"  config id     {report['config_id']}")
    lines.append(f"  description   {report['deployment_description']}")
    if "live" in report:
        live = report["live"]
        lines.append("Live smoke (greedy, 64 tokens)")
        lines.append(
            f"  finish        {live['finish_reason']} eos={live['terminated_by_eos']} "
            f"think_end={live['think_end_reached']}"
        )
        lines.append(f"  ttft_ms       {live['ttft_ms']}")
        lines.append(f"  tokens/sec    {live['tokens_per_sec']}")
        lines.append(f"  text          {json.dumps(live['text_head'])}")
    if "overhead" in report:
        o = report["overhead"]
        frac = o["overhead_fraction"]
        lines.append("Instrumentation overhead (capture on vs off)")
        lines.append(f"  capture on    {o['capture_on_tps']:.1f} tok/s")
        lines.append(f"  capture off   {o['capture_off_tps']:.1f} tok/s")
        lines.append(
            f"  overhead      {frac * 100:.1f}%" if frac is not None else "  overhead      n/a"
        )
    return "\n".join(lines)
