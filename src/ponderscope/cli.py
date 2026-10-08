"""PonderScope command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config.schema import ExperimentSpec

MODEL_REPO_DEFAULT = "Qwen/Qwen3.5-0.8B"
REVISION_DEFAULT = "2fc06364715b967f1860aea9cf38778875588b17"


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import format_doctor, run_doctor

    report = run_doctor(
        repo_id=args.model_repo,
        revision=args.revision,
        backend_name=args.backend,
        live=args.live,
        overhead=args.overhead,
    )
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_doctor(report))
    if args.save:
        from .evidence.store import atomic_write_json

        atomic_write_json(Path(args.save), report)
        print(f"saved capability/provenance report to {args.save}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from .evidence.store import write_jsonl
    from .tasks import generate_pack

    tasks = generate_pack(
        families=args.families or None,
        n_per_family=args.n_per_family,
        pack=args.pack,
        split=args.split,
        seed=args.seed,
        variant=args.variant,
        prompt_policy=args.prompt_policy,
    )
    out = Path(args.out)
    write_jsonl(out, [t.to_dict() for t in tasks])
    counts: dict[str, int] = {}
    for t in tasks:
        counts[t.family] = counts.get(t.family, 0) + 1
    print(f"wrote {len(tasks)} validated tasks to {out}")
    print(f"families: {counts}")
    print(f"pack={args.pack} split={args.split} seed={args.seed}")
    return 0


def _load_spec(args: argparse.Namespace) -> ExperimentSpec:
    from .config.schema import ExperimentSpec

    if args.spec:
        data = json.loads(Path(args.spec).read_text())
        return ExperimentSpec.from_dict(data)
    if not args.name:
        raise SystemExit("either --spec or --name is required")
    return ExperimentSpec(
        name=args.name,
        task_pack=args.task_pack,
        prompt_policy=args.prompt_policy,
        model_policy=args.model_policy,
        families=args.families or [],
        n_per_family=args.n_per_family,
        task_seed=args.task_seed,
        split=args.split,
        greedy_repeats=args.greedy_repeats,
        sampled_seeds=args.sampled_seeds or [],
        sampled_repeats_per_seed=args.sampled_repeats,
        max_tokens=args.max_tokens,
        probe=args.probe,
        n_probes=args.n_probes,
        capture_logprob_digest=args.logprob_digest,
        capture_level=args.capture_level,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    from .analysis import analyze_run, generate_report
    from .config.schema import ExperimentSpec  # noqa: F401
    from .experiment import run_experiment

    spec = _load_spec(args)
    result = run_experiment(
        spec,
        model_repo=args.model_repo,
        model_revision=args.revision,
        backend_name=args.backend,
        artifact_path=args.artifact_path,
        runs_dir=args.runs_dir,
        run_suffix=args.suffix,
        allow_dirty=args.allow_dirty,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(f"run {result.store.run_id}")
    print(f"  path: {result.store.path}")
    print(f"  generations: {result.n_generations}  probes: {result.n_probes}")
    if args.analyze:
        analysis = analyze_run(result.store)
        md, _ = generate_report(result.store, analysis)
        print(f"  analysis: {result.store.path / 'analysis.json'}")
        print(f"  report:   {result.store.path / 'summary.md'}")
        head = md.splitlines()[:1]
        if head:
            print(f"  {head[0]}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .analysis import analyze_run, generate_report
    from .evidence.run import RunStore

    store = RunStore.latest(args.runs_dir) if args.latest else RunStore.load(args.run)
    analysis = analyze_run(store, allow_unverified=args.allow_unverified)
    print(
        f"analyzed {store.run_id}: {analysis['n_generations']} generations, {analysis['n_probes']} probes"
    )
    print(
        f"  evidence_verified={analysis['evidence_verified']} "
        f"publication_grade={analysis['publication_grade']}"
    )
    if args.report:
        generate_report(store, analysis)
        print(f"report: {store.path / 'summary.md'}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    import datetime as _dt

    from .analysis import compare_configs
    from .evidence.run import RunStore
    from .evidence.store import atomic_write_json

    a = RunStore.load(args.a)
    b = RunStore.load(args.b)
    comparison = compare_configs(
        a,
        b,
        mode=args.mode,
        allow_confounded=args.allow_confounded,
        decoding_intentional=args.decoding_intentional,
        prompt_policy_intentional=args.prompt_policy_intentional,
        horizon_intentional=args.horizon_intentional,
        capture_intentional=args.capture_intentional,
        require_complete=args.require_complete,
        allow_unverified=args.allow_unverified,
    )
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        a.path / "comparisons" / f"{ts}-{a.deployment_id}-vs-{b.deployment_id}-{args.mode}.json"
    )
    atomic_write_json(out_path, comparison)
    print(f"compared {a.deployment_id} vs {b.deployment_id} [{args.mode}]")
    print(f"  written: {out_path}")
    validity = comparison["validity"]
    print(
        f"  contrast: {comparison.get('contrast')} "
        f"(changed: {validity.get('contrast_changed_fields')})"
    )
    pop = comparison.get("trial_population", {})
    print(
        f"  trial population: matched={pop.get('matched_pairs')} "
        f"expected A={pop.get('expected_trials_a')} B={pop.get('expected_trials_b')} "
        f"observed A={pop.get('observed_executions_a')} B={pop.get('observed_executions_b')} "
        f"missing A={pop.get('missing_trials_a')} B={pop.get('missing_trials_b')} "
        f"unexpected A={pop.get('unexpected_trials_a')} B={pop.get('unexpected_trials_b')} "
        f"duplicates A={pop.get('duplicate_keys_a')} B={pop.get('duplicate_keys_b')} "
        f"ambiguous A={pop.get('ambiguous_draws_a')} B={pop.get('ambiguous_draws_b')} "
        f"complete={pop.get('complete')}"
    )
    ver = comparison.get("verification", {})
    if ver.get("checked"):
        print(
            f"  evidence verified: A={bool(ver.get('a', {}).get('pass'))} "
            f"B={bool(ver.get('b', {}).get('pass'))}"
        )
    if comparison.get("refused"):
        print("  REFUSED: contrast is confounded, under-specified, unverified, or incomplete.")
        for reason in comparison.get("refusal_reasons", validity["reasons"]):
            print(f"    - {reason}")
        print("  expected: controlled dimensions identical; only the manipulated dimension differs")
        print(
            "  correct by aligning the listed fields, or pass the matching "
            "*_intentional flag, or use --allow-confounded / --allow-unverified for a "
            "labelled exploratory run"
        )
        return 1
    if comparison.get("exploratory"):
        print("  EXPLORATORY / CONFOUNDED (override applied)")
        for reason in comparison.get("refusal_reasons", validity["reasons"]):
            print(f"    - {reason}")
    print(
        f"  matched trials={comparison['n_matched_trials']} tasks={comparison['n_matched_tasks']}"
    )
    for metric, m in comparison["metrics"].items():
        d = m["delta"]
        print(
            f"  {metric:<26} delta={d.get('mean')} ci=[{d.get('lo')}, {d.get('hi')}] "
            f"noise={m['noise_scale']} -> {m['classification']}"
        )
    rmst = comparison.get("rmst", {}).get("endpoints", {})
    for endpoint, m in rmst.items():
        if m.get("mean") is None:
            continue
        print(
            f"  rmst[{endpoint}] delta={m.get('mean')} ci=[{m.get('lo')}, {m.get('hi')}] "
            f"clusters={m.get('n_clusters')}"
        )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from .evidence.run import RunStore, verify_run

    store = RunStore.latest(args.runs_dir) if args.latest else RunStore.load(args.run)
    report = verify_run(store.path)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "PASS" if report["pass"] else "FAIL"
        print(f"verify {report['run_id']}: {status}")
        for check in report["checks"]:
            mark = "ok" if check["ok"] else "MISMATCH"
            print(f"  {check['file']:<22} {mark}")
        for err in report["errors"]:
            print(f"  error: {err}")
    return 0 if report["pass"] else 1


def _cmd_calibrate_prefix(args: argparse.Namespace) -> int:
    from .backends import get_backend
    from .calibration import run_cap_prefix_invariance
    from .config.identity import SourceArtifactIdentity
    from .evidence.store import atomic_write_json
    from .tasks import generate_pack

    backend = get_backend(args.backend)
    source = backend.load(SourceArtifactIdentity(repo_id=args.model_repo, revision=args.revision))
    tasks = generate_pack(
        families=args.families or None,
        n_per_family=args.n_per_family,
        pack="tasks-v1",
        split="calibration",
        seed=args.task_seed,
        prompt_policy=args.prompt_policy,
    )
    prompts = [
        {
            "task_id": t.task_id,
            "family": t.family,
            "token_ids": backend.tokenize_prompt(
                [{"role": "user", "content": t.prompt}], enable_thinking=True
            ),
        }
        for t in tasks
    ]
    result = run_cap_prefix_invariance(backend, prompts, args.caps)
    result["model"] = {"repo_id": source.repo_id, "revision": source.revision}
    result["source_artifact_id"] = source.source.source_artifact_id
    result["prompt_policy"] = args.prompt_policy
    atomic_write_json(Path(args.out), result)
    print(
        f"cap-prefix invariance: status={result['status']} "
        f"all_exact_prefix={result['all_exact_prefix']} "
        f"comparisons={result['n_comparisons']}"
    )
    print(f"  caps={result['caps']} prompts={result['n_prompts']}")
    for t in result["per_task"]:
        print(
            f"  {t['family']}/{t['task_id'][:8]} lengths={t['lengths']} ok={t['invariance']['exact_prefix']}"
        )
    print(f"  saved {args.out}")
    return 0


def _cmd_calibrate_replay(args: argparse.Namespace) -> int:
    from .backends import get_backend
    from .calibration import run_same_seed_replay_subset
    from .config.conditions import condition_identities
    from .config.identity import SourceArtifactIdentity
    from .config.schema import ExperimentSpec
    from .evidence.store import atomic_write_json
    from .tasks import generate_pack

    data = json.loads(Path(args.spec).read_text())
    spec = ExperimentSpec.from_dict(data)
    backend = get_backend(args.backend)
    source = backend.load(SourceArtifactIdentity(repo_id=args.model_repo, revision=args.revision))
    conds = condition_identities(spec)
    if len(conds) != 1:
        raise SystemExit(f"replay needs exactly one condition in the spec, found {len(conds)}")
    decoding = conds[0]
    if decoding.mode != "sampled" or decoding.seed is None:
        raise SystemExit("replay requires a sampled condition with an explicit seed")
    tasks = generate_pack(
        families=spec.families or None,
        n_per_family=spec.n_per_family,
        pack=spec.task_pack,
        split=spec.split,
        seed=spec.task_seed,
        prompt_policy=spec.prompt_policy,
    )
    wanted = list(dict.fromkeys(args.families))
    selected = []
    seen: set[str] = set()
    for task in tasks:
        if task.family in wanted and task.family not in seen:
            selected.append(task)
            seen.add(task.family)
    prompts = [
        {
            "task_id": task.task_id,
            "family": task.family,
            "token_ids": backend.tokenize_prompt(
                [{"role": "user", "content": task.prompt}], enable_thinking=True
            ),
        }
        for task in selected
    ]
    result = run_same_seed_replay_subset(backend, prompts, decoding, repeats=args.repeats)
    result["model"] = {"repo_id": source.repo_id, "revision": source.revision}
    result["source_artifact_id"] = source.source.source_artifact_id
    result["prompt_policy"] = spec.prompt_policy
    result["spec_hash"] = spec.canonical_hash()
    atomic_write_json(Path(args.out), result)
    print(
        f"same-seed replay: scope={result['presence_scope']} seed={result['seed']} "
        f"all_token_identical={result['all_token_identical']} tasks={result['n_tasks']}"
    )
    for p in result["per_task"]:
        print(f"  {p['family']}/{p['task_id'][:8]} identical={p['token_identical']}")
    print(f"  saved {args.out}")
    return 0


def _cmd_bundle(args: argparse.Namespace) -> int:
    from .evidence.bundle import bundle_run
    from .evidence.run import RunStore

    store = RunStore.latest(args.runs_dir) if args.latest else RunStore.load(args.run)
    result = bundle_run(store.path, args.output)
    print(f"bundled {result['run_id']} -> {result['archive']}")
    print(f"  sha256: {result['sha256']}")
    print(f"  bytes:  {result['bytes']}  members: {len(result['members'])}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .analysis import generate_report
    from .evidence.run import RunStore

    store = RunStore.latest(args.runs_dir) if args.latest else RunStore.load(args.run)
    if not store.has_analysis():
        raise SystemExit(f"{store.run_id} has no analysis.json; run `ponderscope analyze` first")
    md, html_text = generate_report(store)
    print(
        f"wrote {store.path / 'summary.md'} and {store.path / 'summary.html'} ({len(md)} md chars, {len(html_text)} html chars)"
    )
    return 0


def _cmd_model_policy(args: argparse.Namespace) -> int:
    from .config.policies import load_model_policy

    profile = load_model_policy(args.id)
    if args.json:
        print(json.dumps(profile.to_dict(), indent=2))
    else:
        print(
            f"{profile.profile_id}: {profile.repo_id}@{profile.revision} "
            f"({profile.generation_mode})"
        )
        print(f"  equivalence label: {profile.condition_label}")
        print(f"  recommended: {profile.sampling_kwargs()}")
        print(f"  provenance: {profile.provenance}")
    return 0


def _cmd_survival(args: argparse.Namespace) -> int:
    from .analysis import analyze_run
    from .evidence.run import RunStore

    store = RunStore.latest(args.runs_dir) if args.latest else RunStore.load(args.run)
    analysis = analyze_run(store, allow_unverified=args.allow_unverified)
    print(json.dumps(analysis["survival"], indent=2, default=str))
    return 0


def _cmd_loop_diagnostics(args: argparse.Namespace) -> int:
    from .analysis import analyze_run
    from .evidence.run import RunStore

    store = RunStore.latest(args.runs_dir) if args.latest else RunStore.load(args.run)
    analysis = analyze_run(store, allow_unverified=args.allow_unverified)
    print(json.dumps(analysis["loop"], indent=2, default=str))
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    from .backends.mlx_backend import resolve_local_snapshot
    from .conversion import (
        check_storage,
        convert_q4_variant,
        source_identity_from_snapshot,
    )

    if args.dry_run:
        snapshot = resolve_local_snapshot(args.source_repo, args.revision)
        source = source_identity_from_snapshot(args.source_repo, args.revision, snapshot)
        storage = check_storage(snapshot, args.out)
        report = {
            "dry_run": True,
            "source_repo": args.source_repo,
            "revision": args.revision,
            "snapshot": str(snapshot),
            "source_artifact_id": source.source_artifact_id,
            "source_weight_files": source.weight_files,
            "chat_template_sha256": source.chat_template_sha256,
            "storage": storage,
            "requested": {
                "bits": args.bits,
                "group_size": args.group_size,
                "mode": args.mode,
                "dtype": args.dtype,
            },
            "output": str(args.out),
        }
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(
                f"convert dry-run: {args.source_repo}@{args.revision[:12]} "
                f"src={source.source_artifact_id} -> {args.out}"
            )
            print(
                f"  source bytes={storage['source_bytes']} free={storage['free_bytes']} "
                f"required>={storage['required_bytes']}"
            )
            print(
                f"  requested Q{args.bits} group={args.group_size} mode={args.mode} dtype={args.dtype}"
            )
        return 0

    provenance = convert_q4_variant(
        args.source_repo,
        args.revision,
        args.out,
        bits=args.bits,
        group_size=args.group_size,
        mode=args.mode,
        dtype=args.dtype,
    )
    if args.json:
        print(json.dumps(provenance, indent=2))
        return 0
    quant = provenance["quantization_params"]
    print(f"converted {args.source_repo}@{args.revision[:12]} -> {args.out}")
    print(
        f"  source artifact: {provenance['source_artifact']['repo_id']}"
        f"@{provenance['source_artifact']['revision'][:12]}"
    )
    print(f"  representation:  {provenance['representation']} precision={provenance['precision']}")
    print(
        f"  quantization:    {provenance['quantization']} bits={provenance['quantization_bits']} "
        f"group={provenance['quantization_group_size']}"
    )
    if quant.get("layer_schemes"):
        print(f"  actual schemes:  {quant['layer_schemes']}")
    print(f"  think-end id:    {provenance['think_end_token_id']}")
    print(f"  provenance:      {args.out}/ponderscope_variant.json")
    return 0


def _cmd_variant_audit(args: argparse.Namespace) -> int:
    from .conversion import audit_variant_load
    from .evidence.store import atomic_write_json

    report = audit_variant_load(
        args.source_repo,
        args.revision,
        args.path,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
    if args.out:
        atomic_write_json(Path(args.out), report)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"variant audit: {report['artifact_path']}")
    print(
        f"  representation={report['representation']} precision={report['precision']} "
        f"quant={report['quantization']} bits={report['quantization_bits']} "
        f"group={report['quantization_group_size']}"
    )
    print(f"  bits_per_weight={report.get('bits_per_weight')}")
    print(
        f"  source={report['source_artifact_id']} variant={report['weight_variant_id']} "
        f"derived_from={report['derived_from_source_artifact_id']}"
    )
    print(
        f"  load_audit_ok={report['load_audit'].get('ok')} "
        f"quantized_modules={report['load_audit'].get('n_quantized_modules')}"
    )
    print(f"  tokenizer semantic match: {report['tokenizer']['semantic_match']}")
    g = report["generation"]
    print(
        f"  generation: tokens={g['generated_tokens']} finish={g['finish_reason']} "
        f"think_end={g['think_end_reached']} eos={g['eos_observed']} error={g['error']}"
    )
    print(
        f"  load_seconds={report['load_seconds']:.2f} "
        f"peak_load={report['mlx_peak_after_load_bytes']} "
        f"peak_generate={report['mlx_peak_after_generate_bytes']}"
    )
    if args.out:
        print(f"  saved {args.out}")
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ponderscope", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="environment / runtime / model capability")
    p.add_argument("--backend", default="mlx")
    p.add_argument("--model-repo", default=MODEL_REPO_DEFAULT)
    p.add_argument("--revision", default=REVISION_DEFAULT)
    p.add_argument("--live", action="store_true")
    p.add_argument("--overhead", action="store_true")
    p.add_argument("--save", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("generate", help="generate and validate a versioned task pack")
    p.add_argument("--pack", default="tasks-v1")
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--n-per-family", type=int, default=4)
    p.add_argument("--split", default="dev")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", default=None)
    p.add_argument("--prompt-policy", default="pp-v1")
    p.add_argument("--out", default="tasks/tasks-v1.dev.jsonl")
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("run", help="run a declared deployment/task experiment")
    p.add_argument("--spec")
    p.add_argument("--name")
    p.add_argument("--task-pack", default="tasks-v1")
    p.add_argument("--prompt-policy", default="pp-v1")
    p.add_argument("--model-policy", default="")
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--n-per-family", type=int, default=4)
    p.add_argument("--task-seed", type=int, default=0)
    p.add_argument("--split", default="dev")
    p.add_argument("--greedy-repeats", type=int, default=1)
    p.add_argument("--sampled-seeds", nargs="*", type=int, default=None)
    p.add_argument("--sampled-repeats", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--n-probes", type=int, default=4)
    p.add_argument("--logprob-digest", action="store_true")
    p.add_argument("--capture-level", default="research", choices=["minimal", "research", "digest"])
    p.add_argument("--backend", default="mlx")
    p.add_argument("--model-repo", default=MODEL_REPO_DEFAULT)
    p.add_argument("--revision", default=REVISION_DEFAULT)
    p.add_argument(
        "--artifact-path",
        default=None,
        help="load a derived weight variant (e.g. a controlled Q4) from this directory",
    )
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--suffix", default=None)
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--analyze", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("analyze", help="analyze immutable saved evidence")
    p.add_argument("--run")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument(
        "--allow-unverified",
        action="store_true",
        help="analyze a run whose evidence seal does not verify (labelled forensic)",
    )
    p.add_argument("--report", action="store_true")
    p.set_defaults(func=_cmd_analyze)

    p = sub.add_parser("compare", help="compare two deployment configurations")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--mode", default="greedy", choices=["greedy", "sampled"])
    p.add_argument("--allow-confounded", action="store_true")
    p.add_argument("--require-complete", action="store_true")
    p.add_argument("--allow-unverified", action="store_true")
    p.add_argument("--decoding-intentional", action="store_true")
    p.add_argument("--prompt-policy-intentional", action="store_true")
    p.add_argument("--horizon-intentional", action="store_true")
    p.add_argument("--capture-intentional", action="store_true")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("report", help="regenerate Markdown/HTML from saved evidence")
    p.add_argument("--run")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--runs-dir", default="runs")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("verify", help="verify a sealed run's raw evidence and manifest hash")
    p.add_argument("--run")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser(
        "calibrate-prefix", help="qualify cap-prefix invariance for termination calibration"
    )
    p.add_argument("--backend", default="mlx")
    p.add_argument("--model-repo", default=MODEL_REPO_DEFAULT)
    p.add_argument("--revision", default=REVISION_DEFAULT)
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--n-per-family", type=int, default=1)
    p.add_argument("--task-seed", type=int, default=0)
    p.add_argument("--prompt-policy", default="pp-v1")
    p.add_argument("--caps", nargs="+", type=int, default=[256, 512, 1024, 2048])
    p.add_argument("--out", default="runs/calibration-prefix.phase1_2.json")
    p.set_defaults(func=_cmd_calibrate_prefix)

    p = sub.add_parser(
        "calibrate-replay",
        help="same-seed technical replay subset (not part of any primary N)",
    )
    p.add_argument("--spec", required=True)
    p.add_argument("--backend", default="mlx")
    p.add_argument("--model-repo", default=MODEL_REPO_DEFAULT)
    p.add_argument("--revision", default=REVISION_DEFAULT)
    p.add_argument("--families", nargs="+", default=["arith", "order", "sm"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default="runs/same-seed-replay.phase1_3b.json")
    p.set_defaults(func=_cmd_calibrate_replay)

    p = sub.add_parser("bundle", help="build a deterministic evidence archive for a sealed run")
    p.add_argument("--run")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--output", required=True)
    p.set_defaults(func=_cmd_bundle)

    p = sub.add_parser("model-policy", help="show a versioned model-policy profile")
    p.add_argument("--id", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_model_policy)

    p = sub.add_parser("survival", help="censor-aware time-to-closure for a saved run")
    p.add_argument("--run")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--allow-unverified", action="store_true")
    p.set_defaults(func=_cmd_survival)

    p = sub.add_parser("loop-diagnostics", help="descriptive loop structure for a saved run")
    p.add_argument("--run")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--allow-unverified", action="store_true")
    p.set_defaults(func=_cmd_loop_diagnostics)

    p = sub.add_parser(
        "convert",
        help="convert a pinned source snapshot to a derived MLX quantized weight variant",
    )
    p.add_argument("--source-repo", default="Qwen/Qwen3.5-4B")
    p.add_argument("--revision", required=True)
    p.add_argument("--out", required=True, help="new output directory for the derived artifact")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--mode", default="affine", choices=["affine", "mxfp4", "nvfp4", "mxfp8"])
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--dry-run", action="store_true", help="check source/storage without converting")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_convert)

    p = sub.add_parser(
        "variant-audit",
        help="real load + short-generation audit of a derived weight variant",
    )
    p.add_argument("--path", required=True, help="derived artifact directory")
    p.add_argument("--source-repo", default="Qwen/Qwen3.5-4B")
    p.add_argument("--revision", required=True)
    p.add_argument("--prompt", default="What is 17 + 25? Answer with the integer only.")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--out", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_variant_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
