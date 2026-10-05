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
        runs_dir=args.runs_dir,
        run_suffix=args.suffix,
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
    analysis = analyze_run(store)
    print(
        f"analyzed {store.run_id}: {analysis['n_generations']} generations, {analysis['n_probes']} probes"
    )
    if args.report:
        generate_report(store, analysis)
        print(f"report: {store.path / 'summary.md'}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from .analysis import compare_configs
    from .evidence.run import RunStore

    a = RunStore.load(args.a)
    b = RunStore.load(args.b)
    comparison = compare_configs(a, b, mode=args.mode)
    (a.path / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(
        f"compared {a.config_id} vs {b.config_id} [{args.mode}], matched={comparison['n_matched']}"
    )
    for metric, m in comparison["metrics"].items():
        d = m["delta"]
        print(
            f"  {metric:<26} delta={d.get('mean')} ci=[{d.get('lo')}, {d.get('hi')}] "
            f"noise={m['noise_scale']} -> {m['classification']}"
        )
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ponderscope", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="environment / runtime / model capability")
    p.add_argument("--backend", default="mlx")
    p.add_argument("--model-repo", default=MODEL_REPO_DEFAULT)
    p.add_argument("--revision", default=REVISION_DEFAULT)
    p.add_argument("--live", action="store_true")
    p.add_argument("--overhead", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("generate", help="generate and validate a versioned task pack")
    p.add_argument("--pack", default="tasks-v1")
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--n-per-family", type=int, default=4)
    p.add_argument("--split", default="dev")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", default=None)
    p.add_argument("--out", default="tasks/tasks-v1.dev.jsonl")
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("run", help="run a declared deployment/task experiment")
    p.add_argument("--spec")
    p.add_argument("--name")
    p.add_argument("--task-pack", default="tasks-v1")
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
    p.add_argument("--backend", default="mlx")
    p.add_argument("--model-repo", default=MODEL_REPO_DEFAULT)
    p.add_argument("--revision", default=REVISION_DEFAULT)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--suffix", default=None)
    p.add_argument("--analyze", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("analyze", help="analyze immutable saved evidence")
    p.add_argument("--run")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--report", action="store_true")
    p.set_defaults(func=_cmd_analyze)

    p = sub.add_parser("compare", help="compare two deployment configurations")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--mode", default="greedy", choices=["greedy", "sampled"])
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("report", help="regenerate Markdown/HTML from saved evidence")
    p.add_argument("--run")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--runs-dir", default="runs")
    p.set_defaults(func=_cmd_report)

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
