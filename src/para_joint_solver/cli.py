"""Command-line entry points; simulation/visual imports stay optional for core users."""

import argparse
import json
from pathlib import Path
import sys

from . import SolverError, available_robots, load_solver


def parser():
    p = argparse.ArgumentParser(prog="pjs", description="Configurable parallel joint solver")
    p.add_argument("--robots-dir", help="Robot registry root (or set PJS_ROBOTS_DIR)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List configured robots")
    solve = sub.add_parser("solve", help="Compute the three RP/AB maps as JSON")
    solve.add_argument("robot")
    q = solve.add_mutually_exclusive_group(required=True)
    q.add_argument("--q-ab", nargs=2, type=float, metavar=("A", "B"))
    q.add_argument("--q-rp", nargs=2, type=float, metavar=("ROLL", "PITCH"))
    solve.add_argument("--seed-rp", nargs=2, type=float)
    solve.add_argument("--dq-ab", nargs=2, type=float)
    solve.add_argument("--tau-rp", nargs=2, type=float)
    export = sub.add_parser("export", help="Export a portable MuJoCo closed-chain model")
    export.add_argument("robot")
    export.add_argument("--output", type=Path, required=True)
    for command in ("validate", "benchmark"):
        validate = sub.add_parser(
            command,
            help="Numerical checks and optional error plots"
            if command == "validate"
            else "Checks plus core solver latency benchmark",
        )
        target = validate.add_mutually_exclusive_group(required=True)
        target.add_argument("--robot")
        target.add_argument("--all", action="store_true")
        validate.add_argument("--core-only", action="store_true", help="Skip MuJoCo checks")
        validate.add_argument("--grid", type=int, default=5)
        validate.add_argument("--random", type=int, default=12)
        validate.add_argument("--seed", type=int, default=42)
        validate.add_argument("--plot", action="store_true", help="Plot per-pose numerical errors")
        validate.add_argument("--output", type=Path, default=Path("artifacts") / command)
        if command == "benchmark":
            validate.add_argument("--samples", type=int, default=100)
            validate.add_argument("--repeats", type=int, default=3)
    inspect = sub.add_parser("inspect", help="Real-time AB/RP sliders in a local browser")
    inspect.add_argument("robot")
    inspect.add_argument("--port", type=int, default=8765)
    inspect.add_argument("--no-browser", action="store_true")
    return p


def run(args):
    if args.command == "list":
        print("\n".join(available_robots(args.robots_dir)))
        return 0
    if args.command in {"validate", "benchmark"}:
        from .validation.numerical import validate_solver, write_json
        from .validation.report import write_html

        names = available_robots(args.robots_dir) if args.all else [args.robot]
        if not names:
            raise ValueError("No robots found")
        if args.grid < 2 or args.random < 0:
            raise ValueError("--grid must be >= 2; --random must be >= 0")
        reports = []
        for name in names:
            directory = args.output / Path(name).name
            try:
                solver = load_solver(name, args.robots_dir)
                report = validate_solver(
                    solver,
                    mujoco=not args.core_only,
                    grid=args.grid,
                    random=args.random,
                    seed=args.seed,
                )
                if args.command == "benchmark":
                    from .validation.benchmark import benchmark_solver

                    try:
                        report["benchmark"] = benchmark_solver(
                            solver,
                            samples=args.samples,
                            repeats=args.repeats,
                            seed=args.seed,
                        )
                    except Exception as exc:
                        report["passed"] = False
                        report["failures"].append(
                            dict(stage="benchmark", error=f"{type(exc).__name__}: {exc}")
                        )
                if args.plot:
                    from .validation.plots import plot_errors

                    try:
                        report["plots"] = [plot_errors(report, directory)]
                    except Exception as exc:
                        report["passed"] = False
                        report["failures"].append(
                            dict(stage="plot", error=f"{type(exc).__name__}: {exc}")
                        )
            except Exception as exc:
                report = dict(
                    robot=Path(name).name,
                    passed=False,
                    checks={},
                    failures=[dict(stage="load", error=f"{type(exc).__name__}: {exc}")],
                )
            write_json(report, directory / "report.json")
            reports.append(report)
            print(
                f"{'PASS' if report['passed'] else 'FAIL'} {name}: {directory / 'report.json'}",
                flush=True,
            )
            for failure in report["failures"][:3]:
                print(f"  {failure}", flush=True)
            for check, metric in report["checks"].items():
                if not metric["passed"]:
                    print(f"  {check}: {metric}", flush=True)
        passed = all(r["passed"] for r in reports)
        write_json(dict(passed=passed, robots=reports), args.output / "summary.json")
        print(write_html(reports, args.output))
        return 0 if passed else 1
    solver = load_solver(args.robot, args.robots_dir)
    if args.command == "solve":
        state = (
            solver.at_pose(args.q_rp)
            if args.q_rp is not None
            else solver.solve(args.q_ab, args.seed_rp)
        )
        result = dict(
            robot=solver.mechanism.id,
            q_ab=state.q_ab.tolist(),
            q_rp=state.q_rp.tolist(),
            J_ab_rp=state.J.tolist(),
            condition=state.condition,
            closure_error_m=state.closure_error,
        )
        if args.dq_ab is not None:
            result["dq_rp"] = state.velocity_to_rp(args.dq_ab).tolist()
        if args.tau_rp is not None:
            result["tau_ab"] = state.torque_to_ab(args.tau_rp).tolist()
        print(json.dumps(result, indent=2))
    elif args.command == "export":
        from .simulation.models import export_model

        print(export_model(solver, args.output))
    elif args.command == "inspect":
        from .visualization.inspector import serve

        serve(solver, port=args.port, open_browser=not args.no_browser)
    return 0


def main(argv=None):
    try:
        return run(parser().parse_args(argv))
    except (SolverError, ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"pjs: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
