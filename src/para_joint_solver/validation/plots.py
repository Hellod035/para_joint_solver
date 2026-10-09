"""Numerical error plots; no renderer, mesh assets, or OpenGL context required."""

from pathlib import Path

import numpy as np


def plot_errors(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Position", "rad", [
            ("core.fk_rad", "Core FK"),
            ("mujoco.fk_rad", "MuJoCo FK"),
        ]),
        ("Closure and geometry", "m", [
            ("core.closure_m", "Core closure"),
            ("mujoco.closure_m", "MuJoCo closure"),
            ("mujoco.endpoints_m", "Attachment points"),
        ]),
        ("Velocity", "rad/s", [
            ("core.velocity_finite_difference_rad_s", "Finite difference"),
            ("mujoco.velocity_rad_s", "MuJoCo FK Jacobian"),
            ("mujoco.closed_chain_velocity_rad_s", "MuJoCo constraint Jacobian"),
        ]),
        ("Torque", "N·m", [
            ("mujoco.torque_Nm", "MuJoCo FK Jacobian"),
            ("mujoco.closed_chain_torque_Nm", "MuJoCo constraint Jacobian"),
        ]),
        ("Jacobian", "rad/rad", [
            ("core.jacobian_finite_difference", "Finite difference"),
            ("mujoco.jacobian", "MuJoCo FK Jacobian"),
        ]),
        ("Virtual power", "W", [
            ("core.virtual_power_W", "Core"),
            ("mujoco.independent_power_W", "MuJoCo"),
        ]),
    ]
    samples = report.get("samples", [])
    x = [sample["index"] for sample in samples]
    failed = sorted(
        {f["sample"] for f in report["failures"] if "sample" in f}
        | {
            s["index"] for s in samples
            if any(
                not np.isfinite(value) or value > report["checks"][name]["tolerance"]
                for name, value in s["values"].items()
            )
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout="constrained")
    try:
        for ax, (title, unit, series) in zip(axes.flat, panels):
            ceiling = 1e-14
            for name, label in series:
                if name not in report["checks"]:
                    continue
                values = np.array([s["values"].get(name, np.nan) for s in samples], dtype=float)
                values[~np.isfinite(values)] = np.nan
                finite = values[np.isfinite(values)]
                ceiling = max(
                    ceiling, report["checks"][name]["tolerance"],
                    float(finite.max()) if finite.size else 0,
                )
                line, = ax.plot(x, values, ".-", linewidth=1, markersize=3, label=label)
                ax.axhline(
                    report["checks"][name]["tolerance"],
                    color=line.get_color(), linestyle="--", alpha=0.5, linewidth=1,
                )
            if ax.lines:
                ax.legend(fontsize=8, loc="best")
            else:
                ax.text(0.5, 0.5, "No checks available", ha="center", transform=ax.transAxes)
            if failed:
                ax.scatter(
                    failed, np.full(len(failed), 0.04), marker="x", color="#c73333",
                    transform=ax.get_xaxis_transform(), zorder=5,
                )
            ax.set_yscale("symlog", linthresh=1e-14)
            ax.set_ylim(0, ceiling * 3)
            ax.set(title=title, xlabel="Validation sample", ylabel=f"Max absolute error / {unit}")
            ax.grid(alpha=0.2)
        fig.suptitle(
            f"{report['robot']} · {'PASS' if report['passed'] else 'FAIL'} · numerical errors\n"
            "Dashed: tolerance · red crosses: failed samples · linear scale below 1e-14",
            fontsize=13,
        )
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = "errors.png"
        fig.savefig(directory / filename, dpi=150)
        return filename
    finally:
        plt.close(fig)
