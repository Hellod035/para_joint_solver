"""Numerical acceptance reports. A missing requested MuJoCo installation is a failure."""

from itertools import product
import json
from pathlib import Path

import numpy as np
from ..visualization.config import inspection_range

from ..simulation.backends import MujocoBackend


class Checks:
    def __init__(self):
        self.values = {}
        self.samples = []

    def start_sample(self, index, q_rp):
        self.samples.append(dict(index=index, q_rp=q_rp.tolist(), values={}))

    def add(self, name, error, tolerance):
        error = float(np.max(np.abs(error)))
        self.samples[-1]["values"][name] = error
        previous = self.values.get(name, {}).get("max_error", 0.0)
        value = max(previous, error) if np.isfinite(error) else float("inf")
        self.values[name] = dict(max_error=value, tolerance=tolerance, passed=value <= tolerance)


def sample_poses(mechanism, grid=5, random=12, seed=42):
    if grid < 2 or random < 0:
        raise ValueError("grid must be >= 2 and random samples >= 0")
    bounds = inspection_range(mechanism)
    poses = [mechanism.home_rp.copy()]
    poses += [
        np.array(q) for q in product(np.linspace(*bounds[0], grid), np.linspace(*bounds[1], grid))
    ]
    rng = np.random.default_rng(seed)
    poses += list(rng.uniform(bounds[:, 0], bounds[:, 1], size=(random, 2)))
    # A continuous interior trajectory tests warm-start FK across changing motor angles.
    center, amplitude = bounds.mean(axis=1), np.diff(bounds, axis=1).ravel() * 0.38
    poses += [
        center + amplitude * [np.sin(t), np.sin(1.7 * t)] for t in np.linspace(0, 2 * np.pi, 31)
    ]
    return poses


def validate_solver(solver, mujoco=True, grid=5, random=12, seed=42):
    m = solver.mechanism
    checks, failures, ref = Checks(), [], None
    versions = {}
    if mujoco:
        try:
            ref = MujocoBackend(solver)
            versions["mujoco"] = ref.mj.__version__
        except Exception as exc:
            failures.append(dict(stage="mujoco", error=f"{type(exc).__name__}: {exc}"))
    rng = np.random.default_rng(seed)
    poses = sample_poses(m, grid, random, seed)
    previous = None
    for idx, q in enumerate(poses):
        checks.start_sample(idx, q)
        try:
            state = solver.at_pose(q)
            # Grid/random samples are disconnected; only the final trajectory is warm-started.
            fk = solver.solve(state.q_ab, previous if idx > len(poses) - 31 else None)
            previous = fk
            checks.add("core.closure_m", state.closure_error, m.tolerance)
            checks.add("core.fk_rad", fk.q_rp - q, 1e-7)
            checks.add("core.condition", state.condition, m.max_condition)
            eps = 1e-6
            jfd = np.column_stack(
                [
                    (solver.ik(q + np.eye(2)[k] * eps) - solver.ik(q - np.eye(2)[k] * eps))
                    / (2 * eps)
                    for k in range(2)
                ]
            )
            checks.add("core.jacobian_finite_difference", jfd - state.J, 1e-5)
            dq = rng.uniform(-0.3, 0.3, 2)
            tau = rng.uniform(-12, 12, 2)
            dq_rp, tau_ab = state.velocity_to_rp(dq), state.torque_to_ab(tau)
            # AB finite difference independently verifies the requested forward velocity map.
            qplus = solver.fk(state.q_ab + eps * dq, state)
            qminus = solver.fk(state.q_ab - eps * dq, state)
            checks.add(
                "core.velocity_finite_difference_rad_s", (qplus - qminus) / (2 * eps) - dq_rp, 2e-5
            )
            checks.add("core.virtual_power_W", tau @ dq_rp - tau_ab @ dq, 1e-10)
            b_core, c_core, _, _, _ = solver.geometry(q, state.q_ab)
            if ref is not None:
                try:
                    r = ref.solve(state.q_ab)  # Independent FK starts at configured home.
                    b, c, _, _ = ref.points_and_jacobians(q, state.q_ab)
                    checks.add("mujoco.endpoints_m", np.r_[b - b_core, c - c_core], 1e-10)
                    checks.add("mujoco.closure_m", r.closure_error, m.tolerance)
                    checks.add("mujoco.fk_rad", r.q_rp - state.q_rp, 1e-7)
                    checks.add("mujoco.jacobian", r.J - state.J, 1e-6)
                    checks.add("mujoco.velocity_rad_s", r.velocity_to_rp(dq) - dq_rp, 1e-6)
                    checks.add("mujoco.torque_Nm", r.torque_to_ab(tau) - tau_ab, 1e-5)
                    checks.add(
                        "mujoco.independent_power_W", tau @ r.velocity_to_rp(dq) - tau_ab @ dq, 1e-6
                    )
                    g = ref.constrained_velocity_map(r.q_rp, state.q_ab)
                    checks.add("mujoco.closed_chain_velocity_rad_s", g @ dq - dq_rp, 1e-6)
                    checks.add("mujoco.closed_chain_torque_Nm", g.T @ tau - tau_ab, 1e-5)
                except Exception as exc:
                    failures.append(
                        dict(
                            stage="mujoco",
                            sample=idx,
                            q_rp=q.tolist(),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
        except Exception as exc:
            failures.append(
                dict(
                    stage="core", sample=idx, q_rp=q.tolist(), error=f"{type(exc).__name__}: {exc}"
                )
            )
    return dict(
        robot=m.id,
        passed=not failures and all(c["passed"] for c in checks.values.values()),
        sample_count=len(poses),
        seed=seed,
        grid=grid,
        versions=versions,
        checks=checks.values,
        samples=checks.samples,
        failures=failures,
    )


def write_json(report, destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)

    # JSON standard has no Infinity or NaN. Preserve failures as strings for portable readers.
    def finite(obj):
        if isinstance(obj, dict):
            return {k: finite(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [finite(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return str(obj)
        return obj

    path.write_text(
        json.dumps(finite(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return path
