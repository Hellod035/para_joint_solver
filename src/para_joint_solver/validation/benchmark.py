"""Core solver latency measurements, excluding validation, I/O and rendering."""

import platform
import sys
import time

import numpy as np

from ..visualization.config import inspection_range


def benchmark_solver(solver, samples=100, repeats=3, seed=42):
    if samples < 1 or repeats < 1:
        raise ValueError("samples and repeats must be >= 1")
    rng = np.random.default_rng(seed)
    bounds = inspection_range(solver.mechanism)
    q = rng.uniform(bounds[:, 0], bounds[:, 1], size=(samples, 2))
    states = [solver.at_pose(x) for x in q]
    ab = [s.q_ab for s in states]
    dq, tau = np.array([0.15, -0.12]), np.array([8.0, -3.0])
    operations = dict(
        ik=lambda i: solver.ik(q[i]),
        fk=lambda i: solver.solve(ab[i]),
        pose_and_jacobian=lambda i: solver.at_pose(q[i]),
        velocity_map=lambda i: states[i].velocity_to_rp(dq),
        torque_map=lambda i: states[i].torque_to_ab(tau),
    )
    result = {}
    for name, operation in operations.items():
        for i in range(min(10, samples)):
            operation(i)
        times = []
        for _ in range(repeats):
            for i in range(samples):
                start = time.perf_counter_ns()
                operation(i)
                times.append((time.perf_counter_ns() - start) / 1000)
        result[name] = dict(
            calls=len(times),
            median_us=float(np.median(times)),
            p95_us=float(np.percentile(times, 95)),
            mean_us=float(np.mean(times)),
        )
    return dict(
        samples=samples,
        repeats=repeats,
        seed=seed,
        timings=result,
        python=sys.version.split()[0],
        platform=platform.platform(),
        numpy=np.__version__,
        note="Core solver Python latency; excludes loading, MuJoCo validation and rendering. Velocity/torque maps reuse a solved State.",
    )
