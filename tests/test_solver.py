from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from para_joint_solver import (
    ConfigurationError,
    ConvergenceError,
    LimitViolation,
    SingularConfiguration,
    UnreachablePose,
    available_robots,
    load_solver,
)

from para_joint_solver.visualization.config import inspection_range, load_visuals

ROBOTS = available_robots()


def test_benchmark_times_only_core_maps():
    from para_joint_solver.validation.benchmark import benchmark_solver

    result = benchmark_solver(load_solver("g1"), samples=3, repeats=1)
    assert set(result["timings"]) == {
        "ik", "fk", "pose_and_jacobian", "velocity_map", "torque_map",
    }
    for timing in result["timings"].values():
        assert timing["calls"] == 3 and timing["p95_us"] >= timing["median_us"] > 0


def modify(directory, filename, callback):
    path = directory / filename
    data = yaml.safe_load(path.read_text())
    callback(data)
    path.write_text(yaml.safe_dump(data))


@pytest.mark.parametrize("name", ROBOTS)
def test_mechanism_is_self_contained(name, tmp_path):
    baseline = load_solver(name)
    path = tmp_path / "mechanism.yaml"
    shutil.copyfile(baseline.mechanism.directory / "mechanism.yaml", path)
    solver = load_solver(path)
    q = baseline.mechanism.home_rp + [0.02, -0.03]
    expected, actual = baseline.at_pose(q), solver.at_pose(q)
    np.testing.assert_allclose(solver.fk(actual.q_ab), q, atol=1e-8)
    np.testing.assert_allclose(actual.J, expected.J, atol=1e-12)
    np.testing.assert_allclose(
        actual.velocity_to_rp([0.2, -0.1]), expected.velocity_to_rp([0.2, -0.1])
    )
    np.testing.assert_allclose(actual.torque_to_ab([8, -3]), expected.torque_to_ab([8, -3]))
    # Even a broken visual configuration must not affect core loading or solving.
    (tmp_path / "visuals.yaml").write_text("this is not a visual configuration")
    np.testing.assert_allclose(load_solver(path).ik(q), actual.q_ab)


def test_output_limit_is_optional_and_inspection_does_not_limit_solver(tmp_path):
    original = load_solver("waist_symmetric")
    path = tmp_path / "mechanism.yaml"
    shutil.copyfile(original.mechanism.directory / "mechanism.yaml", path)
    solver = load_solver(path)
    assert not np.isfinite(solver.mechanism.rp_limits).any()
    # Outside the supplied inspection range, but still a valid closed-chain pose.
    state = solver.at_pose([0.3, -0.1])
    np.testing.assert_allclose(solver.fk(state.q_ab), state.q_rp, atol=1e-8)
    d = yaml.safe_load(path.read_text())
    next(j for j in d["output"]["joints"] if j["name"] == "roll")["limits"] = [-0.2, 0.2]
    path.write_text(yaml.safe_dump(d))
    with pytest.raises(LimitViolation):
        load_solver(path).at_pose(state.q_rp)


@pytest.mark.parametrize("name", ROBOTS)
def test_three_maps_and_continuity(name):
    solver = load_solver(name)
    bounds = inspection_range(solver.mechanism)
    center, amplitude = bounds.mean(axis=1), np.diff(bounds).ravel() * 0.38
    previous = None
    for t in np.linspace(0, 2 * np.pi, 30):
        rp = center + amplitude * [np.sin(t), np.sin(1.7 * t)]
        state = solver.at_pose(rp)
        previous = solver.solve(state.q_ab, previous)
        np.testing.assert_allclose(previous.q_rp, rp, atol=1e-8)
        dq, tau = np.array([0.2, -0.1]), np.array([8.0, -3.0])
        eps = 1e-6
        fd = (solver.fk(state.q_ab + eps * dq, state) - solver.fk(state.q_ab - eps * dq, state)) / (
            2 * eps
        )
        np.testing.assert_allclose(state.velocity_to_rp(dq), fd, atol=2e-6)
        # Finite displacement work, not only an algebraic J^-1/J^-T identity.
        np.testing.assert_allclose(state.torque_to_ab(tau) @ dq, tau @ fd, atol=2e-5)
        assert state.closure_error < 1e-10
    with pytest.raises(ValueError):
        previous.q_rp[0] = 5


def test_invalid_inputs_and_unreachable_geometry():
    solver = load_solver("g1")
    for invalid in ([float("nan"), 0], [0], [0, 0, 0]):
        with pytest.raises(ConfigurationError):
            solver.solve(invalid)
    with pytest.raises(LimitViolation):
        solver.solve([100, 0])
    with pytest.raises((UnreachablePose, LimitViolation)):
        solver.at_pose([0.26, -0.86])
    with pytest.raises((ConvergenceError, UnreachablePose, LimitViolation)):
        solver.solve([1.49, -1.49])


def test_forward_solve_near_zero_seed(calibrated_robot):
    solver = load_solver(calibrated_robot)
    state = solver.at_pose([0.025, 0.04])
    np.testing.assert_allclose(solver.fk(state.q_ab, seed=[-1e-16, -2e-16]), state.q_rp, atol=1e-9)


@pytest.mark.parametrize(
    "filename,extra",
    [
        ("mechanism.yaml", {"units": {"length": "m", "angle": "rad"}}),
        ("mechanism.yaml", {"inertias": {}}),
        ("visuals.yaml", {"style": {}}),
        ("visuals.yaml", {"scene": {}}),
    ],
)
def test_only_supported_configuration_fields(robot_copy, filename, extra):
    modify(robot_copy, filename, lambda d: d.update(extra))
    with pytest.raises(ConfigurationError, match="unknown keys"):
        if filename == "visuals.yaml":
            load_visuals(load_solver(robot_copy).mechanism)
        else:
            load_solver(robot_copy)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["branches"]["A"].update(rod_length=-1),
        lambda d: d["branches"]["A"]["motor"].update(scale=0),
        lambda d: d["branches"]["A"]["motor"].update(axis=[0, 2, 0]),
        lambda d: d["assembly"].update(motor_reference=[2, 0]),
        lambda d: d["output"]["joints"][0].update(position=[0, 0]),
        lambda d: d["branches"]["A"].update(platform_anchor=[1, 2, 3]),
        lambda d: d["branches"]["A"]["motor"].update(offset=0),
        lambda d: d["output"].update(tool={}),
        lambda d: d["output"]["joints"][0].update(origin={}),
    ],
)
def test_invalid_mechanisms(robot_copy, mutation):
    modify(robot_copy, "mechanism.yaml", mutation)
    with pytest.raises(ConfigurationError):
        load_solver(robot_copy)


def test_missing_and_escaping_mesh(robot_copy):
    for name in ("meshes/missing.obj", "../outside.obj"):
        modify(robot_copy, "visuals.yaml", lambda d: d["bodies"][0].update(file=name))
        with pytest.raises(ConfigurationError, match="Mesh must exist"):
            load_visuals(load_solver(robot_copy).mechanism)


def test_singular_mechanism_is_rejected(robot_copy):
    def mutate(d):
        d["branches"]["B"] = deepcopy(d["branches"]["A"])

    modify(robot_copy, "mechanism.yaml", mutate)
    with pytest.raises(SingularConfiguration):
        load_solver(robot_copy)


def test_global_geometry_rigid_transform_does_not_change_maps(robot_copy):
    baseline = load_solver(robot_copy)
    rotation = Rotation.from_euler("xyz", [0.37, -0.22, 0.18])
    translation = np.array([0.12, -0.07, 0.03])

    def mutate(d):
        joints = d["output"]["joints"] + [b["motor"] for b in d["branches"].values()]
        for joint in joints:
            joint["position"] = (rotation.apply(joint["position"]) + translation).tolist()
            joint["axis"] = rotation.apply(joint["axis"]).tolist()
        for branch in d["branches"].values():
            for endpoint in ("crank_tip", "platform_anchor"):
                branch[endpoint] = (rotation.apply(branch[endpoint]) + translation).tolist()

    modify(robot_copy, "mechanism.yaml", mutate)
    rotated = load_solver(robot_copy)
    for q in ([0, 0], [0.08, -0.12], [-0.1, 0.1]):
        a, b = baseline.at_pose(q), rotated.at_pose(q)
        np.testing.assert_allclose(a.q_ab, b.q_ab, atol=1e-12)
        np.testing.assert_allclose(a.J, b.J, atol=1e-12)


def test_reversed_output_chain_keeps_reference_points_in_base_frame(robot_copy):
    def mutate(d):
        d["output"]["joints"].reverse()

    modify(robot_copy, "mechanism.yaml", mutate)
    modify(
        robot_copy,
        "visuals.yaml",
        lambda d: d.update(inspection=dict(roll=[-0.05, 0.05], pitch=[-0.08, 0.08])),
    )
    solver = load_solver(robot_copy)
    from para_joint_solver.validation.numerical import validate_solver

    report = validate_solver(solver, mujoco=False, grid=3, random=4)
    assert report["passed"], report["failures"]


def test_example_assets_are_self_contained():
    for name in ROBOTS:
        solver = load_solver(name)
        mesh_bodies = [b for b in load_visuals(solver.mechanism)["bodies"] if b["type"] == "mesh"]
        assert len(mesh_bodies) == 6
        assert {b["frame"] for b in mesh_bodies} == {
            "base",
            "platform",
            "motor_A",
            "motor_B",
            "rod_A",
            "rod_B",
        }
        assert {Path(b["file"]).stem for b in mesh_bodies} == {
            "base_platform",
            "output_platform",
            "crank_A",
            "crank_B",
            "rod_A",
            "rod_B",
        }
        for b in mesh_bodies:
            assert (solver.mechanism.directory / b["file"]).stat().st_size > 100
        assert not list(solver.mechanism.directory.glob("**/SOURCES.json"))
