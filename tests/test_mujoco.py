from pathlib import Path
import shutil
import yaml

import numpy as np
import pytest

from para_joint_solver import available_robots, load_solver
from para_joint_solver.simulation.backends import MujocoBackend
from para_joint_solver.simulation.models import export_model
from para_joint_solver.validation.numerical import validate_solver

pytestmark = pytest.mark.integration
mujoco = pytest.importorskip("mujoco")


@pytest.mark.parametrize("name", available_robots())
def test_engine_geometry_and_constraints(name, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Kinematic validation must not run dynamics")

    for function in ("mj_step", "mj_forward", "mj_inverse"):
        monkeypatch.setattr(mujoco, function, forbidden)
    report = validate_solver(load_solver(name), grid=3, random=4)
    assert report["passed"], (
        report["failures"],
        {k: v for k, v in report["checks"].items() if not v["passed"]},
    )


def test_reference_fk_does_not_call_core_solver(monkeypatch, calibrated_robot):
    solver = load_solver(calibrated_robot)
    state = solver.at_pose([0.075, -0.08])

    def forbidden(*args, **kwargs):
        raise AssertionError("Reference path called the analytic solver")

    for name in ("solve", "ik", "geometry", "closure", "at_pose"):
        monkeypatch.setattr(solver, name, forbidden)
    engine = MujocoBackend(solver)
    ref = engine.solve(state.q_ab)
    np.testing.assert_allclose(ref.q_rp, state.q_rp, atol=1e-9)
    g = engine.constrained_velocity_map(ref.q_rp, state.q_ab)
    np.testing.assert_allclose(g @ state.J, np.eye(2), atol=1e-9)


@pytest.mark.parametrize("name", available_robots())
def test_reference_axis_lines_and_endpoints_are_given_in_base_coordinates(name):
    solver = load_solver(name)
    m = solver.mechanism
    b, c, *_ = solver.geometry(m.home_rp, m.motor_reference)
    engine = MujocoBackend(solver)
    bm, cm, *_ = engine.points_and_jacobians(m.home_rp, m.motor_reference)
    expected_b = np.array([branch.crank_tip for branch in m.branches])
    expected_c = np.array([branch.platform_anchor for branch in m.branches])
    for actual, expected in ((b, expected_b), (bm, expected_b), (c, expected_c), (cm, expected_c)):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    for joint in (*m.output, *(branch.motor for branch in m.branches)):
        name = joint.name if joint.name in {"roll", "pitch"} else f"motor_{joint.name}"
        idx = engine.model.joint(name).id
        np.testing.assert_allclose(engine.data.xanchor[idx], joint.position, atol=1e-12)
        np.testing.assert_allclose(engine.data.xaxis[idx], joint.axis, atol=1e-12)


def test_reference_maps_are_independent_of_inertial_placeholders():
    solver = load_solver("waist_offset")
    state = solver.at_pose([0.07, -0.08])
    engine = MujocoBackend(solver)
    before = engine.constrained_velocity_map(state.q_rp, state.q_ab)
    # Change each body's placeholders independently over several orders of magnitude.
    factors = np.geomspace(0.01, 100, engine.model.nbody - 1)
    engine.model.body_mass[1:] *= factors
    engine.model.body_inertia[1:] *= factors[:, None]
    mujoco.mj_setConst(engine.model, engine.data)
    ref = engine.solve(state.q_ab)
    after = engine.constrained_velocity_map(ref.q_rp, state.q_ab)
    np.testing.assert_allclose(after, before, atol=1e-9)
    np.testing.assert_allclose(ref.q_rp, state.q_rp, atol=1e-9)
    np.testing.assert_allclose(after @ [0.2, -0.1], state.velocity_to_rp([0.2, -0.1]), atol=1e-9)
    np.testing.assert_allclose(after.T @ [8, -3], state.torque_to_ab([8, -3]), atol=1e-9)
    assert engine.data.time == 0 and engine.model.nu == 0


@pytest.mark.parametrize("name", ["g1", "tiangong"])
def test_fk_at_exact_output_limits_and_back(name):
    solver = load_solver(name)
    engine = MujocoBackend(solver)
    home = solver.at_pose(solver.mechanism.home_rp)
    for axis in range(2):
        for bound in solver.mechanism.rp_limits[axis]:
            q = home.q_rp.copy()
            q[axis] = bound
            endpoint = solver.at_pose(q)
            for backend in (solver, engine):
                actual = backend.solve(endpoint.q_ab, seed=home.q_rp)
                np.testing.assert_allclose(actual.q_rp, q, atol=1e-9)
                assert actual.closure_error < solver.mechanism.tolerance
                returned = backend.solve(home.q_ab, seed=actual.q_rp)
                np.testing.assert_allclose(returned.q_rp, home.q_rp, atol=1e-9)


def test_reference_needs_no_visuals_and_ignores_mesh_placement(tmp_path):
    solver = load_solver("waist_symmetric")
    state = solver.at_pose([0.08, -0.12])
    path = tmp_path / "mechanism.yaml"
    shutil.copyfile(solver.mechanism.directory / "mechanism.yaml", path)
    bare = MujocoBackend(load_solver(path))
    assert bare.model.ngeom == bare.model.nmesh == 0
    assert not bare.model.jnt_limited[bare.model.joint("roll").id]
    np.testing.assert_allclose(bare.solve(state.q_ab).q_rp, state.q_rp, atol=1e-8)
    # A deliberately unrelated display offset changes the geom, never a joint/site.
    v = dict(
        bodies=[
            dict(
                frame="platform",
                type="box",
                size=[0.02, 0.03, 0.01],
                pose=dict(xyz=[1, 2, 3], quat_wxyz=[0, 1, 0, 0]),
            )
        ]
    )
    (tmp_path / "visuals.yaml").write_text(yaml.safe_dump(v))
    display = MujocoBackend(load_solver(path), visual=True)
    np.testing.assert_allclose(display.solve(state.q_ab).J, bare.solve(state.q_ab).J, atol=1e-10)
    np.testing.assert_allclose(
        display.constrained_velocity_map(state.q_rp, state.q_ab),
        bare.constrained_velocity_map(state.q_rp, state.q_ab),
        atol=1e-10,
    )


@pytest.mark.parametrize("name", available_robots())
def test_export_is_portable_with_meshes(name, tmp_path):
    original = tmp_path / "original"
    path = export_model(load_solver(name), original)
    assert "external/" not in path.read_text()
    assert str(Path.cwd()) not in path.read_text()
    destination = tmp_path / "relocated"
    shutil.move(original, destination)
    model = mujoco.MjModel.from_xml_path(str(destination / "model.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    assert model.neq == 2 and model.nv == 8 and model.nmesh == model.ngeom == 6
    assert model.nu == 0
    assert np.max(np.abs(data.efc_pos)) < 1e-8


def test_stl_axes_follow_source_frame_with_visual_pose_and_scale(calibrated_robot):
    from scipy.spatial.transform import Rotation
    from para_joint_solver.visualization.annotations import Annotations

    path = calibrated_robot / "visuals.yaml"
    visual = yaml.safe_load(path.read_text())
    rotation = Rotation.from_euler("xyz", [0.3, -0.4, 0.2]).as_matrix()
    quaternion = Rotation.from_matrix(rotation).as_quat(scalar_first=True).tolist()
    offset = np.array([0.035, -0.023, 0.012])
    for body in visual["bodies"]:
        body["pose"] = dict(xyz=offset.tolist(), quat_wxyz=quaternion)
        body["scale"] = [0.8, 1.2, 0.7]
    path.write_text(yaml.safe_dump(visual))
    solver = load_solver(calibrated_robot)
    engine = MujocoBackend(solver, visual=True)
    annotations = Annotations(engine)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, offset
    for q in (solver.mechanism.home_rp, [0.075, -0.08]):
        state = solver.at_pose(q)
        engine.set_configuration(state.q_rp, state.q_ab, close_rods=True)
        *_, frames = solver.geometry(state.q_rp, state.q_ab)
        for mesh in annotations.meshes:
            if mesh["frame"] not in frames:
                continue  # Passive rod rotations are tested separately below.
            actual = annotations.mesh_transform(mesh)
            expected = frames[mesh["frame"]] @ transform
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            # MuJoCo's compiled mesh geom has a different origin after centering.
            assert not np.allclose(actual[:3, 3], engine.data.geom_xpos[int(mesh["id"])])


def test_rod_stl_axes_use_passive_body_frame():
    from para_joint_solver.core.geometry import align_z
    from para_joint_solver.visualization.annotations import Annotations

    solver = load_solver("igris_c_waist")
    engine = MujocoBackend(solver, visual=True)
    annotations = Annotations(engine)
    for q in (solver.mechanism.home_rp, [0.04, -0.07]):
        state = solver.at_pose(q)
        engine.set_configuration(state.q_rp, state.q_ab, close_rods=True)
        b, c, *_ = solver.geometry(state.q_rp, state.q_ab)
        for i, name in enumerate(("A", "B")):
            mesh = next(m for m in annotations.meshes if m["frame"] == f"rod_{name}")
            frame = annotations.mesh_transform(mesh)
            np.testing.assert_allclose(frame[:3, 3], b[i], atol=1e-12)
            np.testing.assert_allclose(frame[:3, 2], (c[i] - b[i]) / np.linalg.norm(c[i] - b[i]),
                                       atol=1e-12)
            if np.array_equal(q, solver.mechanism.home_rp):
                np.testing.assert_allclose(frame[:3, :3], align_z(c[i] - b[i]), atol=1e-12)


@pytest.mark.parametrize("name", ["tiangong", "waist_offset"])
def test_reference_markers_coincide_with_engine_joint_axes(name):
    from para_joint_solver.visualization.annotations import Annotations

    solver = load_solver(name)
    m = solver.mechanism
    engine = MujocoBackend(solver, visual=True)
    engine.set_configuration(m.home_rp, m.motor_reference, close_rods=True)
    annotations = Annotations(engine)
    paths = [f"output.joints[{i}]" for i in range(2)] + [f"branches.{n}.motor" for n in "AB"]
    options = annotations.settings(dict(
        base_axes=False,
        parameters=[f"{path}.{field}" for path in paths for field in ("position", "axis")]
                   + ["assembly.home_rp", "assembly.motor_reference"],
    ))
    scene = mujoco.MjvScene(engine.model, maxgeom=200)
    labels, _ = annotations.draw(scene, 0.04, options)
    for joint in (*m.output, *(b.motor for b in m.branches)):
        engine_name = f"motor_{joint.name}" if joint.name in "AB" else joint.name
        idx = engine.model.joint(engine_name).id
        anchor, axis = engine.data.xanchor[idx], engine.data.xaxis[idx]
        refs = [position for position, text, *_ in labels if text.startswith(f"{joint.name} ref")]
        assert len(refs) == 3  # Position, axis reference, and assembly reading.
        for position in refs:
            np.testing.assert_allclose(position, anchor, atol=1e-12)
        tip = next(position for position, text, *_ in labels
                   if text.startswith(f"{joint.name} axis="))
        np.testing.assert_allclose(np.cross(tip - anchor, axis), 0, atol=1e-12)
        assert np.dot(tip - anchor, axis) > 0
