"""Validate the reversed waist against the original lower-fixed LimX MJCF.

The source revision and license are recorded in robots/limx_waist/NOTICE.
Reference calculations use the source joint tree and connect anchors directly.
"""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.optimize import least_squares

from para_joint_solver import load_solver
from para_joint_solver.simulation.backends import MujocoBackend
from para_joint_solver.visualization.annotations import Annotations

pytestmark = pytest.mark.integration
mj = pytest.importorskip("mujoco")
JOINT_NAMES = ("waist_roll_joint", "waist_pitch_joint", "waist_A_joint", "waist_B_joint")


@pytest.fixture
def official_waist():
    source = ET.parse(Path(__file__).with_name("fixtures") / "limx_hu_d04_01.xml").getroot()
    waist = deepcopy(source.find(".//body[@name='waist_yaw_link']"))
    waist.remove(waist.find("joint"))  # Fix independent yaw; keep the original RP chain.
    for parent in list(waist.iter()):
        for child in list(parent):
            if child.tag == "geom" or (
                child.tag == "body" and not child.get("name", "").startswith("waist_")
            ):
                parent.remove(child)
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", jacobian="dense")
    ET.SubElement(root, "worldbody").append(waist)
    equality = ET.SubElement(root, "equality")
    for constraint in source.findall("equality/connect"):
        if constraint.get("name").startswith("waist_"):
            equality.append(deepcopy(constraint))
    model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert model.neq == 2
    return model, mj.MjData(model)


def official_geometry(model, data, rp, ab):
    for name, value in zip(JOINT_NAMES, np.r_[rp, ab]):
        data.qpos[model.joint(name).qposadr[0]] = value
    mj.mj_fwdPosition(model, data)
    columns = [model.joint(name).dofadr[0] for name in JOINT_NAMES]
    points, derivatives, residuals = [], [], []
    for name in "AB":
        eq = model.equality(f"waist_connect_{name}").id
        lower, rod = model.eq_obj1id[eq], model.eq_obj2id[eq]
        b = data.xpos[rod].copy()
        c = data.xpos[lower] + data.xmat[lower].reshape(3, 3) @ model.eq_data[eq, :3]
        jb, jc = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mj.mj_jac(model, data, jb, None, b, rod)
        mj.mj_jac(model, data, jc, None, c, lower)
        length = np.linalg.norm(model.eq_data[eq, 3:6])
        delta = c - b
        points.append((b, c))
        derivatives.append(delta @ (jc - jb)[:, columns] / length)
        residuals.append((delta @ delta - length**2) / (2 * length))
    return np.array(points), np.array(derivatives), np.array(residuals)


def test_reference_axes_and_limits_match_official_waist(official_waist):
    model, data = official_waist
    mechanism = load_solver("limx_waist").mechanism
    official_geometry(model, data, mechanism.home_rp, mechanism.motor_reference)
    upper = model.body("waist_pitch_link").id
    rotation = data.xmat[upper].reshape(3, 3)
    output = {joint.name: joint for joint in mechanism.output}
    joints = (output["roll"], output["pitch"], *(b.motor for b in mechanism.branches))
    for i, (name, joint) in enumerate(zip(JOINT_NAMES, joints)):
        native = model.joint(name).id
        expected_axis = rotation.T @ data.xaxis[native]
        if i < 2:
            expected_axis = -expected_axis
        expected_position = rotation.T @ (data.xanchor[native] - data.xpos[upper])
        np.testing.assert_allclose(joint.position, expected_position, atol=1e-12)
        np.testing.assert_allclose(joint.axis, expected_axis, atol=1e-12)
        np.testing.assert_allclose(joint.limits, model.jnt_range[native], atol=1e-12)


def test_maps_match_official_moving_motor_geometry(official_waist):
    model, data = official_waist
    solver = load_solver("limx_waist")
    rng = np.random.default_rng(42)
    poses = np.vstack(([0, 0], [[r, p] for r in [-0.16, 0.16] for p in [-0.18, 0.18]],
                       rng.uniform([-0.16, -0.18], [0.16, 0.18], (12, 2))))
    for rp in poses:
        state = solver.at_pose(rp)
        points, derivative, residual = official_geometry(model, data, rp, state.q_ab)
        np.testing.assert_allclose(residual, 0, atol=1e-12)
        upper = model.body("waist_pitch_link").id
        rotation = data.xmat[upper].reshape(3, 3)
        relative = (points - data.xpos[upper]) @ rotation
        b, c, *_ = solver.geometry(rp, state.q_ab)
        np.testing.assert_allclose(relative[:, 0], b, atol=1e-12)
        np.testing.assert_allclose(relative[:, 1], c, atol=1e-12)

        j = -np.linalg.solve(derivative[:, 2:], derivative[:, :2])
        np.testing.assert_allclose(state.J, j, atol=1e-10)
        dq, tau = np.array([0.2, -0.1]), np.array([8, -3])
        np.testing.assert_allclose(state.velocity_to_rp(dq), np.linalg.solve(j, dq), atol=1e-10)
        np.testing.assert_allclose(state.torque_to_ab(tau), np.linalg.solve(j.T, tau), atol=1e-10)

        # Solve each direction using only the source model, seeded at zero.
        def fk_reference(q):
            _, derivative, residual = official_geometry(model, data, q, state.q_ab)
            return residual, derivative[:, :2]

        def ik_reference(q):
            _, derivative, residual = official_geometry(model, data, rp, q)
            return residual, derivative[:, 2:]

        for function, expected in ((fk_reference, rp), (ik_reference, state.q_ab)):
            result = least_squares(lambda q: function(q)[0], [0, 0],
                                   jac=lambda q: function(q)[1],
                                   xtol=1e-14, ftol=1e-14, gtol=1e-15)
            assert result.success
            np.testing.assert_allclose(result.fun, 0, atol=1e-12)
            np.testing.assert_allclose(result.x, expected, atol=1e-9)


def test_stl_frames_match_official_bodies(official_waist):
    model, data = official_waist
    solver = load_solver("limx_waist")
    engine = MujocoBackend(solver, visual=True)
    annotations = Annotations(engine)
    assert len(annotations.meshes) == 6
    for rp in ([0, 0], [0.1, -0.15], [-0.12, 0.14]):
        ab = solver.ik(rp)
        official_geometry(model, data, rp, ab)
        engine.set_configuration(rp, ab, close_rods=True)
        upper = model.body("waist_pitch_link").id
        rotation = data.xmat[upper].reshape(3, 3)
        for mesh in annotations.meshes:
            source = model.body(Path(mesh["file"]).stem).id
            expected = np.eye(4)
            expected[:3, :3] = rotation.T @ data.xmat[source].reshape(3, 3)
            expected[:3, 3] = rotation.T @ (data.xpos[source] - data.xpos[upper])
            actual = annotations.mesh_transform(mesh)
            if mesh["frame"].startswith("rod_") and np.any(rp):
                # Ball-joint spin is free; compare the origin and connection vector.
                np.testing.assert_allclose(actual[:3, 3], expected[:3, 3], atol=1e-12)
                eq = model.equality(f"waist_connect_{mesh['frame'][-1]}").id
                lower = model.eq_obj1id[eq]
                anchor = data.xpos[lower] + data.xmat[lower].reshape(3, 3) @ model.eq_data[eq, :3]
                end = actual[:3, 3] + actual[:3, :3] @ model.eq_data[eq, 3:6]
                np.testing.assert_allclose(end, rotation.T @ (anchor - data.xpos[upper]), atol=1e-12)
            else:
                np.testing.assert_allclose(actual, expected, atol=1e-12)
