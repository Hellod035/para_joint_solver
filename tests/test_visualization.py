import numpy as np
import pytest

from para_joint_solver import SolverError, LimitViolation, available_robots, load_solver

pytestmark = pytest.mark.integration
pytest.importorskip("mujoco")


@pytest.mark.parametrize("mode,reset", [("rp", False), ("ab", False), ("rp", True)])
def test_inspector_timing_excludes_validation_rendering_and_range_search(monkeypatch, mode, reset):
    from para_joint_solver.core.solver import State
    from para_joint_solver.visualization import inspector as module

    inspector = module.Inspector(load_solver("g1"))
    clock = 0

    def add_time(target, name, nanoseconds):
        original = getattr(target, name)

        def measured(*args, **kwargs):
            nonlocal clock
            clock += nanoseconds
            return original(*args, **kwargs)

        monkeypatch.setattr(target, name, measured)

    monkeypatch.setattr(module, "perf_counter_ns", lambda: clock)
    add_time(inspector.solver, "solve" if mode == "ab" else "at_pose", 1_000_000)
    add_time(State, "velocity_to_rp", 2_000_000)
    add_time(State, "torque_to_ab", 3_000_000)
    add_time(inspector.engine, "solve", 100_000_000)
    add_time(inspector, "render", 200_000_000)
    add_time(inspector, "feasible_rp_ranges", 300_000_000)
    try:
        data = inspector.update(dict(mode=mode, reset=reset))
        assert data["solve_time_ms"] == 6.0
        assert clock >= 606_000_000
        assert data["pose_error_rad"] < 1e-8
    finally:
        inspector.renderer.close()


def test_inspector_ab_rp_roundtrip_and_invalid_input():
    from para_joint_solver.visualization.inspector import Inspector

    inspector = Inspector(load_solver("tiangong"))
    try:
        rp = inspector.update(dict(mode="rp", q=[0.08, -0.15]))
        ab = inspector.update(dict(mode="ab", q=rp["q_ab"]))
        np.testing.assert_allclose(ab["q_rp"], rp["q_rp"], atol=1e-9)
        assert ab["pose_error_rad"] < 1e-8 and ab["image"].startswith("data:image/jpeg;base64,")
        before = inspector.state
        with pytest.raises(SolverError):
            inspector.update(dict(mode="ab", q=[100, 100]))
        assert inspector.state is before
        home = inspector.update(dict(reset=True))
        np.testing.assert_allclose(home["q_rp"], [0, 0], atol=1e-12)
    finally:
        inspector.renderer.close()


def test_inspector_limits_are_json_safe_and_failures_are_explained():
    import json
    from para_joint_solver.visualization.inspector import Inspector, input_error

    inspector = Inspector(load_solver("waist_symmetric"))
    try:
        data = inspector.update(dict(mode="rp", q=[0.3, 0]))
        json.dumps(data, allow_nan=False)
        assert data["limits"]["rp"][0]["status"] == "coupled"
        assert data["limits"]["ab"][0]["margin"] > 0
        before = inspector.state
        with pytest.raises(LimitViolation) as caught:
            inspector.update(dict(mode="ab", q=[1.31, 0]))
        error = input_error(caught.value)
        assert error["kind"] == "limit"
        assert error["violation"]["coordinates"] == "q_ab"
        assert error["violation"]["limits"][0]["status"] == "outside"
        assert inspector.state is before
    finally:
        inspector.renderer.close()


def test_camera_moves_without_changing_pose_after_invalid_input():
    from para_joint_solver.visualization.inspector import Inspector

    inspector = Inspector(load_solver("g1"))
    try:
        data = inspector.update(dict(mode="rp", q=[0.04, -0.08]))
        before = inspector.state
        with pytest.raises(LimitViolation):
            inspector.update(dict(mode="ab", q=[100, 100]))
        rotated = inspector.render(camera=dict(azimuth=90, elevation=-30, zoom=0.7))
        assert rotated["image"] != data["image"]
        assert inspector.state is before
        assert inspector.engine.rod_gaps().max() < 1e-9
        assert inspector.camera.distance == pytest.approx(inspector.camera_distance * 0.7)
        with pytest.raises(ValueError):
            inspector.render(camera=dict(azimuth=0, elevation=float("nan"), zoom=1))
        assert inspector.camera.azimuth == 90
        assert inspector.camera.elevation == -30
        restored = inspector.render(camera=dict(azimuth=135, elevation=-18, zoom=1))
        assert restored["image"] == data["image"]
    finally:
        inspector.renderer.close()


@pytest.mark.parametrize("name", available_robots())
def test_rp_slider_endpoints_remain_reachable_when_axes_are_combined(name):
    from para_joint_solver.visualization.inspector import Inspector

    inspector = Inspector(load_solver(name))
    try:
        data = inspector.update(dict(reset=True))
        fixed_ranges = data["slider_ranges"]
        # Move each axis to both ends, including combinations that invalidate the
        # original independent RP limits (notably IGRIS motor travel).
        for axis, side in ((0, 0), (1, 0), (0, 1), (1, 1), (0, 0), (1, 0)):
            q = data["q_rp"].copy()
            q[axis] = data["feasible_rp"][axis][side]
            data = inspector.update(dict(mode="rp", q=q))
            assert data["slider_ranges"] == fixed_ranges
            np.testing.assert_allclose(data["q_rp"], q, atol=1e-10)
            assert data["pose_error_rad"] < 1e-8
            assert data["rod_gap_m"] < 1e-9
            assert all(limit["status"] != "outside" for limit in data["limits"]["ab"])
    finally:
        inspector.renderer.close()


def test_annotations_change_only_rendering_and_preserve_the_last_valid_state():
    from para_joint_solver.visualization.inspector import Inspector

    inspector = Inspector(load_solver("igris_c_waist"))
    try:
        assert inspector.annotations.options["parameters"] == []
        data = inspector.update(dict(mode="rp", q=[0.04, -0.07]))
        before = inspector.state
        rgba = inspector.engine.model.geom_rgba.copy()
        baseline = inspector.render(overlays=dict(base_axes=False, stl_frames=[],
                                                 geometry=False, transparent=False))
        counts = [inspector.renderer.scene.ngeom]
        for settings in (
            dict(base_axes=True),
            dict(stl_frames=[mesh["id"] for mesh in inspector.annotations.meshes]),
            dict(geometry=True, transparent=True, parameters=["branches.A.platform_anchor"]),
        ):
            rendered = inspector.render(overlays=settings)
            assert rendered["image"] != baseline["image"]
            assert inspector.state is before
            assert inspector.engine.rod_gaps().max() < 1e-9
            counts.append(inspector.renderer.scene.ngeom)
        assert all(count > counts[0] for count in counts[1:])
        np.testing.assert_array_equal(inspector.engine.model.geom_rgba, rgba)
        options = inspector.annotations.options.copy()
        for invalid in ({"stl_frames": ["missing"]}, {"stl_frames": "all"},
                        {"base_axes": "yes"},
                        {"parameters": ["bad.field"]}, {"parameters": "bad.field"},
                        {"unknown": True}):
            with pytest.raises(ValueError):
                inspector.render(overlays=invalid)
            assert inspector.annotations.options == options
        with pytest.raises(LimitViolation):
            inspector.update(dict(mode="ab", q=[100, 100]))
        inspector.render(overlays=dict(stl_frames=["0"], parameters=["output.joints[0].axis"]))
        assert inspector.state is before
        np.testing.assert_array_equal(before.q_rp, data["q_rp"])
    finally:
        inspector.renderer.close()


def test_parameter_values_stay_on_reference_geometry_while_stl_frames_move():
    from para_joint_solver.visualization.inspector import Inspector

    inspector = Inspector(load_solver("waist_offset"))
    try:
        annotations = inspector.annotations
        options = dict(base_axes=False, stl_frames=[], geometry=False, transparent=False)
        snapshots = []
        for q in (inspector.solver.mechanism.home_rp, [0.07, -0.08]):
            state = inspector.solver.at_pose(q)
            inspector.engine.set_configuration(state.q_rp, state.q_ab, close_rods=True)
            snapshot = {}
            for path, entry in annotations.entries.items():
                inspector.renderer.update_scene(inspector.engine.data, camera=inspector.camera)
                labels, _ = annotations.draw(inspector.renderer.scene, 0.04,
                                            annotations.settings(dict(**options, parameters=[path])))
                assert labels
                text = " ".join(label[1] for label in labels)
                for value in entry["values"]:
                    expected = f"{value:.7g}" if isinstance(value, (int, float)) else value
                    assert expected in text, (path, value, text)
                snapshot[path] = labels
            snapshots.append(snapshot)
        for path in snapshots[0]:
            first, second = snapshots[0][path], snapshots[1][path]
            assert [label[1] for label in first] == [label[1] for label in second]
            np.testing.assert_allclose([label[0] for label in first],
                                       [label[0] for label in second], atol=1e-12)
        # The assembly datum is (0,0,0), including when the model's home is nonzero.
        inspector.renderer.update_scene(inspector.engine.data, camera=inspector.camera)
        _, axes = annotations.draw(inspector.renderer.scene, 0.04,
                                   annotations.settings({**options, "base_axes": True}))
        assert len(axes) == 3
        for i, (origin, end, _) in enumerate(axes):
            np.testing.assert_array_equal(origin, [0, 0, 0])
            assert end[i] > 0
            np.testing.assert_array_equal(np.delete(end, i), [0, 0])
    finally:
        inspector.renderer.close()


def test_parameter_buttons_can_combine_and_clear_annotations():
    from para_joint_solver.visualization.inspector import Inspector

    inspector = Inspector(load_solver("igris_c_waist"))
    try:
        baseline = inspector.render()["image"]
        before = inspector.state
        a, b = "branches.A.crank_tip", "branches.B.rod_length"
        one = inspector.render(overlays=dict(parameters=[a]))["image"]
        both = inspector.render(overlays=dict(parameters=[a, b]))["image"]
        remaining = inspector.render(overlays=dict(parameters=[b]))["image"]
        assert len({baseline, one, both, remaining}) == 4
        assert inspector.render(overlays=dict(parameters=[]))["image"] == baseline
        assert inspector.state is before
    finally:
        inspector.renderer.close()


def test_annotation_projection_matches_rendered_points():
    from para_joint_solver.visualization.inspector import Inspector
    from para_joint_solver.visualization.labels import project

    inspector = Inspector(load_solver("igris_c_waist"))
    try:
        mj = inspector.engine.mj
        for azimuth in (135, 45):
            inspector.camera.azimuth = azimuth
            inspector.renderer.update_scene(inspector.engine.data, camera=inspector.camera)
            scene = inspector.renderer.scene
            camera = mj.mjv_averageCamera(*scene.camera)
            scene.ngeom = 0
            background = inspector.renderer.render().astype(float)
            for point in ([0, 0, 0], [0.05, 0.05, 0.05]):
                mj.mjv_initGeom(scene.geoms[0], mj.mjtGeom.mjGEOM_SPHERE, np.full(3, .0015),
                                np.array(point), np.eye(3).ravel(),
                                np.array([1, 0, 0, 1], np.float32))
                scene.ngeom = 1
                pixels = inspector.renderer.render().astype(float)
                y, x = np.nonzero(np.max(np.abs(pixels - background), axis=2) > 30)
                assert len(x)
                expected = project(point, camera, 800, 600)
                np.testing.assert_allclose([x.mean() + .5, y.mean() + .5], expected, atol=1)
    finally:
        inspector.renderer.close()
