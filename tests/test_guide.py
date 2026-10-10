import json

import pytest
import yaml

from para_joint_solver import available_robots, load_solver
from para_joint_solver.visualization.guide import mechanism_guide


@pytest.mark.parametrize("name", available_robots())
def test_guide_explains_every_configured_number(name):
    mechanism = load_solver(name).mechanism
    guide = mechanism_guide(mechanism)
    json.dumps(guide, allow_nan=False)
    entries = {entry["path"]: entry for group in guide["groups"] for entry in group["entries"]}
    source = yaml.safe_load((mechanism.directory / "mechanism.yaml").read_text())
    assert yaml.safe_load(guide["source_yaml"]) == source
    assert {e["path"]: e["value"] for e in guide["metadata"]} == {
        key: source[key] for key in ("id", "topology", "description", "sources") if key in source
    }

    def check(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                check(child, f"{path}.{key}" if path else key)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            for i, child in enumerate(value):
                check(child, f"{path}[{i}]")
        elif isinstance(value, (int, float)) or (
            isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value)
        ):
            entry = entries[path]
            assert entry["values"] == (value if isinstance(value, list) else [value])
            assert len(entry["components"]) == len(entry["values"])
            assert entry["meaning"] and entry["target"]

    check(source)
    assert entries["assembly.home_rp"]["components"] == ["Roll", "Pitch"]
    assert entries["assembly.motor_reference"]["components"] == ["A", "B"]
    assert guide["output_order"] == [joint.name for joint in mechanism.output]
    for i, joint in enumerate(mechanism.output):
        assert entries[f"output.joints[{i}].name"]["values"] == [joint.name]


def test_guide_keeps_reference_values_with_signed_calibration(calibrated_robot):
    solver = load_solver(calibrated_robot)
    before = mechanism_guide(solver.mechanism)
    solver.at_pose([0.075, -0.08])
    assert mechanism_guide(solver.mechanism) == before
    entries = {e["path"]: e for g in before["groups"] for e in g["entries"]}
    assert entries["branches.A.motor.scale"]["values"] == [-1.2]
    assert entries["assembly.home_rp"]["values"] == [0.025, -0.02]
    assert "-0.03" in entries["branches.B.motor.scale"]["meaning"]
