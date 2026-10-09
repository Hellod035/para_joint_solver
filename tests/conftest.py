import shutil

import numpy as np
import pytest
import yaml

from para_joint_solver.core.config import robots_directory


@pytest.fixture
def robot_copy(tmp_path):
    destination = tmp_path / "custom"
    shutil.copytree(robots_directory() / "g1", destination)
    return destination


@pytest.fixture
def calibrated_robot(robot_copy):
    """Exercise nonzero homes and signed gearing on a calibrated copy of G1."""
    path = robot_copy / "mechanism.yaml"
    d = yaml.safe_load(path.read_text())
    visual_path = robot_copy / "visuals.yaml"
    v = yaml.safe_load(visual_path.read_text())
    home_rp, home_ab = [0.025, -0.02], [0.04, -0.03]
    for j in d["output"]["joints"]:
        i = 0 if j["name"] == "roll" else 1
        scale = [1.1, -0.9][i]
        j.update(scale=scale, limits=sorted(np.array(j["limits"]) / scale + home_rp[i]))
        v["inspection"][j["name"]] = sorted(
            np.array(v["inspection"][j["name"]]) / scale + home_rp[i]
        )
    for i, name in enumerate(("A", "B")):
        j = d["branches"][name]["motor"]
        scale = [-1.2, 0.8][i]
        j.update(scale=scale, limits=sorted(np.array(j["limits"]) / scale + home_ab[i]))
    d["assembly"] = dict(home_rp=home_rp, motor_reference=home_ab)
    # Cast numpy scalars to native floats for the plain YAML contract.
    for j in d["output"]["joints"]:
        j["limits"] = list(map(float, j["limits"]))
    for b in d["branches"].values():
        b["motor"]["limits"] = list(map(float, b["motor"]["limits"]))
    v["inspection"] = {k: list(map(float, v)) for k, v in v["inspection"].items()}
    path.write_text(yaml.safe_dump(d))
    visual_path.write_text(yaml.safe_dump(v))
    return robot_copy
