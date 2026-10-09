"""Visual asset placement and finite inspection ranges, independent of the solver."""

import numpy as np
import yaml
from ..core.config import keys, vector
from ..core.errors import ConfigurationError
from ..core.geometry import transform


def pose(data, label):
    keys(data, ("xyz", "quat_wxyz"), (), label)
    xyz = vector(data.get("xyz", [0, 0, 0]), 3, label + ".xyz")
    quat = vector(data.get("quat_wxyz", [1, 0, 0, 0]), 4, label + ".quat_wxyz")
    if abs(np.linalg.norm(quat) - 1) > 1e-7:
        raise ConfigurationError(f"{label}: quaternion must be unit length (w, x, y, z)")
    return transform(xyz, quat)


def read_visuals(mechanism):
    path = mechanism.directory / "visuals.yaml"
    if not path.exists():
        return {"bodies": []}
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load visuals at {path}: {exc}") from exc
    keys(data, ("bodies", "inspection"), ("bodies",), "visuals")
    return data


def load_visuals(mechanism):
    data = read_visuals(mechanism)
    validate_visuals(data, mechanism.directory)
    if not data["bodies"]:
        raise ConfigurationError("Rendering requires visuals.yaml with nonempty bodies")
    return data


def inspection_range(mechanism):
    """Plot/sample bounds, never hard joint limits or a claim of reachability."""
    data = read_visuals(mechanism).get("inspection")
    if data is None:
        bounds = mechanism.home_rp[:, None] + [-0.1, 0.1]
        bounds[:, 0] = np.maximum(bounds[:, 0], mechanism.rp_limits[:, 0])
        bounds[:, 1] = np.minimum(bounds[:, 1], mechanism.rp_limits[:, 1])
    else:
        keys(data, ("roll", "pitch"), ("roll", "pitch"), "inspection")
        bounds = np.array([vector(data[n], 2, "inspection." + n) for n in ("roll", "pitch")])
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ConfigurationError("inspection ranges must be increasing")
    if np.any(bounds[:, 0] < mechanism.rp_limits[:, 0]) or np.any(
        bounds[:, 1] > mechanism.rp_limits[:, 1]
    ):
        raise ConfigurationError("inspection ranges must be within configured output limits")
    return bounds


def validate_visuals(v, directory):
    keys(v, ("bodies", "inspection"), ("bodies",), "visuals")
    if not isinstance(v["bodies"], list):
        raise ConfigurationError("visuals.bodies must be a list")
    for b in v["bodies"]:
        keys(
            b, ("frame", "type", "size", "pose", "file", "scale"), ("frame", "type"), "visual body"
        )
        if b["frame"] not in {
            "base",
            "platform",
            "roll",
            "pitch",
            "motor_A",
            "motor_B",
            "rod_A",
            "rod_B",
        }:
            raise ConfigurationError(f"Unknown visual frame: {b['frame']}")
        pose(b.get("pose", {}), "visual.pose")
        if b["type"] == "mesh":
            path = (directory / b.get("file", "")).resolve()
            if not path.is_relative_to((directory / "meshes").resolve()) or not path.is_file():
                raise ConfigurationError("Mesh must exist inside this robot's meshes/ directory")
            scale = vector(b.get("scale", [1, 1, 1]), 3, "mesh.scale")
            if np.any(scale <= 0):
                raise ConfigurationError("mesh.scale must be positive")
        elif b["type"] in {"box", "sphere", "cylinder", "capsule"}:
            n = {"box": 3, "sphere": 1, "cylinder": 2, "capsule": 2}[b["type"]]
            if np.any(vector(b.get("size"), n, "visual.size") <= 0):
                raise ConfigurationError("visual.size must be positive")
        else:
            raise ConfigurationError(f"Unsupported visual type: {b['type']}")
