"""Load reference-pose geometry in one base frame; no visual assets are read here."""

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import yaml

from .errors import ConfigurationError


def vector(value, n, label):
    try:
        a = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label}: expected {n} finite numbers") from exc
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ConfigurationError(f"{label}: expected {n} finite numbers")
    a.setflags(write=False)
    return a


def keys(data, allowed, required, label):
    if not isinstance(data, dict):
        raise ConfigurationError(f"{label}: expected a mapping")
    unknown = set(data) - set(allowed)
    missing = set(required) - set(data)
    if unknown or missing:
        raise ConfigurationError(
            f"{label}: unknown keys {sorted(unknown)}, missing {sorted(missing)}"
        )


def positive(value, label):
    try:
        value = float(value)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"{label}: expected positive finite number") from exc
    if not np.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{label}: expected positive finite number")
    return value


@dataclass(frozen=True)
class Joint:
    name: str
    position: np.ndarray
    axis: np.ndarray
    scale: float
    reference: float
    limits: np.ndarray

    def angle(self, q):
        return self.scale * (q - self.reference)


def joint(data, name, reference, *, output=False):
    keys(
        data,
        ("position", "axis", "scale", "limits") + (("name",) if output else ()),
        ("position", "axis", "name" if output else "limits"),
        name,
    )
    axis = vector(data["axis"], 3, name + ".axis")
    if abs(np.linalg.norm(axis) - 1) > 1e-7:
        raise ConfigurationError(f"{name}: axis must be unit length")
    scale = float(vector([data.get("scale", 1)], 1, name + ".scale")[0])
    if abs(scale) < 1e-12:
        raise ConfigurationError(f"{name}: scale must be nonzero")
    limits = (
        vector(data["limits"], 2, name + ".limits")
        if "limits" in data
        else np.array([-np.inf, np.inf])
    )
    if limits[0] >= limits[1]:
        raise ConfigurationError(f"{name}: limits must be increasing")
    return Joint(
        name, vector(data["position"], 3, name + ".position"), axis, scale, reference, limits
    )


@dataclass(frozen=True)
class Branch:
    motor: Joint
    crank_tip: np.ndarray
    platform_anchor: np.ndarray
    rod_length: float


@dataclass(frozen=True)
class Mechanism:
    id: str
    directory: Path
    output: tuple
    branches: tuple
    home_rp: np.ndarray
    motor_reference: np.ndarray
    tolerance: float
    max_condition: float

    @property
    def rp_limits(self):
        return np.array(
            [next(j.limits for j in self.output if j.name == n) for n in ("roll", "pitch")]
        )

    @property
    def ab_limits(self):
        return np.array([b.motor.limits for b in self.branches])


def robots_directory():
    if os.environ.get("PJS_ROBOTS_DIR"):
        return Path(os.environ["PJS_ROBOTS_DIR"]).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[3] / "robots"
    return checkout if checkout.is_dir() else Path(__file__).parents[1] / "robots"


def available_robots(root=None):
    return sorted(p.parent.name for p in Path(root or robots_directory()).glob("*/mechanism.yaml"))


def load_mechanism(robot, root=None):
    directory = Path(robot).expanduser()
    if directory.is_file():
        mechanism_path = directory.resolve()
        directory = mechanism_path.parent
    else:
        if not directory.is_dir():
            directory = Path(root or robots_directory()) / robot
        mechanism_path = directory / "mechanism.yaml"
    directory = directory.resolve()
    try:
        data = yaml.safe_load(mechanism_path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load robot at {directory}: {exc}") from exc
    keys(
        data,
        (
            "id",
            "topology",
            "description",
            "sources",
            "output",
            "branches",
            "assembly",
        ),
        ("id", "topology", "output", "branches", "assembly"),
        "mechanism",
    )
    if data["topology"] != "dual_crank_two_rod":
        raise ConfigurationError("Supported topology: dual_crank_two_rod")
    keys(
        data["assembly"], ("home_rp", "motor_reference"), ("home_rp", "motor_reference"), "assembly"
    )
    home = vector(data["assembly"]["home_rp"], 2, "home_rp")
    reference = vector(data["assembly"]["motor_reference"], 2, "motor_reference")
    keys(data["output"], ("joints",), ("joints",), "output")
    joints = data["output"]["joints"]
    if (
        not isinstance(joints, list)
        or len(joints) != 2
        or not all(isinstance(j, dict) for j in joints)
        or {j.get("name") for j in joints} != {"roll", "pitch"}
    ):
        raise ConfigurationError(
            "output.joints must contain roll and pitch, in physical chain order"
        )
    output = tuple(
        joint(j, j["name"], home[0 if j["name"] == "roll" else 1], output=True) for j in joints
    )
    keys(data["branches"], ("A", "B"), ("A", "B"), "branches")
    branches = []
    for i, name in enumerate(("A", "B")):
        b = data["branches"][name]
        keys(
            b,
            ("motor", "crank_tip", "platform_anchor", "rod_length"),
            ("motor", "crank_tip", "platform_anchor", "rod_length"),
            name,
        )
        motor = joint(b["motor"], name, reference[i])
        tip = vector(b["crank_tip"], 3, name + ".crank_tip")
        if np.linalg.norm(np.cross(motor.axis, tip - motor.position)) < 1e-9:
            raise ConfigurationError(f"{name}: crank radius must be nonzero")
        anchor = vector(b["platform_anchor"], 3, name + ".platform_anchor")
        length = positive(b["rod_length"], name + ".rod_length")
        if abs(np.linalg.norm(anchor - tip) - length) > 5e-4:
            raise ConfigurationError(
                f"{name}: reference crank_tip/platform_anchor distance must equal rod_length "
                "within 5e-4 m"
            )
        branches.append(Branch(motor, tip, anchor, length))
    limits = np.array([next(j.limits for j in output if j.name == n) for n in ("roll", "pitch")])
    if np.any(home < limits[:, 0]) or np.any(home > limits[:, 1]):
        raise ConfigurationError("home_rp is outside limits")
    ab_limits = np.array([b.motor.limits for b in branches])
    if np.any(reference < ab_limits[:, 0]) or np.any(reference > ab_limits[:, 1]):
        raise ConfigurationError("motor_reference is outside limits")
    return Mechanism(
        str(data["id"]),
        directory,
        output,
        tuple(branches),
        home,
        reference,
        1e-9,
        1e5,
    )
