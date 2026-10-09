"""Small, convention-explicit geometry helpers. SI units, active rotations."""

import numpy as np
from scipy.spatial.transform import Rotation


def rotation(axis, angle):
    return Rotation.from_rotvec(np.asarray(axis) * angle).as_matrix()


def transform(xyz=(0, 0, 0), quat_wxyz=(1, 0, 0, 0)):
    t = np.eye(4)
    w, x, y, z = quat_wxyz
    t[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    t[:3, 3] = xyz
    return t


def rotate_transform(axis, angle):
    t = np.eye(4)
    t[:3, :3] = rotation(axis, angle)
    return t


def point(t, p):
    return t[:3, :3] @ p + t[:3, 3]


def quaternion(r):
    x, y, z, w = Rotation.from_matrix(r).as_quat()
    return np.array([w, x, y, z])


def align_z(direction):
    z = np.asarray(direction, dtype=float)
    z = z / np.linalg.norm(z)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    y = np.cross(z, helper)
    y /= np.linalg.norm(y)
    return np.column_stack((np.cross(y, z), y, z))


def fmt(values):
    return " ".join(f"{float(x):.16g}" for x in np.asarray(values).ravel())
