"""Automatic camera framing from all configured mesh/primitive bounds."""

import numpy as np


def camera_for(backend):
    camera = backend.mj.MjvCamera()
    backend.mj.mjv_defaultCamera(camera)
    camera.azimuth, camera.elevation = 135.0, -18.0
    lower, upper = [], []
    for i in range(backend.model.ngeom):
        if backend.model.geom_type[i] == backend.mj.mjtGeom.mjGEOM_PLANE:
            continue
        r = backend.data.geom_xmat[i].reshape(3, 3)
        aabb = backend.model.geom_aabb[i]
        center = backend.data.geom_xpos[i] + r @ aabb[:3]
        extent = np.abs(r) @ aabb[3:]
        lower.append(center - extent)
        upper.append(center + extent)
    lo, hi = np.min(lower, axis=0), np.max(upper, axis=0)
    camera.lookat[:] = (lo + hi) / 2
    camera.distance = max(0.15, np.linalg.norm(hi - lo) * 1.7)
    return camera
