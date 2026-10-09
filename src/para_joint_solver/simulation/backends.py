"""Independent engine geometry and constrained reference solves.

The reference FK is given AB and a seed only. Neither core IK/FK nor its analytic
Jacobians are used to compute the reference solution or the differential maps.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ..core.config import vector
from ..core.errors import ConvergenceError, SingularConfiguration
from .models import mjcf
from ..core.solver import check_limits


@dataclass
class ReferenceState:
    q_rp: np.ndarray
    J: np.ndarray
    closure_error: float

    def velocity_to_rp(self, dq_ab):
        return np.linalg.solve(self.J, dq_ab)

    def torque_to_ab(self, tau_rp):
        return np.linalg.solve(self.J.T, tau_rp)


class MujocoBackend:
    def __init__(self, solver, *, visual=False):
        self.mechanism = solver.mechanism
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError("Install MuJoCo with: uv sync --extra sim") from exc
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(mjcf(solver, visual=visual))
        self.data = mujoco.MjData(self.model)
        joints = [
            next(j for j in self.mechanism.output if j.name == n) for n in ("roll", "pitch")
        ] + [b.motor for b in self.mechanism.branches]
        names = ["roll", "pitch", "motor_A", "motor_B"]
        self.scales = np.array([j.scale for j in joints])
        self.qidx = [int(self.model.joint(n).qposadr[0]) for n in names]
        self.vidx = [int(self.model.joint(n).dofadr[0]) for n in names]
        self.site_ids = {
            n: self.model.site(n).id for n in ("B_A", "B_B", "C_A", "C_B", "R_A", "R_B")
        }
        # Engine home frames determine passive-joint reference orientations.
        self.mj.mj_fwdPosition(self.model, self.data)
        self.rod_rotations = []
        for name in ("A", "B"):
            crank = self.model.body(f"motor_{name}").id
            rod = self.model.body(f"rod_{name}").id
            self.rod_rotations.append(
                self.data.xmat[crank].reshape(3, 3).T @ self.data.xmat[rod].reshape(3, 3)
            )

    def set_configuration(self, rp, ab, close_rods=False):
        self.data.qpos[self.qidx] = np.r_[rp, ab] * self.scales
        self.mj.mj_fwdPosition(self.model, self.data)
        if close_rods:
            for i, name in enumerate(("A", "B")):
                b = self.data.site_xpos[self.site_ids[f"B_{name}"]]
                c = self.data.site_xpos[self.site_ids[f"C_{name}"]]
                crank_r = self.data.xmat[self.model.body(f"motor_{name}").id].reshape(3, 3)
                direction = self.rod_rotations[i].T @ crank_r.T @ (c - b)
                direction /= np.linalg.norm(direction)
                x, y, z = direction
                angles = [np.arctan2(-y, z), np.arctan2(x, np.hypot(y, z))]
                for axis, angle in zip(("x", "y"), angles):
                    idx = self.model.joint(f"rod_{name}_{axis}").qposadr[0]
                    self.data.qpos[idx] = angle
            self.mj.mj_fwdPosition(self.model, self.data)

    def points_and_jacobians(self, rp, ab):
        self.set_configuration(rp, ab)
        points, jacobians = {}, {}
        for name in ("B_A", "B_B", "C_A", "C_B"):
            idx = self.site_ids[name]
            points[name] = self.data.site_xpos[idx].copy()
            jac = np.zeros((3, self.model.nv))
            self.mj.mj_jacSite(self.model, self.data, jac, None, idx)
            jacobians[name] = jac[:, self.vidx] * self.scales
        return (
            np.array([points[n] for n in ("B_A", "B_B")]),
            np.array([points[n] for n in ("C_A", "C_B")]),
            np.array([jacobians[n] for n in ("B_A", "B_B")]),
            np.array([jacobians[n] for n in ("C_A", "C_B")]),
        )

    def rod_gaps(self):
        return np.array(
            [
                np.linalg.norm(
                    self.data.site_xpos[self.site_ids[f"R_{n}"]]
                    - self.data.site_xpos[self.site_ids[f"C_{n}"]]
                )
                for n in ("A", "B")
            ]
        )

    def constrained_velocity_map(self, rp, ab):
        """Compute dRP/dAB from MuJoCo's full six-row equality Jacobian.

        Includes the four passive rod coordinates; does not reconstruct scalar rod
        constraints or reuse the analytic solver's transmission Jacobian.
        """
        self.set_configuration(rp, ab, close_rods=True)
        rows = self.data.efc_type == self.mj.mjtConstraint.mjCNSTR_EQUALITY
        constraint = self.data.efc_J.reshape(self.data.nefc, self.model.nv)[rows]
        active = self.vidx[2:]
        passive = [i for i in range(self.model.nv) if i not in active]
        if constraint.shape != (6, 8):
            raise ValueError(f"Expected 6 constraints and 8 velocities, got {constraint.shape}")
        tangent = np.zeros((self.model.nv, 2))
        tangent[active] = np.diag(self.scales[2:])
        tangent[passive] = np.linalg.solve(
            constraint[:, passive], -constraint[:, active] @ tangent[active]
        )
        return tangent[self.vidx[:2]] / self.scales[:2, None]

    def closure(self, rp, ab):
        b, c, db, dc = self.points_and_jacobians(rp, ab)
        lengths = np.array([x.rod_length for x in self.mechanism.branches])
        d = c - b
        f = (np.sum(d**2, axis=1) - lengths**2) / (2 * lengths)
        # Engine Jacobian columns are [roll, pitch, A, B] in public coordinates.
        df = np.einsum("ni,nij->nj", d, dc - db) / lengths[:, None]
        return f, df[:, :2], df[:, 2:]

    def at_configuration(self, rp, ab):
        f, drp, dab = self.closure(rp, ab)
        if np.linalg.cond(dab) > self.mechanism.max_condition:
            raise SingularConfiguration("Reference motor closure Jacobian is singular")
        j = -np.linalg.solve(dab, drp)
        if np.linalg.cond(j) > self.mechanism.max_condition:
            raise SingularConfiguration("Reference transmission Jacobian is singular")
        return ReferenceState(np.array(rp), j, float(np.max(np.abs(f))))

    def solve(self, q_ab, seed=None):
        m = self.mechanism
        ab = vector(q_ab, 2, "q_ab")
        check_limits(ab, m.ab_limits, "q_ab")
        guess = m.home_rp if seed is None else vector(seed, 2, "seed")
        result = least_squares(
            lambda delta: self.closure(guess + delta, ab)[0],
            np.zeros(2),
            jac=lambda delta: self.closure(guess + delta, ab)[1],
            bounds=(m.rp_limits[:, 0] - guess, m.rp_limits[:, 1] - guess),
            method="dogbox",
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
            max_nfev=100,
        )
        state = self.at_configuration(guess + result.x, ab)
        if state.closure_error > m.tolerance:
            raise ConvergenceError(f"Reference FK residual {state.closure_error:g} m")
        return state
