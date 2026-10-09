"""Closed-chain geometry, analytic differential maps, and assembly-aware FK.

Public coordinates are always RP = [roll, pitch], AB = [A, B].
J means d(AB)/d(RP). Power duality therefore gives tau_AB = J^-T tau_RP.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .config import Mechanism, load_mechanism, vector
from .errors import (
    ConfigurationError,
    ConvergenceError,
    LimitViolation,
    SingularConfiguration,
    UnreachablePose,
)
from .geometry import point, rotate_transform, transform

RP_INDEX = {"roll": 0, "pitch": 1}


def readonly(a):
    a = np.array(a, dtype=float, copy=True)
    a.setflags(write=False)
    return a


def check_limits(q, limits, label):
    if np.any(q < limits[:, 0] - 1e-10) or np.any(q > limits[:, 1] + 1e-10):
        raise LimitViolation(
            f"{label}={q.tolist()} outside {limits.tolist()}",
            coordinates=label,
            values=q.tolist(),
            limits=limits.tolist(),
        )


@dataclass(frozen=True)
class State:
    q_ab: np.ndarray
    q_rp: np.ndarray
    J: np.ndarray
    closure_error: float
    condition: float
    iterations: int = 0

    def velocity_to_rp(self, dq_ab):
        return np.linalg.solve(self.J, vector(dq_ab, 2, "dq_ab"))

    def torque_to_ab(self, tau_rp):
        return np.linalg.solve(self.J.T, vector(tau_rp, 2, "tau_rp"))


class ParallelJointSolver:
    def __init__(self, mechanism: Mechanism):
        self.mechanism = mechanism
        self._branch_signs = []
        for i in range(2):
            roots = self._roots(i, mechanism.home_rp, mechanism.motor_reference[i])
            selected = min(range(2), key=lambda j: abs(roots[j] - mechanism.motor_reference[i]))
            self._branch_signs.append(selected)
            if abs(roots[selected] - mechanism.motor_reference[i]) > 1e-6:
                raise ConfigurationError(
                    "assembly.motor_reference must close at home_rp; "
                    f"branch {i}: expected {roots[selected]:.12g}"
                )
        # Reject a singular home instead of exporting an unusable mechanism.
        self.at_pose(mechanism.home_rp)

    def geometry(self, q_rp, q_ab):
        """B/C points, their derivatives, and body transforms in the mechanism base."""
        m = self.mechanism
        frames = {"base": np.eye(4)}
        t = np.eye(4)
        axes, origins = {}, {}
        for joint in m.output:
            axes[joint.name] = t[:3, :3] @ joint.axis * joint.scale
            origins[joint.name] = point(t, joint.position)
            # Each reference axis line is specified directly in the base frame.
            frame = t @ transform(joint.position) @ rotate_transform(
                joint.axis, joint.angle(q_rp[RP_INDEX[joint.name]])
            )
            frames[joint.name] = frame
            t = frame @ transform(-joint.position)
        frames["platform"] = frames[m.output[-1].name]
        b_points, c_points, db, dc = [], [], [], []
        for i, branch in enumerate(m.branches):
            motor = branch.motor
            mt = transform(motor.position) @ rotate_transform(motor.axis, motor.angle(q_ab[i]))
            frames[f"motor_{motor.name}"] = mt
            b = point(mt, branch.crank_tip - motor.position)
            c = point(t, branch.platform_anchor)
            b_points.append(b)
            c_points.append(c)
            db.append(
                np.cross(motor.axis * motor.scale, b - motor.position)
            )
            dc.append(
                np.column_stack([np.cross(axes[n], c - origins[n]) for n in ("roll", "pitch")])
            )
        return np.array(b_points), np.array(c_points), np.array(db), np.array(dc), frames

    def closure(self, q_rp, q_ab):
        b, c, db, dc, _ = self.geometry(q_rp, q_ab)
        lengths = np.array([branch.rod_length for branch in self.mechanism.branches])
        d = c - b
        f = (np.sum(d * d, axis=1) - lengths**2) / (2 * lengths)
        drp = np.einsum("ni,nij->nj", d, dc) / lengths[:, None]
        dab = -np.sum(d * db, axis=1) / lengths
        return f, drp, dab

    def _roots(self, i, q_rp, reference):
        m = self.mechanism
        branch = m.branches[i]
        _, c, _, _, _ = self.geometry(q_rp, m.motor_reference)
        motor = branch.motor
        local_c = c[i] - motor.position
        tip = branch.crank_tip - motor.position
        parallel = motor.axis * np.dot(motor.axis, tip)
        u = tip - parallel
        v = np.cross(motor.axis, u)
        d = local_c - parallel
        alpha, beta = np.dot(d, u), np.dot(d, v)
        gamma = (np.dot(d, d) + np.dot(u, u) - branch.rod_length**2) / 2
        radius = np.hypot(alpha, beta)
        if radius < 1e-14:
            raise SingularConfiguration(f"Motor {motor.name}: undetermined crank angle")
        ratio = gamma / radius
        if abs(ratio) > 1 + 1e-12:
            raise UnreachablePose(f"Motor {motor.name}: no rod/crank intersection at {q_rp}")
        phi, delta = np.arctan2(beta, alpha), np.arccos(np.clip(ratio, -1, 1))
        roots = np.array([phi + delta, phi - delta]) / motor.scale + motor.reference
        period = 2 * np.pi / abs(motor.scale)
        roots += period * np.round((reference - roots) / period)
        # Prefer an equivalent angle within limits when the nearest unwrap falls outside.
        for j, root in enumerate(roots):
            kmin = int(np.ceil((motor.limits[0] - root - 1e-12) / period))
            kmax = int(np.floor((motor.limits[1] - root + 1e-12) / period))
            if kmin <= kmax:
                roots[j] += period * np.clip(0, kmin, kmax)
        return roots

    def ik(self, q_rp):
        q = vector(q_rp, 2, "q_rp")
        check_limits(q, self.mechanism.rp_limits, "q_rp")
        result = np.array(
            [
                self._roots(i, q, self.mechanism.motor_reference[i])[sign]
                for i, sign in enumerate(self._branch_signs)
            ]
        )
        check_limits(result, self.mechanism.ab_limits, "q_ab")
        return result

    def _state(self, q_rp, q_ab, iterations=0):
        f, drp, dab = self.closure(q_rp, q_ab)
        # Relative singularity check: lengths of crank and rod do not set a universal scale.
        radii = np.array(
            [
                np.linalg.norm(np.cross(b.motor.axis, b.crank_tip - b.motor.position))
                * abs(b.motor.scale)
                for b in self.mechanism.branches
            ]
        )
        if np.any(np.abs(dab) / radii < 1 / self.mechanism.max_condition):
            raise SingularConfiguration("Crank/rod transmission is at a dead centre")
        j = -drp / dab[:, None]
        condition = float(np.linalg.cond(j))
        if not np.isfinite(condition) or condition > self.mechanism.max_condition:
            raise SingularConfiguration(
                f"Output Jacobian is singular/ill-conditioned: {condition:g}"
            )
        residual = float(np.max(np.abs(f)))
        if residual > self.mechanism.tolerance:
            raise ConvergenceError(f"Closure residual {residual:g} m exceeds tolerance")
        return State(readonly(q_ab), readonly(q_rp), readonly(j), residual, condition, iterations)

    def at_pose(self, q_rp):
        """Analytic RP→AB IK, using the assembly branch selected at the configured home."""
        q = vector(q_rp, 2, "q_rp")
        return self._state(q, self.ik(q))

    def solve(self, q_ab, seed=None):
        """AB→RP local numerical FK. Supply the previous State to preserve continuity.

        A closed chain can have multiple output assembly modes for the same motor angles.
        The configured home (or caller's seed) selects a local solution, not a global proof.
        """
        ab = vector(q_ab, 2, "q_ab")
        m = self.mechanism
        check_limits(ab, m.ab_limits, "q_ab")
        guess = m.home_rp if seed is None else seed.q_rp if isinstance(seed, State) else seed
        guess = vector(guess, 2, "seed")
        check_limits(guess, m.rp_limits, "seed")
        # Dogbox can reach an exact joint bound without TRF's bound-scaled
        # gradient stopping the solve before the closure residual is small.
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
        rp = guess + result.x
        if np.max(np.abs(self.closure(rp, ab)[0])) > m.tolerance:
            raise ConvergenceError(f"FK failed after {result.nfev} evaluations: {result.message}")
        try:
            branch_ab = self.ik(rp)
        except (LimitViolation, UnreachablePose) as exc:
            raise ConvergenceError("FK reached an incompatible assembly") from exc
        if np.max(np.abs(branch_ab - ab)) > 1e-6:
            raise ConvergenceError("FK reached the other crank assembly; use a closer seed")
        return self._state(rp, ab, result.nfev)

    def fk(self, q_ab, seed=None):
        return self.solve(q_ab, seed).q_rp


def load_solver(robot, root=None):
    return ParallelJointSolver(load_mechanism(robot, root))
