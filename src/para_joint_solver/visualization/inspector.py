"""Local browser controls for real-time AB/RP geometry inspection."""

import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter_ns
import webbrowser

import numpy as np

from ..simulation.backends import MujocoBackend
from ..core.errors import (
    ConvergenceError, LimitViolation, SingularConfiguration, SolverError, UnreachablePose,
)
from .config import inspection_range
from .scene import camera_for


def limit_states(values, bounds):
    result = []
    for value, (lower, upper) in zip(values, bounds):
        if not np.isfinite([lower, upper]).all():
            result.append(dict(lower=None, upper=None, margin=None, status="coupled"))
            continue
        margin = float(min(value - lower, upper - value))
        status = (
            "outside"
            if margin < -1e-10
            else "near"
            if margin <= 0.05 * (upper - lower)
            else "inside"
        )
        result.append(dict(lower=float(lower), upper=float(upper), margin=margin, status=status))
    return result


def input_error(exc):
    kind = "input"
    for error_type, label in (
        (LimitViolation, "limit"),
        (UnreachablePose, "unreachable"),
        (SingularConfiguration, "singular"),
        (ConvergenceError, "convergence"),
    ):
        if isinstance(exc, error_type):
            kind = label
            break
    result = dict(error=str(exc), kind=kind)
    if isinstance(exc, LimitViolation) and exc.coordinates:
        result["violation"] = dict(
            coordinates=exc.coordinates,
            values=exc.values,
            limits=limit_states(exc.values, exc.limits),
        )
    return result


class Inspector:
    def __init__(self, solver):
        self.solver = solver
        self.engine = MujocoBackend(solver, visual=True)
        self.camera = camera_for(self.engine)
        self.camera_distance = self.camera.distance
        self.state = solver.at_pose(solver.mechanism.home_rp)
        limits = solver.mechanism.rp_limits
        self.ranges = np.where(np.isfinite(limits), limits, inspection_range(solver.mechanism))
        self.renderer = self.engine.mj.Renderer(self.engine.model, height=600, width=800)

    def feasible_rp_ranges(self, state):
        """Local RP intervals at the other axis's current angle, within display bounds."""
        bounds = self.ranges.copy()
        bounds[:, 0] = np.minimum(bounds[:, 0], state.q_rp)
        bounds[:, 1] = np.maximum(bounds[:, 1], state.q_rp)
        orientation = np.linalg.slogdet(state.J)[0]
        for axis in range(2):
            q = state.q_rp.copy()

            def reachable(value):
                q[axis] = value
                try:
                    return np.linalg.slogdet(self.solver.at_pose(q).J)[0] == orientation
                except SolverError:
                    return False

            for side in range(2):
                origin = good = state.q_rp[axis]
                for value in np.linspace(origin, bounds[axis, side], 33)[1:]:
                    if reachable(value):
                        good = value
                        continue
                    bad = value
                    for _ in range(24):
                        middle = (good + bad) / 2
                        if reachable(middle):
                            good = middle
                        else:
                            bad = middle
                    # Stay inside the boundary despite decimal display rounding and
                    # numerical differences between analytic and engine geometry.
                    bounds[axis, side] = good - np.sign(good - origin) * min(
                        1e-5, abs(good - origin) / 2
                    )
                    break
        bounds[:, 0] = np.maximum(bounds[:, 0], self.ranges[:, 0])
        bounds[:, 1] = np.minimum(bounds[:, 1], self.ranges[:, 1])
        return [row.tolist() if row[0] <= row[1] else None for row in bounds]

    def render(self, state=None, camera=None):
        from PIL import Image

        if camera is not None:
            azimuth = float(camera["azimuth"])
            elevation = float(camera["elevation"])
            zoom = float(camera["zoom"])
            if not np.isfinite([azimuth, elevation, zoom]).all() or zoom <= 0:
                raise ValueError("Camera angles must be finite and zoom must be positive")
            self.camera.azimuth = azimuth % 360
            self.camera.elevation = np.clip(elevation, -85, 85)
            self.camera.distance = self.camera_distance * np.clip(zoom, 0.25, 4)
        state = self.state if state is None else state
        self.engine.set_configuration(state.q_rp, state.q_ab, close_rods=True)
        self.renderer.update_scene(self.engine.data, camera=self.camera)
        buffer = BytesIO()
        Image.fromarray(self.renderer.render()).save(buffer, format="JPEG", quality=88)
        return dict(image="data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode())

    def update(self, request):
        mode = request.get("mode", "rp")
        if mode not in {"ab", "rp"}:
            raise ValueError("mode must be ab or rp")
        q = request.get("q", self.state.q_ab if mode == "ab" else self.state.q_rp)
        dq, tau = request.get("dq_ab", [0.1, -0.1]), request.get("tau_rp", [1.0, 1.0])
        start = perf_counter_ns()
        if request.get("reset"):
            candidate = self.solver.at_pose(self.solver.mechanism.home_rp)
        elif mode == "ab":
            candidate = self.solver.solve(q, seed=self.state)
        else:
            candidate = self.solver.at_pose(q)
        dq_rp, tau_ab = candidate.velocity_to_rp(dq), candidate.torque_to_ab(tau)
        solve_time_ms = (perf_counter_ns() - start) / 1e6
        ref = self.engine.solve(candidate.q_ab, seed=self.state.q_rp)
        image = self.render(candidate)["image"]
        self.state = candidate  # Commit only after successful solve and render.
        return dict(
            robot=self.solver.mechanism.id,
            solve_time_ms=solve_time_ms,
            camera=dict(
                azimuth=self.camera.azimuth,
                elevation=self.camera.elevation,
                zoom=self.camera.distance / self.camera_distance,
            ),
            q_ab=candidate.q_ab.tolist(),
            q_rp=candidate.q_rp.tolist(),
            dq_rp=dq_rp.tolist(),
            tau_ab=tau_ab.tolist(),
            J=candidate.J.tolist(),
            condition=candidate.condition,
            closure_m=candidate.closure_error,
            mujoco_q_rp=ref.q_rp.tolist(),
            pose_error_rad=float(np.max(np.abs(ref.q_rp - candidate.q_rp))),
            velocity_error=float(np.max(np.abs(ref.velocity_to_rp(dq) - dq_rp))),
            torque_error=float(np.max(np.abs(ref.torque_to_ab(tau) - tau_ab))),
            rod_gap_m=float(self.engine.rod_gaps().max()),
            limits=dict(
                ab=limit_states(candidate.q_ab, self.solver.mechanism.ab_limits),
                rp=limit_states(candidate.q_rp, self.solver.mechanism.rp_limits),
            ),
            slider_ranges=dict(ab=self.solver.mechanism.ab_limits.tolist(), rp=self.ranges.tolist()),
            feasible_rp=self.feasible_rp_ranges(candidate),
            image=image,
        )


def serve(solver, port=8765, open_browser=True):
    inspector = Inspector(solver)
    page = Path(__file__).with_name("inspector.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(200, page, "text/html; charset=utf-8")
            else:
                self.send(404, b"Not found", "text/plain")

        def do_POST(self):
            if self.path not in {"/state", "/camera"}:
                return self.send(404, b"Not found", "text/plain")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 8192:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                result = (
                    inspector.render(camera=request)
                    if self.path == "/camera"
                    else inspector.update(request)
                )
                self.send(200, json.dumps(result, allow_nan=False).encode(), "application/json")
            except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
                self.send(
                    422, json.dumps(input_error(exc), allow_nan=False).encode(), "application/json"
                )

        def log_message(self, *args):
            pass

    # Single-thread server keeps all rendering on the context's original thread.
    try:
        with HTTPServer(("127.0.0.1", port), Handler) as server:
            url = f"http://127.0.0.1:{server.server_port}/"
            print(f"Inspect {solver.mechanism.id}: {url}  (Ctrl-C to stop)", flush=True)
            if open_browser:
                webbrowser.open(url)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        inspector.renderer.close()
