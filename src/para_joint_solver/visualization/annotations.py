"""Display-only axes and linkage annotations, never part of the engine model."""

from pathlib import Path

import numpy as np

from ..core.geometry import align_z, rotation
from .config import load_visuals, pose
from .guide import mechanism_guide


class Annotations:
    def __init__(self, engine):
        self.engine = engine
        self.guide = mechanism_guide(engine.mechanism)
        self.entries = {
            entry["path"]: entry for group in self.guide["groups"] for entry in group["entries"]
        }
        self.meshes = []
        for i, body in enumerate(load_visuals(engine.mechanism)["bodies"]):
            if body["type"] == "mesh":
                self.meshes.append(dict(
                    id=str(i), file=body["file"], frame=body["frame"],
                    label=f"STL{i}", name=Path(body["file"]).name,
                    pose=pose(body.get("pose", {}), "visual.pose"),
                    scale=body.get("scale", [1, 1, 1]),
                ))
        self.options = dict(base_axes=True, stl_frames=[], geometry=False,
                            transparent=False, parameters=[])

    def settings(self, value):
        if not isinstance(value, dict) or set(value) - set(self.options):
            raise ValueError("Invalid annotation settings")
        result = {**self.options, **value}
        for key in ("base_axes", "geometry", "transparent"):
            if type(result[key]) is not bool:
                raise ValueError(f"{key} must be a boolean")
        frames = result["stl_frames"]
        ids = {mesh["id"] for mesh in self.meshes}
        if (not isinstance(frames, list)
                or any(not isinstance(frame, str) or frame not in ids for frame in frames)):
            raise ValueError("Unknown STL frames")
        result["stl_frames"] = list(dict.fromkeys(frames))
        parameters = result["parameters"]
        if (not isinstance(parameters, list)
                or any(not isinstance(p, str) or p not in self.entries for p in parameters)):
            raise ValueError("Unknown mechanism parameters")
        result["parameters"] = list(dict.fromkeys(parameters))
        return result

    def description(self):
        return dict(**self.guide, meshes=[
            {k: v for k, v in mesh.items() if k != "pose"} for mesh in self.meshes
        ])

    def mesh_transform(self, mesh):
        """STL source origin/axes, before MuJoCo's mesh centering/principal-axis transform."""
        body = self.engine.model.body(mesh["frame"]).id
        t = np.eye(4)
        t[:3, :3] = self.engine.data.xmat[body].reshape(3, 3)
        t[:3, 3] = self.engine.data.xpos[body]
        # Positive componentwise mesh scale changes units, not the origin or axis directions.
        return t @ mesh["pose"]

    def draw(self, scene, length, options):
        mj, data, model = self.engine.mj, self.engine.data, self.engine.model
        highlight = (1.0, 0.95, 0.25, 1.0)
        white = (0.95, 0.97, 1.0, 1.0)
        width = length * 0.009
        labels = []
        screen_axes = []

        if options["transparent"]:
            for geom in scene.geoms[:scene.ngeom]:
                if geom.objtype == mj.mjtObj.mjOBJ_GEOM:
                    geom.rgba[3] = 0.22
                    geom.transparent = True

        def geom(kind, position, color, label="", size=None, leader=True):
            if scene.ngeom >= scene.maxgeom:
                raise ValueError("Too many annotations for the render scene")
            g = scene.geoms[scene.ngeom]
            mj.mjv_initGeom(g, kind, np.full(3, length * 0.022) if size is None else size,
                           np.asarray(position), np.eye(3).ravel(), np.array(color, np.float32))
            if label:
                labels.append((np.array(position, copy=True), label, color, leader))
            g.emission = 0.5
            scene.ngeom += 1
            return g

        def arrow(start, end, color, label="", radius=width):
            g = geom(mj.mjtGeom.mjGEOM_ARROW, start, color)
            mj.mjv_connector(g, mj.mjtGeom.mjGEOM_ARROW, radius, start, end)
            if label:
                geom(mj.mjtGeom.mjGEOM_SPHERE, end, color, label, np.full(3, width * 0.25))

        def frame(t, size):
            origin = t[:3, 3]
            colors = ((1, 0.18, 0.16, 1), (0.2, 0.9, 0.3, 1), (0.2, 0.5, 1, 1))
            for i, color in enumerate(colors):
                end = origin + t[:3, i] * size
                arrow(origin, end, color)
                screen_axes.append((origin.copy(), end.copy(), color))
            geom(mj.mjtGeom.mjGEOM_SPHERE, origin, white, size=np.full(3, width))

        if options["base_axes"]:
            frame(np.eye(4), length)
        for mesh in self.meshes:
            if mesh["id"] in options["stl_frames"]:
                frame(self.mesh_transform(mesh), length * 0.75)

        for joint in (*self.engine.mechanism.output,
                      *(b.motor for b in self.engine.mechanism.branches)):
            motor = joint.name in {"A", "B"}
            name = f"motor_{joint.name}" if motor else joint.name
            if not options["geometry"]:
                continue
            idx = model.joint(name).id
            origin, axis = data.xanchor[idx], data.xaxis[idx]
            color = (0.9, 0.55, 1.0, 1.0)
            geom(mj.mjtGeom.mjGEOM_SPHERE, origin, color)
            arrow(origin, origin + axis * length, color, name + ".axis")

        for name in ("A", "B"):
            b = data.site_xpos[self.engine.site_ids[f"B_{name}"]]
            c = data.site_xpos[self.engine.site_ids[f"C_{name}"]]
            if options["geometry"]:
                g = geom(mj.mjtGeom.mjGEOM_CAPSULE, b, white)
                mj.mjv_connector(g, mj.mjtGeom.mjGEOM_CAPSULE, width * 0.65, b, c)
                for position, label in ((b, f"B_{name}"), (c, f"C_{name}")):
                    geom(mj.mjtGeom.mjGEOM_SPHERE, position, white, label)
        for path in options["parameters"]:
            self._parameter(self.entries[path], length, geom, arrow, highlight)
        # Several enabled fields can share reference-point labels.
        unique = {(tuple(label[0]), label[1]): label for label in labels}
        return list(unique.values()), screen_axes

    def _parameter(self, entry, length, geom, arrow, color):
        """Draw configured numbers at the reference geometry, even while the model moves."""
        mj = self.engine.mj
        m = self.engine.mechanism
        target, field = entry["target"], entry["path"].split(".")[-1]

        def number(value):
            return f"{value:.7g}"

        def vector(values):
            return "[" + ", ".join(number(v) for v in values) + "]"

        def label(position, text, rgba=color):
            # Keep anchors on the geometry; text spacing is handled in screen space.
            return geom(mj.mjtGeom.mjGEOM_SPHERE, position, rgba, text,
                        np.full(3, length * 0.002))

        def line(start, end, rgba=color, pixels=1.2):
            if np.linalg.norm(end - start) < 1e-12:
                return
            g = geom(mj.mjtGeom.mjGEOM_LINE, start, rgba)
            mj.mjv_connector(g, mj.mjtGeom.mjGEOM_LINE, pixels, start, end)

        def point_coordinates(position, name):
            geom(mj.mjtGeom.mjGEOM_SPHERE, position, color,
                 f"{name} ref XYZ={vector(position)} m", leader=False)

        if target in {"output", "motors"}:
            joints = m.output if target == "output" else [b.motor for b in m.branches]
            for joint in joints:
                arrow(joint.position, joint.position + joint.axis * length, color)
                label(joint.position,
                      f"{joint.name} ref={number(joint.reference)} rad")
            return

        if target.startswith(("tip:", "anchor:", "rod:")):
            branch = next(b for b in m.branches if b.motor.name == target.split(":")[1])
            b, c, name = branch.crank_tip, branch.platform_anchor, branch.motor.name
            if target.startswith("rod:"):
                line(b, c, pixels=2)
                geom(mj.mjtGeom.mjGEOM_SPHERE, b, color, f"B_{name} ref")
                geom(mj.mjtGeom.mjGEOM_SPHERE, c, color, f"C_{name} ref")
                label((b + c) / 2, f"L={number(branch.rod_length)} m")
            else:
                point_coordinates(b if field == "crank_tip" else c,
                                  f"B_{name}" if field == "crank_tip" else f"C_{name}")
            return

        name = target.split(":")[1]
        joint = (next(j for j in m.output if j.name == name) if target.startswith("joint:")
                 else next(b.motor for b in m.branches if b.motor.name == name))
        origin, axis = joint.position, joint.axis
        if field == "position":
            point_coordinates(origin, name)
        elif field == "name":
            arrow(origin, origin + axis * length, color, name)
        elif field == "axis":
            arrow(origin, origin + axis * length, color,
                  f"{name} axis={vector(axis)}")
            label(origin, f"{name} ref")
        elif field == "scale":
            arrow(origin, origin + axis * length, color)
            label(origin + axis * length, f"{name} scale={number(joint.scale)}")
        elif field == "limits":
            if not np.isfinite(joint.limits).all():
                label(origin, f"{name}: no independent limits")
                return
            # A schematic angular scale, with the reference ray at q=reference.
            # For motors its direction follows the actual reference crank radius.
            radial = align_z(axis)[:, 0]
            if target.startswith("motor:"):
                branch = next(b for b in m.branches if b.motor.name == name)
                radial = branch.crank_tip - origin
                radial = radial - axis * np.dot(axis, radial)
                radial /= np.linalg.norm(radial)
            radius = length * 1.2
            angles = joint.scale * (joint.limits - joint.reference)
            arc = [origin + rotation(axis, angle) @ radial * radius
                   for angle in np.linspace(*angles, 49)]
            for start, end in zip(arc, arc[1:]):
                line(start, end, pixels=1.5)
            for end, bound, title in zip((arc[0], arc[-1]), joint.limits, ("min", "max")):
                line(origin, end)
                label(end, f"{name} {title}={number(bound)} rad")
            end = origin + radial * radius
            line(origin, end, (.75, .82, .9, 1))
            label(end, f"ref={number(joint.reference)} rad")
