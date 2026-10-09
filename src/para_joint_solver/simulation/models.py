"""Generate engine models solely from the robot's two configuration files."""

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from ..core.geometry import align_z, fmt, quaternion
from ..visualization.style import STYLE, frame_color
from ..visualization.config import load_visuals, pose


def sub(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def mj_pose(t):
    return {"pos": fmt(t[:3, 3]), "quat": fmt(quaternion(t[:3, :3]))}


def xml_text(root):
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


def mjcf(solver, *, visual=False):
    m = solver.mechanism
    bodies = load_visuals(m)["bodies"] if visual else []
    style = STYLE
    root = ET.Element("mujoco", model=m.id)
    root.append(ET.Comment(" Kinematic inspection model. Inertials are compilation placeholders. "))
    sub(root, "compiler", angle="radian", autolimits="true", inertiafromgeom="false")
    sub(root, "option", gravity="0 0 0", jacobian="dense")
    default = sub(root, "default")
    sub(default, "geom", contype="0", conaffinity="0", rgba="0.6 0.6 0.65 1")
    visual = sub(root, "visual")
    sub(visual, "global", offwidth="1600", offheight="1200")
    sub(visual, "map", znear="0.005", zfar="20")
    assets = sub(root, "asset")
    sub(
        assets,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.8 0.86 0.95",
        rgb2="0.35 0.42 0.55",
        width="512",
        height="3072",
    )
    world = sub(root, "worldbody")
    sub(world, "light", pos="0 -1 2", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    sub(world, "light", pos="1 1 1", dir="-1 -1 -1", diffuse="0.4 0.4 0.4")
    base = sub(world, "body", name="base")
    frames = {"base": base}

    def inertial(body):
        # MuJoCo requires positive inertials for moving bodies even when only
        # querying geometry. These constants never enter the three requested maps.
        sub(body, "inertial", mass="1", pos="0 0 0", diaginertia="1 1 1")

    def moving_body(parent, name, joint, parent_position):
        # MuJoCo uses relative body positions; input geometry stays in the base frame.
        body = sub(parent, "body", name=name, pos=fmt(joint.position - parent_position))
        limits = np.sort(joint.scale * joint.limits)
        attrs = dict(range=fmt(limits)) if np.isfinite(limits).all() else dict(limited="false")
        sub(
            body,
            "joint",
            name=name,
            type="hinge",
            axis=fmt(joint.axis),
            ref=joint.scale * joint.reference,
            **attrs,
        )
        return body

    parent = base
    parent_position = np.zeros(3)
    for j in m.output:
        parent = moving_body(parent, j.name, j, parent_position)
        parent_position = j.position
        inertial(parent)
        frames[j.name] = parent
    platform = sub(parent, "body", name="platform")
    frames["platform"] = platform
    for branch in m.branches:
        name = branch.motor.name
        crank = moving_body(base, f"motor_{name}", branch.motor, np.zeros(3))
        frames[f"motor_{name}"] = crank
        inertial(crank)
        sub(
            crank,
            "site",
            name=f"B_{name}",
            pos=fmt(branch.crank_tip - branch.motor.position),
            size=style["marker_radius"],
            rgba="1 0.45 0.05 1",
        )
        rod_r = align_z(branch.platform_anchor - branch.crank_tip)
        rod = sub(
            crank,
            "body",
            name=f"rod_{name}",
            pos=fmt(branch.crank_tip - branch.motor.position),
            quat=fmt(quaternion(rod_r)),
        )
        frames[f"rod_{name}"] = rod
        inertial(rod)
        # A universal joint removes the irrelevant free axial spin of a spherical rod.
        sub(rod, "joint", name=f"rod_{name}_x", type="hinge", axis="1 0 0", limited="false")
        sub(rod, "joint", name=f"rod_{name}_y", type="hinge", axis="0 1 0", limited="false")
        sub(
            rod,
            "site",
            name=f"R_{name}",
            pos=fmt([0, 0, branch.rod_length]),
            size=style["marker_radius"] * 0.65,
            rgba="1 0.1 0.2 1",
        )
        sub(
            platform,
            "site",
            name=f"C_{name}",
            pos=fmt(branch.platform_anchor - m.output[-1].position),
            size=style["marker_radius"],
            rgba="0.1 1 0.5 0.65",
        )

    for idx, visual_body in enumerate(bodies):
        b = visual_body
        attrs = dict(
            type=b["type"],
            rgba=fmt(frame_color(b["frame"])),
            **mj_pose(pose(b.get("pose", {}), "visual.pose")),
        )
        if b["type"] == "mesh":
            sub(
                assets,
                "mesh",
                name=f"asset_{idx}",
                file=str((m.directory / b["file"]).resolve()),
                scale=fmt(b.get("scale", [1, 1, 1])),
            )
            attrs["mesh"] = f"asset_{idx}"
        else:
            attrs["size"] = fmt(b["size"])
        sub(frames[b["frame"]], "geom", name=f"visual_{idx}", **attrs)
    equality = sub(root, "equality")
    for name in ("A", "B"):
        sub(
            equality,
            "connect",
            name=f"closure_{name}",
            site1=f"R_{name}",
            site2=f"C_{name}",
        )
    return xml_text(root)


def export_model(solver, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = ET.fromstring(mjcf(solver, visual=True))
    for asset in root.findall("asset/mesh"):
        source = Path(asset.attrib["file"])
        destination = directory / "assets" / (asset.attrib["name"] + source.suffix)
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
        asset.set("file", destination.relative_to(directory).as_posix())
    path = directory / "model.xml"
    path.write_text(xml_text(root))
    return path
