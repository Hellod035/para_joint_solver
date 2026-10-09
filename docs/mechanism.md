# Mechanism configuration

`robots/<name>/mechanism.yaml` describes a dual-crank, two-rod mechanism with two
rotational output joints. It contains the complete geometry needed for solving.
All positions and directions are specified in **one mechanism base frame, at the
reference pose**. STL files and mesh coordinate systems do not define this frame.

The solver provides `q_AB → q_RP`, `dq_AB → dq_RP`, and `tau_RP → tau_AB`.
Public vector order is always `RP = [roll, pitch]` and `AB = [A, B]`.
Lengths use metres, angles radians, angular velocities rad/s, and torques N·m.

## Coordinate convention

Choose a right-handed frame fixed to the base platform. A convenient origin is
an output-axis intersection, or a mounting datum when the axes do not intersect.
Use this same frame for **every** `position`, `axis`, `crank_tip`, and
`platform_anchor` in the file.

`assembly.home_rp` and `assembly.motor_reference` specify the public joint readings
at which those coordinates were measured. They need not be zero. At this pose:

- `position` is a point on a joint's rotation axis, expressed in the base frame.
- `axis` is the axis's unit direction in the base frame. Positive physical rotation
  follows the right-hand rule.
- `crank_tip` is the crank-to-rod attachment point in the base frame.
- `platform_anchor` is the rod-to-output-platform attachment point in the base frame.

These are absolute reference coordinates, including for the second output joint.
During motion, the first output joint carries the second joint and its axis;
both joints move the platform anchors. The two motor axes remain fixed to the base.
The two rods constrain the distance between each moving pair of attachment points.

For example, if a motor is at `[0, 0.08, 0.02]` and its reference crank tip is at
`[0.05, 0.08, 0.02]`, write those two points directly. The crank-tip field is the
point `[0.05, 0.08, 0.02]`, with no subtraction of the motor position.

## Complete minimal example

This is the geometry of the supplied symmetric waist, with a new ID. Save it as
`robots/my_robot/mechanism.yaml` to use the commands below.

```yaml
id: my_robot
topology: dual_crank_two_rod
output:
  joints:
  - name: roll
    position: [0, 0, 0.185]
    axis: [1, 0, 0]
  - name: pitch
    position: [0, 0, 0.185]
    axis: [0, 1, 0]
branches:
  A:
    motor:
      position: [0, 0.085, 0.025]
      axis: [0, 1, 0]
      limits: [-1.3, 1.3]
    crank_tip: [0.05, 0.085, 0.025]
    platform_anchor: [0.05, 0.085, 0.185]
    rod_length: 0.16
  B:
    motor:
      position: [0, -0.085, 0.025]
      axis: [0, 1, 0]
      limits: [-1.3, 1.3]
    crank_tip: [0.05, -0.085, 0.025]
    platform_anchor: [0.05, -0.085, 0.185]
    rod_length: 0.16
assembly:
  home_rp: [0, 0]
  motor_reference: [0, 0]
```

Both output axes pass through `[0, 0, 0.185]`. Each crank has a 0.05 m radius;
the distance from its tip to its platform anchor is 0.16 m at the reference pose.
Swapping the order of the output joints changes the mechanism's motion, even
though these reference coordinates remain the same.

## Every field

### Top level

| Field | Required / default | Meaning |
| --- | --- | --- |
| `id` | Required | Robot identifier shown in results. Use the robot directory name. |
| `topology` | Required | `dual_crank_two_rod`, the supported topology. Different geometry within this topology needs only configuration and visual assets; a different linkage topology needs solver changes. |
| `description` | Optional | Human-readable description; unused by the solver. |
| `sources` | Optional | List of geometry reference URLs; unused by the solver. |
| `output` | Required | Defines the two rotational output axes and their physical order. |
| `branches` | Required | Exactly two branch mappings, `A` and `B`. |
| `assembly` | Required | Joint readings at the geometric reference pose. |

### `output.joints`

`output` contains a `joints` list with exactly two entries, named `roll` and
`pitch`. Write them in physical order from the base toward the output platform.
`[pitch, roll]` and `[roll, pitch]` are both supported. Public RP arrays remain
`[roll, pitch]` for either order. The names identify coordinates; the `axis`
vectors define their actual directions.

| Field in each joint | Required / default | Meaning |
| --- | --- | --- |
| `name` | Required | `roll` or `pitch`, each appearing once. |
| `position` | Required | Three coordinates of a point on this output axis in the base frame at the reference pose, in metres. |
| `axis` | Required | Three components of its unit direction in the same reference frame. |
| `scale` | Optional; `1` | Signed multiplier from a public angle change to a physical rotation. Negative values reverse the coordinate direction. Must be finite and nonzero. |
| `limits` | Optional; no independent RP bound | `[lower, upper]` bounds on the public coordinate, in radians. Both values must be finite and increasing when supplied. |

For an output coordinate `q`, the physical rotation from the reference pose is:

```text
rotation = scale * (q - assembly.home_rp[coordinate_index])
```

For example, a roll reference of `0.025`, scale `-1`, and requested roll `0.125`
produce a physical rotation of `-0.1` rad about the configured axis. No separate
zero-offset transform is needed: the points and axis directions already describe
the reference pose.

The output axes may be intersecting, separated, tilted, or expressed with negative
directions. A valid reference assembly must still provide a nonsingular two-DOF map.

### `branches.A` and `branches.B`

Each branch has exactly these four fields:

| Field | Required / default | Meaning |
| --- | --- | --- |
| `motor` | Required | Rotation axis and coordinate settings for this branch's crank. |
| `crank_tip` | Required | Reference crank-to-rod attachment point `[x, y, z]` in the base frame, in metres. |
| `platform_anchor` | Required | Reference rod-to-platform attachment point `[x, y, z]` in the base frame, in metres. It is fixed to the moving output platform. |
| `rod_length` | Required | Positive distance between the rod's attachment centres, in metres. It is independent of the STL's visible dimensions. |

`motor` has the following fields. Its name is determined by the enclosing `A` or
`B` key, so there is no nested motor `name` field.

| Field in `motor` | Required / default | Meaning |
| --- | --- | --- |
| `position` | Required | A point on the fixed motor axis, normally its shaft centre, in base coordinates and metres. |
| `axis` | Required | Unit direction of the motor axis in the base frame. |
| `scale` | Optional; `1` | Signed multiplier from the public AB coordinate change to crank rotation. Must be finite and nonzero. |
| `limits` | Required | Increasing finite `[lower, upper]` bounds for that public A or B angle, in radians. |

The crank's rotation from the reference pose is:

```text
rotation_A = scale_A * (q_A - assembly.motor_reference[0])
rotation_B = scale_B * (q_B - assembly.motor_reference[1])
```

The crank radius is determined by the axis line and reference tip:

```text
crank_radius = norm(cross(motor.axis, crank_tip - motor.position))
```

It must be nonzero. A tip may also have displacement along the motor axis, so its
distance to `motor.position` is not always the crank radius. Specifying a separate
radius would duplicate geometry already fixed by the axis and point.

At the reference pose, each branch must satisfy:

```text
norm(platform_anchor - crank_tip) = rod_length
```

The loader checks this consistency to `1e-9` m. The rod length cannot be changed
independently while leaving an inconsistent pair of reference points. During motion,
the solver enforces the same distance constraint using the moving attachment points.

### `assembly`

| Field | Required / default | Meaning |
| --- | --- | --- |
| `home_rp` | Required | `[roll, pitch]` public angles at which all reference geometry is specified. Also the default FK seed and the inspector's home pose. |
| `motor_reference` | Required | `[A, B]` public motor readings at that same pose. The explicit reference crank tips select the crank assembly branches. |

Both vectors must contain two finite values and lie within any configured limits.
Changing these readings alone relabels the same geometric reference pose. To use
a different physical reference pose, update the output axis lines and attachment
points to their positions at that pose as well.

All geometric numbers must be finite. Axis vectors must have unit length; the
loader checks them rather than silently normalizing them. Unknown fields are rejected.
The schema has no `version`, `units`, mass, inertia, `origin`, `offset`, or `tool` fields.

## Limits and coordinate-dependent torques

AB bounds constrain motor travel. Optional RP bounds add independent restrictions
on the output coordinates. The reachable RP set also depends on the linkage,
assembly branch, singularities, and both AB bounds; two independent RP intervals
do not describe that coupled set.

The inspector keeps slider bounds fixed and shows a gray feasible interval that
changes with the other angle. Dragging stops at that interval's boundary.
`visuals.yaml` inspection ranges select display and benchmark samples; they do not
change the mechanism's joint limits. Choose benchmark ranges inside a reachable
region, leaving room for the small perturbations used by finite-difference checks.

Velocity and torque vectors use the same public coordinates as the angles,
including `scale`. RP torques are generalized torques conjugate to `[roll, pitch]`,
rather than three-dimensional moments expressed in the base frame. The mapping
uses virtual power and requires no mass or inertia parameters.

## Mesh alignment belongs in `visuals.yaml`

The mechanism frame is determined by geometric datums. Exported STL origins,
mesh orientations, and mesh unit conversion are specified through a visual body's
`pose` and `scale`. Changing them does not change joint axes, rod constraints, or
the three requested solver maps.

The six visual parts attach to these frames:

| Visual `frame` | Frame definition at the reference pose |
| --- | --- |
| `base` | The chosen mechanism base frame. |
| `platform` | Origin at the second output joint's `position`; axes parallel to the base axes. |
| `motor_A`, `motor_B` | Origin at the corresponding motor's `position`; axes parallel to the base axes. |
| `rod_A`, `rod_B` | Origin at `crank_tip`; local +Z points toward `platform_anchor`. |

The optional `roll` and `pitch` visual frames likewise start at their own
`position`, with axes parallel to the base at reference. These frames follow
their respective joints during motion. The rod reference Y axis is the normalized
cross product of Z with base X (base Y is used when `abs(Z.x) >= 0.8`), and X is
`Y × Z`. This fixes rod mesh alignment without consulting an STL.

For a mesh body, placement follows:

```text
point_in_attached_frame = R(pose.quat_wxyz) * (scale * mesh_vertex) + pose.xyz
```

`scale` acts componentwise and defaults to `[1, 1, 1]`; `pose.xyz` defaults to zero,
and `pose.quat_wxyz` to the identity `[1, 0, 0, 0]`. The quaternion must be unit length.
For an STL exported in millimetres, use visual scale `[0.001, 0.001, 0.001]`.
For an STL exported in base coordinates at reference, its placement is the inverse
of the attached frame's reference transform. For example, a base-aligned platform
STL attaches using translation `-second_output_joint.position`.

## Add and check a robot

1. Choose a base frame and a valid reference assembly. Record all axis lines,
   attachment points, rod lengths, joint readings, and known limits in that frame.
2. Create `robots/<name>/mechanism.yaml` following the example. Start with unit
   scales unless your public coordinates need a direction or transmission conversion.
3. Check the geometry before adding meshes:

   ```bash
   uv run pjs solve my_robot --q-rp 0 0 --dq-ab 0.1 -0.1 --tau-rp 1 1
   uv run pjs validate --robot my_robot --core-only
   ```

   Replace `0 0` with the configured `home_rp` when it is nonzero. Without
   `visuals.yaml`, validation samples home ±0.1 rad, clipped by optional RP limits.
4. Add the six meshes and `visuals.yaml`. Use `pose` to align each mesh to its
   mechanism frame. Set finite `inspection.roll` and `inspection.pitch` ranges
   whose sampled combinations are reachable.
5. Check the independent MuJoCo geometry and inspect the result:

   ```bash
   uv run pjs benchmark --robot my_robot --plot
   uv run pjs inspect my_robot
   ```

For a complete visual setup, see [the symmetric waist](../robots/waist_symmetric/visuals.yaml).
The [offset waist](../robots/waist_offset/mechanism.yaml) shows separated output
axes, nonzero reference readings, and signed scales. Its
[visual configuration](../robots/waist_offset/visuals.yaml) keeps mesh alignment
separate from the mechanism geometry.
