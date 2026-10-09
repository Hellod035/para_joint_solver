# para-joint-solver

Configurable dual-crank, two-rod parallel joint solver with MuJoCo cross-validation.
It maps AB angles and velocities to RP, and RP torques to AB. All quantities use SI units.

```bash
uv sync --all-extras --locked
uv run pjs list
```

Examples: `g1`, `tiangong`, `igris_c_waist`, `waist_offset`, and `waist_symmetric`.
The symmetric waist uses intersecting roll-then-pitch axes and equal links;
the offset waist uses separated pitch-then-roll axes and asymmetric links.

## Visualization

```bash
uv run pjs inspect g1
```

The local browser viewer provides live AB/RP sliders, joint limits and remaining
travel, mesh rendering, and numerical comparison with MuJoCo. Invalid inputs
retain the last valid pose. Drag the viewport to rotate, scroll to zoom, and
double-click to reset the camera. Use `--port 8766` to change ports; Ctrl-C stops the server.
RP slider bounds stay fixed. Gray track segments show the feasible interval at the
other angle; dragging stops at that interval's boundary. Numeric inputs remain explicit.
The displayed solve time covers core position, Jacobian, velocity and torque computation.

The viewer displays kinematic poses. Each example has six meshes: base platform,
output platform, two cranks, and two rods.

## Benchmark

```bash
uv run pjs benchmark --all --plot
uv run pjs benchmark --robot waist_symmetric --samples 300 --repeats 5
```

Checks cover closure, position, velocity, torque mapping, finite differences,
and MuJoCo closed-chain Jacobians. Timings report core solver median/P95 latency
for IK, FK, Jacobian evaluation, and velocity/torque maps. MuJoCo is used only for
cross-validation and is excluded from solve timings.

Open `artifacts/benchmark/index.html` for the report. `--plot` adds per-pose position,
closure, velocity, torque, Jacobian, and virtual-power error plots. Benchmarking
needs no meshes or OpenGL rendering. Per-robot JSON files and `summary.json` contain
the results. Any failed check returns a nonzero exit code.

New robots need `robots/<name>/mechanism.yaml`, `visuals.yaml`, and `meshes/`.
See the [mechanism configuration guide](docs/mechanism.md) for every field and a working example.
The mechanism defines axis lines and reference attachment points in one base frame;
it alone is sufficient for solving. Mesh alignment and inspection ranges belong in
`visuals.yaml`. RP limits are optional, and inspection ranges are
not reachability boundaries. `external/` contains optional upstream references and
is ignored by Git.
