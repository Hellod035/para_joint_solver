"""Usage: uv run python scripts/inspect_robot.py tiangong [--port 8765]."""

import sys
from para_joint_solver.cli import main

raise SystemExit(main(["inspect", *sys.argv[1:]]))
