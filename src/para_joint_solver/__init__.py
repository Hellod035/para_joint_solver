"""Configurable parallel joints with explicit RP/AB coordinate and power conventions."""

from .core.config import available_robots, load_mechanism
from .core.errors import (
    ConfigurationError,
    ConvergenceError,
    LimitViolation,
    SingularConfiguration,
    SolverError,
    UnreachablePose,
)
from .core.solver import ParallelJointSolver, State, load_solver

__all__ = [
    "ParallelJointSolver",
    "State",
    "load_solver",
    "load_mechanism",
    "available_robots",
    "SolverError",
    "ConfigurationError",
    "ConvergenceError",
    "LimitViolation",
    "SingularConfiguration",
    "UnreachablePose",
]
