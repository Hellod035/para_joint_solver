class SolverError(ValueError):
    """An explicitly rejected mechanism configuration or operating point."""


class ConfigurationError(SolverError):
    pass


class UnreachablePose(SolverError):
    pass


class LimitViolation(SolverError):
    def __init__(self, message, *, coordinates=None, values=None, limits=None):
        super().__init__(message)
        self.coordinates = coordinates
        self.values = values
        self.limits = limits


class ConvergenceError(SolverError):
    pass


class SingularConfiguration(SolverError):
    pass
