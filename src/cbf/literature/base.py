from .. import Policy, CtrlAffine
import numpy as np


class Barrier:
    def __init__(self):
        pass

    def __call__(self, x):
        return 0.0


class QuadraticBarrier(Barrier):
    def __init__(self, A: np.ndarray, b: np.ndarray, c: float):
        self.A = A
        self.b = b
        self.c = c

    def __call__(self, x):
        return x @ self.A @ x + self.c + self.b @ x


class MethodBase:
    def __init__(
        self,
        base_policy: Policy,
        req_conf_level: float,
        barrier: Barrier,
        dynamics: CtrlAffine,
        solver: str,
        solver_opts: dict,
    ):
        self.barrier = barrier
        self.req_conf_level = req_conf_level
        self.base_policy = base_policy
        self.dynamics = dynamics
        self.ns = dynamics.ns
        self.nu = dynamics.nu
        self.solver = solver
        self.solver_opts = solver_opts
