"""
Implementation of Cosner, et al. 2025 [1]

for the corridor example of Mestres et al. 2025 [2]

[1]: Cosner, Ryan & Culbertson, Preston & Taylor, Andrew & Ames, Aaron. (2023). Robust Safety under Stochastic Uncertainty with Discrete-Time Control Barrier Functions. 10.15607/RSS.2023.XIX.084. 
[2]: Mestres, P., Werner, B., Cosner, R.K., & Ames, A.D. (2025). Probabilistic Control Barrier Functions: Safety in Probability for Discrete-Time Stochastic Systems. ArXiv, abs/2510.01501.
"""
from .base import MethodBase, QuadraticBarrier
from .. import CtrlAffine, Policy
import numpy as np
import cvxpy as cp


class Cosner(MethodBase):
    def __init__(
        self,
        base_policy: Policy,
        req_conf_level: float,
        barrier: QuadraticBarrier,
        dynamics: CtrlAffine,
        Σ: np.ndarray,
        α: float,
        solver: str = cp.CLARABEL,
        solver_opts: dict = {}
    ):
        super().__init__(
            base_policy,
            req_conf_level,
            barrier,
            dynamics,
            solver,
            solver_opts
        )
        self.barrier: QuadraticBarrier
        self.Σ = Σ
        self.α = α
        self.build_cosner()

    def build_cosner(self):
        self.u = cp.Variable(self.nu, name="u")
        self.state = cp.Parameter(self.ns, "x")
        self.u_base = cp.Parameter(self.nu, "u_base")
        self.h = cp.Parameter(name="h")
        self.f_cst = cp.Parameter(name="f_c[1]")
        self.f_lin = cp.Parameter(self.nu, name="f_l[1,:]")
        #objective
        objective = cp.Minimize(cp.sum_squares(self.u - self.u_base))
        #constraint with h = c-y^2
        F2 = cp.square(self.f_cst + self.f_lin @ self.u)        
        constraint = (self.barrier.c - F2) + cp.trace(
            self.barrier.A @ self.Σ
        ) >= self.α * self.h

        self.problem = cp.Problem(objective, [constraint])

    def __call__(self, state, k=0):
        self.state.value = state
        self.f_cst.value = self.dynamics.f_c(state)[1]
        self.f_lin.value = self.dynamics.f_l(state)[1, :]
        self.u_base.value = self.base_policy(state, k)
        self.h.value = self.barrier(state)
        self.problem.solve(solver=self.solver,**self.solver_opts)
        if self.problem.status == "optimal" or self.problem.status == "optimal_inaccurate":
            if self.u.value is not None:
                return np.squeeze(self.u.value)
            else:
                raise ValueError(
                    f"{self.u} was not set, even though problem status was 'optimal'. This indicates a potential bug."
                )
        else:
            # print(self.problem.status)
            return
