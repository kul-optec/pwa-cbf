"""
Implementation of Fushimi, et al. 2025 [1] Theorem 11

for the corridor example of Mestres et al. 2025 [2]

[1]: Fushimi, S., Hoshino, K., & Nishimura, Y. (2025). Safety-Critical Control for Discrete-time Stochastic Systems with Flexible Safe Bounds using Affine and Quadratic Control Barrier Functions. arXiv preprint arXiv:2501.09324.
[2]: Mestres, P., Werner, B., Cosner, R.K., & Ames, A.D. (2025). Probabilistic Control Barrier Functions: Safety in Probability for Discrete-Time Stochastic Systems. ArXiv, abs/2510.01501.
"""

from .base import MethodBase, QuadraticBarrier
from .. import Policy, CtrlAffine
import numpy as np
import cvxpy as cp


class Fushimi(MethodBase):
    def __init__(
        self,
        base_policy: Policy,
        req_conf_level: float,
        barrier: QuadraticBarrier,
        dynamics: CtrlAffine,
        Σ: np.ndarray,
        a: float,
        β: float,
        solver: str = cp.CLARABEL,
        solver_opts: dict = {},
    ):
        super().__init__(
            base_policy, req_conf_level, barrier, dynamics, solver, solver_opts
        )
        self.barrier: QuadraticBarrier
        self.Σ = Σ
        self.a = a
        self.β = β
        self.build_fushimi()

    def build_fushimi(self):
        self.u = cp.Variable(self.nu, name="u")
        self.f_cst = cp.Parameter(name="f_c[1]")
        self.f_lin = cp.Parameter(self.nu, name="f_l[1,:]")
        self.log = cp.Parameter(name="log_term")
        self.state = cp.Parameter(self.ns, "x")
        self.u_base = cp.Parameter(self.nu, "u_base")
        A = self.barrier.A
        c = self.barrier.c
        # Fushimi constraint with h = c-y^2 and Σ = diag([0, σ^2, 0])
        F2 = cp.square(self.f_cst + self.f_lin @ self.u)
        inv_Σ = np.eye(len(A))
        inv_Σ[1, 1] = 1 / self.Σ[1, 1]
        temp = (-A @ np.linalg.inv((1 / 2 * inv_Σ) + self.a * A) @ A)[1, 1]
        constraint = self.a * (
            F2 * (A[1, 1] + temp * self.a) + c
        ) >= -self.log - 1 / 2 * cp.log(
            np.linalg.det(np.eye(len(A)) + 2 * self.a * self.Σ @ A)
        )
        objective = cp.Minimize(cp.sum_squares(self.u - self.u_base))
        self.problem = cp.Problem(objective, [constraint])

    def __call__(self, state, k=0):
        self.state.value = state
        self.f_cst.value = self.dynamics.f_c(state)[1]
        self.f_lin.value = self.dynamics.f_l(state)[1, :]
        self.u_base.value = self.base_policy(state, k)
        self.log.value = np.log(np.exp(-self.a * self.barrier(state)) + self.β)
        self.problem.solve(solver=self.solver, **self.solver_opts)
        # print("Optimal u:", u.value)
        if (
            self.problem.status == "optimal"
            or self.problem.status == "optimal_inaccurate"
        ):
            if self.u.value is not None:
                return np.squeeze(self.u.value)
            else:
                raise ValueError(
                    f"{self.u} was not set, even though problem status was 'optimal'. This indicates a potential bug."
                )
        else:
            # print(self.problem.status)
            return


def get_param_fushimi(ε, σ, c, sim_len, a=None):
    'Get feasible parameters Fushimi, where A = diag([0, -1, 0]) Σ=diag([0, σ^2, 0]) so only for corridor example'
    if a is None:
        a_ = np.logspace(0, np.log(1 / 2 / σ * 2), 100)
        a_ = a_[::-1]
    else:
        a_ = [a]
    for a in a_:
        det = 1 - 2 * σ**2 * a
        if det > 0:
            lb = max(0, (1 / np.sqrt(det) - 1) * np.exp(-a * c))
            ub = (ε - np.exp(-a * c)) / sim_len
            if lb <= ub:
                print(
                    f"Running with β = {ub:.4e} giving an upperbound on the failure probability of {np.exp(-a * c) + ub * sim_len:.4e}"
                )
                return (a, ub)
    return
