"""Implements the Scenario and Conformal methods from
Mestres, P., Werner, B., Cosner, R.K., & Ames, A.D. (2025). Probabilistic Control Barrier Functions: Safety in Probability for Discrete-Time Stochastic Systems. ArXiv, abs/2510.01501.
Propositions 8 and 9.

The implementations are specialized to the corridor experiment.
"""

from .. import Policy, CtrlAffine, State
from typing import Callable
import numpy as np
import cvxpy as cp


Sampler = Callable[[int], np.ndarray]


class MethodBaseDataDriven:
    def __init__(
        self,
        base_policy: Policy,
        req_conf_level: float,
        dynamics: CtrlAffine,
        sampler: Sampler,
        n_sample: int,
    ):
        self.base_policy = base_policy
        self.req_conf_level = req_conf_level
        self.dynamics = dynamics
        self.sampler = sampler
        self.n_sample = n_sample
        self.samples = sampler(n_sample)
        self.ns = dynamics.ns
        self.nu = dynamics.nu

    def barrier(self, x):
        if len(x.shape) == 1:
            return 0.25 - x[1] ** 2
        return 0.25 - x[:, 1] ** 2


class Conformal(MethodBaseDataDriven):
    def __init__(
        self,
        base_policy: Policy,
        req_conf_level,
        dynamics,
        sampler,
        n_sample,
        α: float,
        γ: float,
    ):
        super().__init__(
            base_policy,
            req_conf_level,
            dynamics,
            sampler,
            n_sample=n_sample,
        )
        self.α = α
        self.γ = γ
        self.solve_time = []

    def dh(self, x, u, d):
        F = self.dynamics.f(x, u) + d
        return self.barrier(F) - self.α * self.barrier(x)

    def __call__(self, x: State, k: int):
        import gurobipy

        M = 1e0
        m = -1e1
        xi = 1e-6

        model = gurobipy.Model()

        # Create variables
        u = model.addMVar(
            shape=self.nu,
            vtype=gurobipy.GRB.CONTINUOUS,
            lb=-gurobipy.GRB.INFINITY,
            name="u",
        )

        z = model.addMVar(shape=self.n_sample, vtype="B", name="z")

        ubase = self.base_policy(x, k)
        # Objective
        model.setObjective((u - ubase) @ (u - ubase), gurobipy.GRB.MINIMIZE)
        # MI constraints
        model.addConstrs(
            (
                -self.dh(x, u, self.samples[i, :]) <= M * (1 - z[i])
                for i in range(self.n_sample)
            )
        )

        model.addConstrs(
            (
                -self.dh(x, u, self.samples[i, :]) >= xi + (m - xi) * z[i]
                for i in range(self.n_sample)
            )
        )

        model.addConstr(
            z.sum()
            >= (self.req_conf_level + np.sqrt(np.log(1 / self.γ) / (2 * self.n_sample)))
            * (self.n_sample + 1)
        )

        model.Params.OutputFlag = 0
        model.optimize()
        self.solve_time.append(model.Runtime)

        if model.Status == gurobipy.GRB.INFEASIBLE:
            return
        #return real solver time

        return u.X, model.Runtime


class Scenario(MethodBaseDataDriven):
    def __init__(
        self,
        base_policy: Policy,
        req_conf_level,
        dynamics: CtrlAffine,
        sampler,
        n_sample,
        α: float,
        γ: float,
        solver: str = cp.CLARABEL,
        solver_opts: dict = {},
        formulation: str = "Stable",
    ):
        super().__init__(
            base_policy,
            req_conf_level,
            dynamics,
            sampler,
            n_sample=n_sample,
        )
        self.ns, self.nu = self.dynamics.ns, self.dynamics.ns
        self.α = α
        self.γ = γ
        self.solver = solver
        self.solver_opts = solver_opts
        self.formulation = formulation  # "Stable", "Fast" or "Specialized"
        self.solve_time = []
        self.build_scenario()

    def build_scenario(self):
        self.u = cp.Variable(self.nu, "u")
        self.state = cp.Parameter(self.ns, "x")
        self.u_base = cp.Parameter(self.nu, "u_base")
        self.f_cst = cp.Parameter(name="f_c[1]")
        self.f_lin_1 = cp.Parameter(self.ns, name="f_l[1,:]")
        self.f_cst2 = cp.Parameter(name="f_c[1]^2", nonneg=True)
        self.f_lin_cst = cp.Parameter(self.ns, name="f_c[1]*f_l[1,:]")
        self.h = cp.Parameter(name="h")

        # Objective
        objective = cp.Minimize(cp.sum_squares(self.u - self.u_base))
        # Constraints c - (f_c + f_l@u + d)[1]^2 >= αh
        constraints = []
        if self.formulation == "Stable":
            Fy2 = cp.square(self.f_cst + self.f_lin_1 @ self.u + self.samples[:, 1])
        elif self.formulation == "Fast":
            # f_c^2 + 2*f_c*d + d^2 + (f_l@u)^2 + 2*f_c*(f_l@u) + 2*d*(f_l@u)
            Fy2 = (
                self.f_cst2
                + 2 * self.f_cst * self.samples[:, 1]
                + self.samples[:, 1] ** 2
                + cp.square(self.f_lin_1 @ self.u)
                + 2 * self.f_lin_cst @ self.u
                + 2 * self.samples[:, 1] * (self.f_lin_1 @ self.u)
            )
        elif self.formulation == "Specialized":
            # if theta_k = 0 forall k, then f_l@u = f_l[1,1]*u[1] (more stable then Fast)
            self.f_lin2 = cp.Parameter(name="f_l[1,1]^2", nonneg=True)
            Fy2 = (
                self.f_cst2
                + 2 * self.f_cst * self.samples[:, 1]
                + self.samples[:, 1] ** 2
                + self.f_lin2 * cp.square(self.u[1])
                + 2 * self.f_lin_cst[1] * self.u[1]
                + 2 * self.samples[:, 1] * self.f_lin_1[1] * self.u[1]
            )
        else:
            raise ValueError("Unknown method for Scenario Approach")

        hplus = 0.25 - Fy2
        constraints.append(hplus >= self.α * self.h)
        self.problem = cp.Problem(objective, constraints)

    def dh(self, x, u, d):
        F = self.dynamics.f(x, u) + d
        return self.barrier(F) - self.α * self.barrier(x)

    def __call__(self, x: State, k: int):
        self.state.value = x
        self.u_base.value = self.base_policy(x, k)

        self.f_cst.value = self.dynamics.f_c(x)[1]
        self.f_lin_1.value = self.dynamics.f_l(x)[1, :]
        if self.formulation == "Specialized" or self.formulation == "Fast":
            self.f_cst2.value = (self.dynamics.f_c(x)[1]) ** 2
            self.f_lin_cst.value = (
                (self.dynamics.f_c(x)[1]) * self.dynamics.f_l(x)[1, :]
            )
        if self.formulation == "Specialized":
            self.f_lin2.value = (self.dynamics.f_l(x)[1, 1]) ** 2

        self.h.value = self.barrier(x)
        self.problem.solve(solver=self.solver, **self.solver_opts)
        if self.problem == "infeasible":
            return
        return self.u.value
