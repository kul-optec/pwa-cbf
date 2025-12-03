import cvxpy as cp
import numpy as np
from ..dynamics import State, Input, InputMat


class LinearCBFsolverCvxpy:
    def __init__(
        self,
        n_in: int,
        n_state: int,
        n_obst: int,
        alpha: float = 0.1,
        constr_input=False,
        internal_solver: str = cp.CLARABEL,
        internal_solver_opts: dict = {},
    ):
        self.alpha = alpha
        self.n_in = n_in
        self.n_st = n_state
        self.constr_input = constr_input
        self.n_obst = n_obst
        self.internal_solver = internal_solver
        self.internal_solver_opts = internal_solver_opts
        self.build()

    def build(self):
        self.rhs = cp.Parameter(self.n_obst, name="rhs")
        self.u = cp.Variable(self.n_in, "u")
        # self.ci = cp.Parameter((self.n_obst, self.n_st), "ci")

        # Linear dynamics in u: x+ = f_cst(x) + f_lin(x) @ u
        self.cf_cst = cp.Parameter(self.n_obst, "f_c")
        self.cf_lin = cp.Parameter((self.n_obst, self.n_in), "f_l")

        # self.bi = cp.Parameter(name="bi")
        self.x = cp.Parameter(self.n_st, "x")
        self.u_base = cp.Parameter(self.n_in, "us")
        self.q = cp.Parameter(self.n_obst, name="q")
        # self.alphahx = cp.Parameter(name="αh(x)")

        f = self.cf_cst + self.cf_lin @ self.u
        # TODO: add α form
        # self.alpha * ci @ x + (self.alpha - 1) * bi
        constraints = [-f - self.rhs <= self.q]
        if self.constr_input:
            self.u_bound = cp.Parameter(name="u_bound")
            constraints += [cp.norm_inf(self.u) <= self.u_bound]
        cost = cp.Minimize(cp.sum_squares(self.u - self.u_base))
        self.problem = cp.Problem(cost, constraints)

    def __call__(
        self,
        x: State,
        ci: np.ndarray,
        bi: np.ndarray,
        u_base: Input,
        quantile: np.ndarray,
        f_cst: State,
        f_lin: InputMat,
        u_bound: float = np.inf,
    ) -> None | np.ndarray:
        self.rhs.value = np.atleast_1d((self.alpha - 1) * bi - self.alpha * ci @ x)
        # self.ci.value = np.atleast_2d(ci)
        C = np.atleast_2d(ci)
        self.u_base.value = u_base
        self.x.value = x
        self.q.value = np.atleast_1d(quantile)
        self.cf_cst.value = C @ f_cst
        self.cf_lin.value = C @ f_lin
        if self.constr_input:
            self.u_bound.value = u_bound
        # self.alphahx.value = self.alpha * (ci @ x - bi)
        #
        self.problem.solve(solver=self.internal_solver,**self.internal_solver_opts)
        if self.problem.status == "infeasible":
            return

        return self.u.value


class CBFsolverMIQP:
    def __init__(
        self,
        n_in: int,
        n_obst: int,
        obst_dims: list[int],
        alpha: float = 0.1,
        constr_input=False,
        internal_solver: str = cp.GUROBI,
        internal_solver_opts: dict = {},
    ):
        self.alpha = alpha
        self.n_in = n_in
        self.constr_input = constr_input
        self.dims = obst_dims
        self.n_obst = n_obst
        self.build()
        self.internal_solver = internal_solver
        self.internal_solver_opts = internal_solver_opts

    def build(self):
        self.qtilde = [
            cp.Parameter(nfi, name=f"qt_{i}") for i, nfi in enumerate(self.dims)
        ]
        self.u = cp.Variable(self.n_in, "u")
        self.u_base = cp.Parameter(self.n_in, "us")

        # Linear dynamics in x: x+ = f_cst(x) + f_lin(x) @ u
        # self.cf_cst = [
        #     cp.Parameter(nfi, name=f"f_c_{i}") for i, nfi in enumerate(self.dims)
        # ]
        self.cf_lin = [
            cp.Parameter((nfi, self.n_in), name=f"f_l_{i}")
            for i, nfi in enumerate(self.dims)
        ]

        self.s = [
            cp.Variable(nfi, f"s_{i}", boolean=True) for i, nfi in enumerate(self.dims)
        ]
        self.M = cp.Parameter(pos=True)

        constraints = [
            qi - cBx @ self.u <= self.M * (1 - si)
            for qi, cBx, si in zip(self.qtilde, self.cf_lin, self.s)
        ]

        constraints += [cp.sum(si) >= 1 for si in self.s]

        if self.constr_input:
            self.u_bound = cp.Parameter(name="u_bound")
            constraints += [cp.norm_inf(self.u) <= self.u_bound]
        cost = cp.Minimize(cp.sum_squares(self.u - self.u_base))
        self.problem = cp.Problem(cost, constraints)

    def __call__(
        self,
        C: list[np.ndarray],
        b: list[np.ndarray],
        u_base: Input,
        quantile: list[np.ndarray],  # One quantile for each c_{ij}.
        f_cst: State,
        f_lin: InputMat,
        u_bound: float = np.inf,
    ) -> None | np.ndarray:
        # self.rhs.value = np.atleast_1d((self.alpha - 1) * bi - self.alpha * ci @ x)
        # self.ci.value = np.atleast_2d(ci)
        # C = np.atleast_2d(ci)
        self.u_base.value = u_base
        for qi, ci, bi, qua in zip(self.qtilde, C, b, quantile):
            q = bi + qua
            # print(f"qij: {qua}")
            # print(f"qijtilde: {q}")
            qi.value = q - ci @ f_cst

        for cf_lin, c in zip(self.cf_lin, C):
            cf_lin.value = c @ f_lin

        self.M.value = 1e6
        if self.constr_input:
            self.u_bound.value = u_bound
        self.problem.solve(self.internal_solver, **self.internal_solver_opts)  #
        if self.problem.status == "infeasible":
            return

        return self.u.value
