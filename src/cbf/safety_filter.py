"""Main implementations our safety filter"""

from cbf.solvers.cp_solvers import CBFsolverMIQP
from .simulation import Policy, State
from .solvers import LinearCBFsolverCvxpy
from .dynamics import CtrlAffine
import numpy as np
import geomin as gp
from typing import Sequence, Iterable, Callable, List, Tuple
from itertools import product
from dataclasses import dataclass, field, asdict
import cvxpy as cp

MAX_H = "max_h"
NOT_INF = "not inf"
MAX_H_BASE = "max_h_base"
INDEX_SELECTORS = [MAX_H, NOT_INF, MAX_H_BASE]


# Confidence, State, obstacle_indices, state, time_index -> list[confs]
Weighting = Callable[[float, List[gp.Polyhedron], List, State, int], List[float]]


def default_weighting(
    conf: float,
    obstacles: List[gp.Polyhedron],
    idx_combination: List,
    x: State,
    t: int,
) -> List[float]:
    return [(1 - conf) / len(obstacles) for _ in obstacles]


@dataclass
class SafetyFilterStats:
    nb_problems_solved: list[int] = field(default_factory=list)
    converged: list[bool] = field(default_factory=list)
    u_base: list[np.ndarray] = field(default_factory=list)
    barrier: list[float] = field(default_factory=list)
    predicted_barrier: list[float] = field(default_factory=list)
    failure_prob: float = np.nan
    j_selection: list[list[int]] = field(default_factory=list)
    barriers: list[list[list[float]]] = field(default_factory=list)
    cij: list[list[list[list[float]]]] = field(default_factory=list)
    bij: list[list[list[float]]] = field(default_factory=list)


class SafetyFilterBase:
    """Prototype class to specify the interface"""

    def __init__(
        self,
        base_policy: Policy,
        req_conf_level: float,
        poly: list[gp.Polyhedron],
        dynamics: CtrlAffine,
        quantile_estimator,
        *,
        state_idx_filter: Sequence[int] | None = None,
        alpha: float = 0.0,
        weighting: Weighting | None = None,
        input_bound: float = np.inf,
    ):
        """
        req_conf_level: required probability of staying safe in one step.
        state_idx_filter: tuple of state entries that are to be corrected in the CBF, often, this is 0, 1 (positions only)
        """
        self.base_policy = base_policy
        self.conf_level = req_conf_level
        self.quantile_estimator = quantile_estimator
        self.dynamics = dynamics
        self.poly = poly
        self.stats: SafetyFilterStats = SafetyFilterStats()
        self.weighting = default_weighting if weighting is None else weighting
        self.state_idx_filter = (
            state_idx_filter
            if state_idx_filter is not None
            else tuple(range(dynamics.ns))
        )
        self.input_constraint = not np.isinf(input_bound)
        self.u_max = input_bound
        self.alpha = alpha

    def flush_stats(self) -> dict:
        """Return current stats and reset them"""
        stats_dict = asdict(self.stats)
        self.stats = SafetyFilterStats()
        return stats_dict

    def h(self, x, obstacle: gp.Polyhedron) -> np.ndarray:
        C, b = obstacle.H, obstacle.h
        return C @ x - b

    def barrier_func(self, x: State):
        return np.min([np.max(p.H @ x - p.h) for p in self.poly])

    def _select_index_max_h(self, x: State, obstacle: gp.Polyhedron) -> list[int]:
        return np.argsort(self.h(x, obstacle), axis=0)[::-1].tolist()

    def filter_idx(self, x: State) -> State:
        return np.take(x, self.state_idx_filter)

    def reset_stats(self):
        self.stats = SafetyFilterStats()

    def failure_prob(self, obstacle: gp.Polyhedron, i: int) -> float:
        return (1.0 - self.conf_level) / len(self.poly)


class SafetyFilter(SafetyFilterBase):
    """Implementation of a safety filter using heuristic index selection and quadratic programs"""

    def __init__(
        self,
        base_policy: Policy,
        req_conf_level: float,
        poly: list[gp.Polyhedron],
        dynamics: CtrlAffine,
        quantile_estimator,
        *,
        index_selector: str = MAX_H,
        state_idx_filter: Sequence[int] | None = None,
        solver: LinearCBFsolverCvxpy | None = None,
        internal_solver: str = cp.CLARABEL,
        internal_solver_opts: dict = {},
        alpha: float = 0.0,
        weighting: Weighting | None = None,
        input_bound: float = np.inf,
        max_search_steps: int = 50,
    ):
        """
        req_conf_level: required probability of staying safe in one step.
        state_idx_filter: tuple of state entries that are to be corrected in the CBF, often, this is 0, 1 (positions only)
        """
        super().__init__(
            base_policy,
            req_conf_level,
            poly,
            dynamics,
            quantile_estimator,
            state_idx_filter=state_idx_filter,
            alpha=alpha,
            weighting=weighting,
            input_bound=input_bound,
        )
        self.index_selector = index_selector
        self.max_search_steps = max_search_steps
        self.cbf_solver = (
            solver
            if solver is not None
            else LinearCBFsolverCvxpy(
                self.dynamics.nu,
                len(self.state_idx_filter),
                len(poly),
                alpha,
                self.input_constraint,
                internal_solver=internal_solver,
                internal_solver_opts=internal_solver_opts,
            )
        )
        self.cbf_solver.alpha = alpha

    def _select_index_max_h(self, x: State, obstacle: gp.Polyhedron) -> list[int]:
        return np.argsort(self.h(x, obstacle), axis=0)[::-1].tolist()

    def filter_idx(self, x: State) -> State:
        return np.take(x, self.state_idx_filter)

    def sort_indices(self, x: State, k: int, obstacle: gp.Polyhedron) -> list[int]:
        """Return a list of constraint indices sorted by their maximum violation.

        If MAX_H is used, sorting is based on the violation in the current state.
        If MAX_H_BASE is used, sorting is based on the violation in the state
        forwarded by the base policy.
        If NOT_INF is used, sorting based on MAX_H, but disregarding infinite h"""
        if self.index_selector == MAX_H:
            return self._select_index_max_h(self.filter_idx(x), obstacle)
        if self.index_selector == MAX_H_BASE:
            x_plus = self.dynamics.f(x, self.base_policy(x, k))
            return self._select_index_max_h(self.filter_idx(x_plus), obstacle)
        if self.index_selector == NOT_INF:
            hs = self.h(self.filter_idx(x), obstacle)
            return [i for i in range(len(hs)) if not np.isinf(hs[i])]
        else:
            raise NotImplementedError(
                f"Index selector {self.index_selector} not supported. Pick one of {INDEX_SELECTORS}"
            )

    def __call__(self, x: State, k: int) -> np.ndarray | None:
        x_restr = np.take(x, self.state_idx_filter)
        index_orderings = [self.sort_indices(x, k, p) for p in self.poly]
        # Below, `it` is always a tuple of `poly` and list of `j_indices`.
        # So we sort self.h of the ith poly, and the j_i[0]'th halfplane
        j_first = [j[0] for j in index_orderings]
        hs = [-self.h(x_restr, p)[j] for p, j in zip(self.poly, j_first)]
        sorting = np.argsort(hs)
        inv_sorting = np.argsort(sorting)
        sorted_ix_orderings = [index_orderings[s] for s in sorting]
        u_base = self.base_policy(x, k)
        self.stats.u_base.append(u_base)
        self.stats.barriers.append([self.h(x_restr, p).tolist() for p in self.poly])
        self.stats.cij.append([p.H.tolist() for p in self.poly])
        self.stats.bij.append([p.h.tolist() for p in self.poly])

        n_problems_solved = 0
        for idx_combination_sorted in product(*sorted_ix_orderings):
            if n_problems_solved > self.max_search_steps:
                break
            # Invert the indices so they match with the poly list again
            idx_combination = [idx_combination_sorted[i] for i in inv_sorting]
            # idx_combination = idx_combination_sorted
            ci_list = np.array([p.H[i] for i, p in zip(idx_combination, self.poly)])
            bi_list = np.array([p.h[i] for i, p in zip(idx_combination, self.poly)])
            q_list = []

            failure_probs = self.weighting(
                self.conf_level, self.poly, idx_combination, x_restr, k
            )

            for i, (j, obstacle) in zip(idx_combination, enumerate(self.poly)):
                ci = obstacle.H[i]
                self.stats.failure_prob = failure_probs[j]
                q = self.quantile_estimator(x, ci, failure_probs[j])
                q_list.append(q)

            solution = self.cbf_solver(
                x_restr,
                ci_list,
                bi_list,
                u_base,
                np.array(q_list),
                np.take(self.dynamics.f_c(x), self.state_idx_filter),
                np.take(self.dynamics.f_l(x), self.state_idx_filter, axis=0),
                u_bound=self.u_max,
            )

            n_problems_solved += 1
            if solution is not None:
                self.stats.j_selection.append(idx_combination)
                self.stats.nb_problems_solved.append(n_problems_solved)
                self.stats.converged.append(True)
                self.stats.barrier.append(self.barrier_func(x_restr))
                self.stats.predicted_barrier.append(
                    self.barrier_func(
                        np.take(self.dynamics.f(x, solution), self.state_idx_filter)
                    )
                )
                return solution

        self.stats.nb_problems_solved.append(n_problems_solved)
        self.stats.converged.append(False)
        return


class SafetyFilterMIQP(SafetyFilterBase):
    """Implementation of a safety filter using a mixed-integer quadratic program"""

    def __init__(
        self,
        base_policy: Policy,
        req_conf_level: float,
        poly: list[gp.Polyhedron],
        dynamics: CtrlAffine,
        quantile_estimator,
        *,
        state_idx_filter: Sequence[int] | None = None,
        solver: CBFsolverMIQP | None = None,
        alpha: float = 0.0,
        weighting: Weighting | None = None,
        input_bound: float = np.inf,
        internal_solver: str = "GUROBI",
        internal_solver_opts: dict = {},
    ):
        """
        req_conf_level: required probability of staying safe in one step.
        state_idx_filter: tuple of state entries that are to be corrected in the CBF, often, this is 0, 1 (positions only)
        """
        super().__init__(
            base_policy,
            req_conf_level,
            poly,
            dynamics,
            quantile_estimator,
            state_idx_filter=state_idx_filter,
            alpha=alpha,
            weighting=weighting,
            input_bound=input_bound,
        )
        self.cbf_solver = (
            solver
            if solver is not None
            else CBFsolverMIQP(
                self.dynamics.nu,
                len(poly),
                [p.nb_inequalities for p in poly],
                alpha,
                self.input_constraint,
                internal_solver=internal_solver,
                internal_solver_opts=internal_solver_opts,
            )
        )
        self.cbf_solver.alpha = alpha

    def __call__(self, x: State, k: int) -> np.ndarray | None:
        x_restr = np.take(x, self.state_idx_filter)
        u_base = self.base_policy(x, k)
        self.stats.u_base.append(u_base)

        quantile_arr = []
        delta_over_n = (1 - self.conf_level) / len(self.poly)
        quantile_arr = [
            np.array([self.quantile_estimator(x, -ci, 1 - delta_over_n) for ci in p.H])
            for p in self.poly
        ]

        solution = self.cbf_solver(
            [p.H for p in self.poly],
            [p.h for p in self.poly],
            u_base,
            quantile_arr,
            np.take(self.dynamics.f_c(x), self.state_idx_filter),
            np.take(self.dynamics.f_l(x), self.state_idx_filter, axis=0),
            u_bound=self.u_max,
        )
        if solution is not None:
            self.stats.nb_problems_solved.append(1)
            self.stats.converged.append(True)
            self.stats.barrier.append(self.barrier_func(x_restr))
            self.stats.predicted_barrier.append(
                self.barrier_func(
                    np.take(self.dynamics.f(x, solution), self.state_idx_filter)
                )
            )
        return solution
