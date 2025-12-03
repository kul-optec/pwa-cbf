"""
Module for the path-planning experiment.

Dynamics:
    x_{k+1} = x_k + Ts * R_z(θ_k) * u_k + d_k
where
    x_k = [x_k, y_k, θ_k]^T,
    u_k = [v_{x,k}, v_{y,k}, ω_k]^T,
    R_z(θ_k) is the rotation matrix around the z-axis by angle θ_k,
    d_k is a random disturbance.

Constraints:
    Obstacles represented as polyhedra.

This module provides:
    - Tools for simulations with known disturbance distributions
    - Path-following base controller
    -
"""

import cvxpy as cp
import cbf
import numpy as np
from typing import Callable
import numpy.random as npr
import geomin as gp
from dataclasses import dataclass, asdict, field
from cbf.safety_filter import MAX_H
from pathlib import Path
import os


@dataclass
class PathPlanningSettings:
    """
    Settings for the path-planning experiment.

    scene is the name of the scene to load, for example "stresstest.svg"
    (see the scenes folder for examples).

    formulation can be:
        - "ours": heuristic approach,
        - "miqp": mixed-integer approach.

    distribution_scale_x is the assumed standard deviation of the distribution along x.
    distribution_scale_y is the assumed standard deviation of the distribution along y.
    simulated_noise_scale is a scaling factor on the noise std. that is used for the simulated noise
    simulate_noise: if True, noise is added in the simulation
    total_conf is the overall confidence level 1 − ε.

    max_search_steps is the maximum number of QPs solved before returning
    infeasible (only used for the "ours" formulation).

    sorting_method controls how candidate hyperplanes are ordered,
    MAX_H_BASE or MAX_H (see cbf.safety_filter for details).
    """

    scene: str
    distribution: str = "gaussian"
    formulation: str = "ours"
    internal_QPsolver: str = cp.QPALM
    internal_QPsolver_opts: dict = field(default_factory=dict)
    internal_MIQPsolver: str = cp.GUROBI
    internal_MIQPsolver_opts: dict = field(default_factory=dict)
    distribution_scale_x: float = 0.01
    distribution_scale_y: float = 0.01
    simulated_noise_scale_factor: float = 1.0
    n_reps: int = 1
    simulation_horizon: int = 150
    max_input_size: float = np.inf
    total_conf: float = 0.9
    max_search_steps: int = 50
    sorting_method: str = MAX_H
    ignore_safety_filter: bool = False

    def __str__(self) -> str:
        return f"{self.scene}_{self.distribution}({self.distribution_scale:2.3f})-{self.formulation}-{self.internal_QPsolver}-{self.sorting_method}-umax={self.max_input_size}-conf={self.total_conf}{'-openloop' if self.ignore_safety_filter else ''}"

    @property
    def Σ(self) -> np.ndarray:
        return (
            np.diag(
                (
                    self.distribution_scale_x,
                    self.distribution_scale_y,
                    0,
                )
            )
            ** 2
        )

    @property
    def simulate_noise(self) -> bool:
        return self.simulated_noise_scale_factor > 0


def get_perturbed_dynamics(dynamics: cbf.Dynamics):
    def noisy_dyn(x: cbf.State, u: cbf.Input, w: cbf.Noise):
        return dynamics.f(x, u) + dynamics.g(x) @ w

    return noisy_dyn


def get_closed_loop_dynamics(dynamics: cbf.Dynamics, policy: cbf.Policy):
    def noisy_dyn(x: cbf.State, u: cbf.Input, w: cbf.Noise):
        u = policy(x, 0)
        return dynamics.f(x, u) + dynamics.g(x) @ w

    return noisy_dyn


def get_empty_sampler(ns: int):
    def sample():
        return np.zeros(ns)

    return sample


def get_loc_scale_sampler(
    loc: np.ndarray, scale: np.ndarray, rng: npr.Generator, distribution: str
):
    if distribution == "gaussian":
        print("Adding gaussian sampler")
        return get_gaussian_sampler(loc, scale, rng)
    else:
        raise ValueError(
            f"Distribution `{distribution}` not yet implemented for the path planning example."
        )


def get_gaussian_sampler(μ: np.ndarray, Σ: np.ndarray, rng: npr.Generator):
    from scipy.stats import multivariate_normal

    ξ_dist = multivariate_normal(μ, Σ, allow_singular=True)

    def sample():
        return np.array(ξ_dist.rvs(1, random_state=rng))

    return sample


def get_safety_filter(
    config: PathPlanningSettings,
    base_policy: cbf.Policy,
    one_step_conf: float,
    all_poly: list[gp.Polyhedron],
    dynamics: cbf.CtrlAffine,
    quantile_estimator: cbf.QuantileEstimator,
):
    if config.formulation == "ours":
        return cbf.SafetyFilter(
            base_policy,
            one_step_conf,
            all_poly,
            dynamics,
            quantile_estimator,
            state_idx_filter=(0, 1),
            index_selector=config.sorting_method,
            input_bound=config.max_input_size,
            max_search_steps=config.max_search_steps,
            internal_solver=config.internal_QPsolver,
            internal_solver_opts=config.internal_QPsolver_opts,
        )
    elif config.formulation == "miqp":
        return cbf.SafetyFilterMIQP(
            base_policy,
            one_step_conf,
            all_poly,
            dynamics,
            quantile_estimator,
            state_idx_filter=(0, 1),
            input_bound=config.max_input_size,
            internal_solver=config.internal_MIQPsolver,
            internal_solver_opts=config.internal_MIQPsolver_opts,
        )
    raise ValueError(
        f"Safety filter formulation name {config.formulation} was not recognized."
    )


class QuadrupedPathFollow:
    def __init__(
        self, ref_path: np.ndarray, gains: np.ndarray | None = None, subdivide: int = 2
    ):
        self.orig_ref_path = ref_path.copy()
        self.ref_path = self.subsample(ref_path, subdivide)
        self.ref_pt_idx = 0
        self.ctrl_gains = np.diag([2, 2, 2.0]) if gains is None else gains
        self._tol = 0.6

    def subsample(self, path, factor: int):
        t = np.linspace(0, 1, factor + 2)[:-1]  # exclude last to avoid duplicates
        return np.vstack(
            [
                (1 - ti) * path[i] + ti * path[i + 1]
                for i in range(len(path) - 1)
                for ti in t
            ]
            + [path[-1]]
        )

    @property
    def ref_pt(self) -> np.ndarray:
        return self.ref_path[self.ref_pt_idx]

    def R(self, x) -> np.ndarray:
        return np.array([[np.cos(x[2]), -np.sin(x[2])], [np.sin(x[2]), np.cos(x[2])]])

    def __call__(self, x: np.ndarray, _k: int):
        closest = np.argmin([np.linalg.norm(x[:2] - p) for p in self.ref_path])
        self.ref_pt_idx = np.minimum(closest + 1, len(self.ref_path) - 1)

        pos = x[:2]
        pos_err = self.ref_pt - pos
        rel_pos_err = self.R(x).T @ pos_err
        ref_angle = np.arctan2(pos_err[1], pos_err[0])
        head_err = ref_angle - x[2]

        return self.ctrl_gains @ np.concat([rel_pos_err, [head_err]])


def get_gaussian_quantile_estimator(
    g: Callable[[cbf.State], np.ndarray], μ: np.ndarray, Σ: np.ndarray
):
    def estimate_quantile(x: cbf.State, ci: np.ndarray, level: float):
        trans = np.atleast_1d(ci @ g(x))
        new_μ, new_σ = cbf.transform_gaussian(μ, Σ, trans)
        q_est = cbf.NormalQuantile(new_μ, new_σ)
        return q_est(level)

    return estimate_quantile


def get_zero_quantile_estimator():
    def estimate_quantile(x: cbf.State, ci: np.ndarray, level: float):
        return 0

    return estimate_quantile


class QuadrupedTargetPursuit(QuadrupedPathFollow):
    """Naive proportional controller towards the goal"""

    @property
    def ref_pt(self) -> np.ndarray:
        return self.ref_path[-1]


def plot_stats(sims: list[cbf.Simulation]):
    import matplotlib.pyplot as plt

    plt.figure()
    sim = sims[0]
    time = np.arange(len(sim.stats["predicted_barrier"]) + 1)
    plt.plot(time[1:], sim.stats["predicted_barrier"], label="predicted barrier")
    plt.plot(time[:-1], sim.stats["barrier"], label="barrier", linestyle="--")
    plt.xlabel("Time step $k$")
    plt.ylabel("CBF $h(x_k)$")
    plt.axhline(0, linestyle=":", color="k")

    plt.figure()
    for sim in sims:
        plt.plot(sim.stats["nb_problems_solved"], color="k", alpha=0.5)
    plt.xlabel("Time step $k$")
    plt.ylabel("#optimization problems solved")

    plt.figure()
    all_times = np.ravel([sim.time for sim in sims])
    plt.hist(all_times / 10**6)
    plt.xlabel("Solver time [ms]")
    plt.ylabel("Rel. frequency")
    plt.yscale("log")


def plot_moving_obstacles(scene: cbf.Scene, t_end: int):
    for obst in scene.moving_obstacles:
        for t in range(0, t_end, 1):
            p = obst.get_poly_at_time_step(t)
            progress = t / t_end
            gp.plot_polytope(p, color="red", alpha=np.clip(progress, 0.01, 0.9))


def main(
    scene: cbf.Scene,
    config: PathPlanningSettings,
):
    """
    hor: simulation horizon
    n_reps: number of repeated experiments in the case of noise
    t_end: time step at which the obstacle is at its final destination
    """
    conf = config.total_conf
    μ = np.zeros(3)
    hor = config.simulation_horizon
    n_reps = config.n_reps

    assert scene.refline is not None, (
        f"Expected a scene with a reference line to follow. Got {scene}"
    )

    Σ = config.Σ
    one_step_conf = conf ** (1.0 / hor)
    print(
        f"Requesting stage-wise success rate of {one_step_conf:2.7f} to reach {conf:2.2f} total."
    )

    dynamics = cbf.quadruped(0.1)
    base_policy = QuadrupedPathFollow(scene.refline)

    x0 = np.array([*scene.refline[0], 0])
    no_noise = get_empty_sampler(dynamics.ns)

    f_sim = (
        get_closed_loop_dynamics(dynamics, QuadrupedPathFollow(scene.refline))
        if config.ignore_safety_filter
        else get_perturbed_dynamics(dynamics)
    )
    baseline_sim = cbf.Simulation(
        x0, f_sim, base_policy, hor, no_noise, label="base policy"
    )

    def g(x):
        return dynamics.g(x)[:2]

    if config.distribution == "gaussian":
        quantile_estimator = get_gaussian_quantile_estimator(g, μ, Σ)
    else:
        raise NotImplementedError(
            f"Distribution class {config.distribution} not supported yet!"
        )

    all_poly = scene.polyhedra + [
        o.get_poly_at_time_step(0) for o in scene.moving_obstacles
    ]
    print(f"Adding {len(all_poly)} polys in total")

    safety_filter = get_safety_filter(
        config,
        base_policy,
        one_step_conf,
        all_poly,
        dynamics,
        quantile_estimator,
    )

    if scene.moving_obstacles:
        n_static = len(scene.polyhedra)
        print("Detected moving obstacles!")

        # Policy also gets the obstacle positions
        def safe_policy(x, k):
            for i in range(len(scene.moving_obstacles)):
                safety_filter.poly[n_static + i] = scene.moving_obstacles[
                    i
                ].get_poly_at_time_step(k)
            return safety_filter(x, k)
    else:
        safe_policy = safety_filter

    rng = npr.default_rng(100)
    sampler = (
        get_loc_scale_sampler(
            μ, config.simulated_noise_scale_factor * Σ, rng, config.distribution
        )
        if config.simulate_noise
        else no_noise
    )

    sims_cbf = [
        cbf.Simulation(
            x0,
            f_sim,
            safe_policy,
            hor,
            sampler,
            get_stats=safety_filter.flush_stats,
            label=f"CBF run {i}",
        )
        for i in range(n_reps)
    ]

    for _, sim_cbf in enumerate(sims_cbf):
        if sim_cbf.failed:
            print(f"Simulation failed at time step {sim_cbf.fail_index}")
    return baseline_sim, sims_cbf


def generate_filename(config: PathPlanningSettings) -> str:
    return str(config) + ".json"


def export_results(
    config: PathPlanningSettings,
    base_sim: cbf.Simulation,
    cbf_sims: list[cbf.Simulation],
):
    filename = Path("data") / "path_planning" / generate_filename(config)
    output = dict(
        config=asdict(config),
        base_sim=base_sim.to_dict(),
        cbf_sims=[s.to_dict() for s in cbf_sims],
    )

    import json

    resp = (
        input(f"Write simulation to file {filename.resolve()}? [y/N] ").strip().lower()
    )
    if resp != "y":
        return
    os.makedirs(filename.parent, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
