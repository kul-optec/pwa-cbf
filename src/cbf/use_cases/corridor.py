"""
Module for the corridor experiment.

Dynamics:
    x_{k+1} = x_k + Ts * R_z(θ_k) * u_k + d_k
where
    x_k = [x_k, y_k, θ_k]^T,
    u_k = [v_{x,k}, v_{y,k}, ω_k]^T,
    R_z(θ_k) is the rotation matrix around the z-axis by angle θ_k,
    d_k = [0, w_k, 0]^T with w_k a random disturbance.

Constraint:
    Corridor constraint: 0.5 - |y_k| >= 0

This module provides:
    - Tools for infeasibility map computation
    - Tools for simulations with known and unknown disturbance distributions
"""

import cbf
from cbf.dynamics import State, Input, Noise
import numpy as np
from typing import Callable, Literal, List
import numpy.random as npr
from dataclasses import dataclass, field
from typing import Tuple
import cvxpy as cp

NomDynamics = Callable[[State, Input, Noise], State]
InfeasTest = Callable[[float, float], bool]

Policy = Callable[[State, int], Input]
FalliblePolicy = Callable[[State, int], Input | None]


@dataclass
class CorridorInfeasSettings:
    """Settings for infeasibility map computation in the corridor experiment.
    M is the number of points per axis (σ, ε).
    lσ and uσ are the lower and upper bounds for the standard deviation of the noise.
    lε and uε are the lower and upper bounds for the logarithm (base 10) of the risk level ε."""

    M: int = 50
    lσ: float = 0.01
    uσ: float = 0.4
    lε: float = -0.1  # log scale
    uε: float = -10  # log scale
    simulation_horizon: int = 20
    Ts: float = 0.1
    start: List[float] = field(default_factory=List)
    QCQP_solver: str = cp.CLARABEL
    QCQP_solver_options: dict = field(default_factory=dict)
    QP_solver: str = cp.CLARABEL
    QP_solver_options: dict = field(default_factory=dict)


@dataclass
class CorridorSimulationSettings:
    """Settings for the simulations in the corridor example.
    If data_driven is True, then samples will be used else the known distribution is used.
    n_reps is the number of simulations to run for each (σ, ε) pair
    probs are the success probabilities over the whole simulation (1-ε) to test.
    sigmas are the standard deviations (σ) to test.
    γ is the outer confidence parameter over a whole simulation for data-driven part.
    """

    data_driven: bool = False
    simulation_horizon: int = 20
    n_reps: int = 5000
    Ts: float = 0.1
    start: List[float] = field(default_factory=List)
    probs: Tuple = (0.80, 0.9, 0.99, 0.999)
    sigmas: Tuple = (0.03,)
    seed: int = 100
    QCQP_solver: str = cp.CLARABEL
    QCQP_solver_options: dict = field(default_factory=dict)
    QP_solver: str = cp.CLARABEL
    QP_solver_options: dict = field(default_factory=dict)
    # if data_driven is True
    data_seed: int = 200
    γ: float = 0.2
    run_conformal: bool = False
    # WARNING Only use Specialized if θ_k = 0 in the corridor example
    scenario_formulation: Literal["Fast", "Stable", "Specialized"] = "Fast"


def get_perturbed_dynamics(dynamics: cbf.Dynamics):
    def noisy_dyn(x: cbf.State, u: cbf.Input, w: cbf.Noise):
        return dynamics.f(x, u) + dynamics.g(x) @ w

    return noisy_dyn


def get_empty_sampler(ns: int):
    def sample():
        return np.zeros(ns)

    return sample


def get_gaussian_sampler_corridor(μ_1: float, σ: float, rng: npr.Generator):
    from scipy.stats import multivariate_normal

    Σ = np.diag([0, σ**2, 0])
    μ = np.array([0, μ_1, 0])
    ξ_dist = multivariate_normal(μ, Σ, allow_singular=True)  # shape (N_samples,3)

    def sample(N_samples: int = 1):
        return np.array(ξ_dist.rvs(N_samples, random_state=rng))

    return sample


def get_laplace_sampler(μ: float, σ: float, rng: npr.Generator):
    from scipy.stats import laplace

    ξ_dist = laplace(μ, σ)

    def sample(N_samples: int = 1):
        samples = ξ_dist.rvs(N_samples, random_state=rng)  # shape (N_samples,)
        samples = np.column_stack([np.zeros(N_samples), samples, np.zeros(N_samples)])
        return np.squeeze(samples)

    return sample


def get_student_sampler(μ: float, σ: float, rng: npr.Generator, dv=8):
    from scipy.stats import t as student_t

    ξ_dist = student_t(dv, μ, σ)

    def sample(N_samples: int = 1):
        samples = ξ_dist.rvs(N_samples, random_state=rng)  # shape (N_samples,)
        samples = np.column_stack([np.zeros(N_samples), samples, np.zeros(N_samples)])
        return np.squeeze(samples)

    return sample


def get_gaussian_quantile_estimator(
    g: Callable[[cbf.State], np.ndarray], μ: np.ndarray, Σ: np.ndarray
):
    def estimate_quantile(x: cbf.State, ci: np.ndarray, level: float):
        trans = np.atleast_1d(ci @ g(x))
        new_μ, new_σ = cbf.transform_gaussian(μ, Σ, trans)
        q_est = cbf.NormalQuantile(new_μ, new_σ)
        return q_est(level)

    return estimate_quantile


def get_emperical_quantile_estimator(
    g: Callable[[cbf.State], np.ndarray], sampler: Callable, N_samples: int, γ: float
):
    samples = sampler(N_samples)

    def estimate_quantile_emp(x: State, ci: np.ndarray, level: float):
        trans = np.atleast_1d(ci @ g(x))
        trans = trans @ samples.T
        q_est = cbf.DistributionFreeQuantile(trans)
        return q_est(level, γ)

    return estimate_quantile_emp


def base_policy(x: np.ndarray, _k: int) -> np.ndarray:
    return np.array([0.2, 1, float(-x[2])])


def unsafe(sim: cbf.Simulation) -> bool:
    return bool(np.any(np.abs(sim.x[:, 1]) > 0.5))


def main_infeas(config: CorridorInfeasSettings, test_inf: InfeasTest):
    """
    Compute the infeasibility map for the corridor experiment for a given
    configuration and infeasibility test.
    """
    sigmas = np.linspace(config.lσ, config.uσ, config.M)
    probs = np.logspace(config.lε, config.uε, config.M)
    infeas = np.empty((config.M, config.M))
    for idxs, σ in enumerate(sigmas):
        for idxe, ε in enumerate(probs):
            infeas[idxe, idxs] = test_inf(ε, σ)
    return infeas


def main_simulation(
    config: CorridorSimulationSettings,
    get_policy: Callable,
    get_sampler: Callable = get_gaussian_sampler_corridor,
):
    """
    Run corridor-experiment simulations for a given configuration, controller generator,
    and noise sampler.

    For each combination of (σ, ε) specified in config.sigmas and config.probs,
    this function performs config.n_reps Monte-Carlo simulations and computes:

        • Empirical failure probabilities (corridor constraint violation)
        • Controller infeasibility
        • Average controller evaluation time
        • Maximum controller evaluation time

    When config.data_driven is True, a new controller is generated for each simulation
    (using get_policy) with an independently sampled dataset.
    """

    dynamics = cbf.quadruped(config.Ts)
    f_sim = get_perturbed_dynamics(dynamics)
    shape = (len(config.sigmas), len(config.probs))
    fail_prob = np.empty(shape)
    average_time = np.zeros(shape)
    max_time = np.zeros(shape)
    infeas = np.zeros(shape)
    start = np.array(config.start)
    for idxs, σ in enumerate(config.sigmas):
        for idxe, succes_prob in enumerate(config.probs):
            rng = npr.default_rng(config.seed)
            # change if non-zero mean noise
            μ = 0
            sampler = get_sampler(μ, σ, rng)
            if config.data_driven:
                rng_data = npr.default_rng(config.data_seed)
                # get_policy every simulation to obtain independent simulations (other data_sets)

                noisy_sim = [
                    cbf.Simulation(
                        start,
                        f_sim,
                        get_policy(1 - succes_prob, get_sampler(μ, σ, rng_data)),
                        config.simulation_horizon,
                        sampler,
                    )
                    for _ in range(config.n_reps)
                ]
            else:
                policy = get_policy(1 - succes_prob, σ)
                noisy_sim = [
                    cbf.Simulation(
                        start, f_sim, policy, config.simulation_horizon, sampler
                    )
                    for _ in range(config.n_reps)
                ]
            # data collection
            exit_count = np.count_nonzero([unsafe(s) for s in noisy_sim])
            fail_prob[idxs, idxe] = exit_count / len(noisy_sim)
            for i, sim in enumerate(noisy_sim):
                if sim.failed:
                    infeas[idxs, idxe] = 1
                    print(f"infeasible at time step  {sim.fail_index}")
                else:
                    mean = np.mean(np.array(sim.time)) / len(noisy_sim)
                    average_time[idxs, idxe] = average_time[idxs, idxe] + mean
                    max_time[idxs, idxe] = max(
                        max_time[idxs, idxe], np.max(np.array(sim.time))
                    )

    return fail_prob, infeas, average_time, max_time
