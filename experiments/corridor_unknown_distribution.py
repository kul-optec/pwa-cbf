from typing import Callable
import geomin as gp
import cbf
import numpy as np
from cbf.literature.data_driven import Conformal, Scenario
from cbf.use_cases.corridor import CorridorSimulationSettings
import cbf.use_cases.corridor as corridor
import cvxpy as cp
import warnings

# Define simulation configuration (see corridor.py for details)
config = CorridorSimulationSettings(
    data_driven=True,
    simulation_horizon=20,
    n_reps=500,
    Ts=0.1,
    start=[0.0, 0.0, 0.0],
    probs=(0.1, 0.5, 0.9), #ε
    γ=0.2,
    sigmas=(0.03,),
    seed=100,
    data_seed=200,
    QCQP_solver=cp.CLARABEL,
    QCQP_solver_options={},
    QP_solver=cp.CLARABEL,
    QP_solver_options={},
    run_conformal=False,
    scenario_formulation="Fast",  # "Fast", "Stable" or "Specialized"(see data_driven.py)
)


def get_n_samples(ε: float, sim_horizon: int, γ: float):
    """Compute the number of samples"""
    δ = 1 - (1 - ε) ** (1 / sim_horizon)
    i_wise_γ = γ / 2
    n_samples = max(
        int(np.ceil(np.log(i_wise_γ) / np.log(1 - (δ) / 2))),
        int(np.ceil((np.log(sim_horizon / γ) / 2 / (δ) ** 2) * 1.3)),
    )
    return n_samples


def get_safety_filter_data_driven(
    config: CorridorSimulationSettings, get_n_samples: Callable
):
    obstacle1 = gp.Polyhedron.from_inequalities(np.array([[0, 1.0]]), np.array([-0.5]))
    obstacle2 = gp.Polyhedron.from_inequalities(np.array([[0, -1.0]]), np.array([-0.5]))
    obstacles = [obstacle1, obstacle2]
    base_policy = corridor.base_policy
    dynamics = cbf.quadruped(config.Ts)

    def safety_filter(ε, sampler):
        n_samples = get_n_samples(ε, config.simulation_horizon, config.γ)

        def g(x):
            return dynamics.g(x)[:2]

        policy = cbf.SafetyFilter(
            base_policy,
            (1 - ε) ** (1 / config.simulation_horizon),
            obstacles,
            dynamics,
            corridor.get_emperical_quantile_estimator(
                g, sampler, n_samples, config.γ / len(obstacles)
            ),
            state_idx_filter=(0, 1),
            internal_solver=config.QP_solver,
        )
        return policy

    return safety_filter


def get_conformal(config: CorridorSimulationSettings, get_n_samples: Callable):
    dynamics = cbf.quadruped(config.Ts)
    base_policy = corridor.base_policy

    def conformal_controller(ε, sampler):
        n_samples = get_n_samples(ε, config.simulation_horizon, config.γ)

        # Barrier coded inside the method
        policy = Conformal(
            base_policy,
            req_conf_level=(1 - ε) ** (1 / config.simulation_horizon),
            dynamics=dynamics,
            sampler=sampler,
            n_sample=n_samples,
            α=0.01,
            γ=config.γ / config.simulation_horizon,
        )
        return policy

    return conformal_controller


def get_scenario(config: CorridorSimulationSettings, get_n_samples: Callable):
    dynamics = cbf.quadruped(config.Ts)
    base_policy = corridor.base_policy

    def scenario_controller(ε, sampler):
        n_samples = get_n_samples(ε, config.simulation_horizon, config.γ)
        # Barrier coded inside the method
        policy = Scenario(
            base_policy,
            req_conf_level=1 - ε,
            dynamics=dynamics,
            sampler=sampler,
            n_sample=n_samples,
            α=0.01,
            γ=config.γ / config.simulation_horizon,
            solver=config.QCQP_solver,
            solver_opts=config.QCQP_solver_options,
            formulation=config.scenario_formulation,
        )
        return policy

    return scenario_controller


if __name__ == "__main__":
    # Run simulations for different methods with gaussian noise
    scenario_controller = get_scenario(config, get_n_samples)
    conformal_controller = get_conformal(config, get_n_samples)
    safety_filter_controller = get_safety_filter_data_driven(config, get_n_samples)
    failure_probs = []
    average_times = []
    max_times = []
    infeasibilities = []
    if config.run_conformal:
        methods = {
            "Conformal": conformal_controller,
            "Scenario": scenario_controller,
            "Ours": safety_filter_controller,
        }
    else:
        methods = {
            "Scenario": scenario_controller,
            "Ours": safety_filter_controller,
        }
        warnings.warn("The Conformal method is turned off. To turn it on put run_conformal=True in the settings. Beware of very long computation time")
    for name, method in methods.items():
        print(f"Running method: {name}")
        # Gaussian sampler in corridor
        fp, inf, avgT, maxT = corridor.main_simulation(config=config, get_policy=method)
        failure_probs.append(fp)
        infeasibilities.append(inf)
        average_times.append(avgT)
        max_times.append(maxT)

    # Print results
    for (name, method), fp, inf, avgT, maxT in zip(
        methods.items(), failure_probs, infeasibilities, average_times, max_times
    ):
        print(f"\n=== {name} ===")
        print("Success probabilities:")

        print(config.probs)

        print("Empirical Failure probabilities:")
        fp = np.where(1 - inf, fp.astype(str), "/")
        print(fp)
        print("Average times (ms):")
        print(avgT * 1e-6)

        print("Max time ms:")
        print(maxT * 1e-6)

