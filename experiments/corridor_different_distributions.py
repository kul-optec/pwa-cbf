from email.charset import QP
from os import name
import os
import matplotlib.pyplot as plt
import geomin as gp
import cbf
import numpy as np
from cbf.use_cases.corridor import CorridorSimulationSettings
import cbf.use_cases.corridor as corridor
import cvxpy as cp

config = CorridorSimulationSettings(
    data_driven=True,
    simulation_horizon=20,
    n_reps=2000,
    Ts=0.1,
    start=[0.0, 0.0, 0.0],
    probs=(0.9,),
    γ=0.01,
    sigmas=(0.03,),
    seed=100,
    data_seed=200,
    QP_solver=cp.CLARABEL,
)


def get_safety_filter_data_driven(config: CorridorSimulationSettings, n_samples: int):
    obstacle1 = gp.Polyhedron.from_inequalities(np.array([[0, 1.0]]), np.array([-0.5]))
    obstacle2 = gp.Polyhedron.from_inequalities(np.array([[0, -1.0]]), np.array([-0.5]))
    obstacles = [obstacle1, obstacle2]
    base_policy = corridor.base_policy
    dynamics = cbf.quadruped(config.Ts)

    def safety_filter(ε, sampler):
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


if __name__ == "__main__":
    # If more than one prob, make n_get_samples function
    stage_wise_success_rate = (config.probs[0]) ** (1 / config.simulation_horizon)
    n_obst = 2

    # Compute minimum number of samples needed for the saftety filter to have the desired guarantees
    n_min = int(
        np.ceil(
            np.log(config.γ / n_obst)
            / np.log(1 - (1 - stage_wise_success_rate) / n_obst)
        )
    )
    n_samples = n_min * 2 ** np.arange(0, 8)
    failure_probs = []

    # Run simulations for different noise distributions and different number of samples
    noises = {
        "Student-t": corridor.get_student_sampler,
        "Laplace": corridor.get_laplace_sampler,
        "Gaussian": corridor.get_gaussian_sampler_corridor,
    }
    noise_colors = ["tab:green", "tab:blue", "tab:orange"]
    for noise_name, sampler in noises.items():
        print(f"Running noise type: {noise_name}")
        for n_s in n_samples:
            print(f"Number of samples: {n_s}")
            safety_filter_controller = get_safety_filter_data_driven(config, int(n_s))
            fp, inf, avgT, maxT = corridor.main_simulation(
                config=config, get_policy=safety_filter_controller, get_sampler=sampler
            )
            failure_probs.append((noise_name, n_s, fp))

    # Plot results
    plt.figure(figsize=(8, 6))
    for color, noise_name in zip(noise_colors, noises.keys()):
        fps = [fp for name, s, fp in failure_probs if name == noise_name]
        plt.plot(n_samples, np.squeeze(fps), label=noise_name, color=color)
    plt.xscale("log")
    plt.legend()
    plt.xlabel("Number of samples")
    plt.ylabel("Failure Probability")
    plt.title("Failure Probability vs Sample Size for different noise distributions")
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/Fail_vs_Samp.png")
    plt.show()
