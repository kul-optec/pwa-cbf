from cbf.literature.fushimi import Fushimi, get_param_fushimi
from cbf.literature.cosner import Cosner
from cbf.use_cases.corridor import CorridorInfeasSettings, main_infeas
from cbf.literature.base import QuadraticBarrier
from cbf.use_cases import corridor
import cbf
import os
import numpy as np
import geomin as gp
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Define infeasibility configuration (see corridor.py for details)

config = CorridorInfeasSettings(
    M=500,
    lσ=0.001,
    uσ=0.4,
    lε=-0.1,  # log
    uε=-10,  # log
    simulation_horizon=20,
    Ts=0.1,
    start=[0.0, 0.0, 0.0],
    QCQP_solver="ECOS",
    QCQP_solver_options={"feastol": 1e-7, "abstol": 1e-7, "reltol": 1e-7},
    QP_solver="ECOS",
    QP_solver_options={"feastol": 1e-7, "abstol": 1e-7, "reltol": 1e-7},
)


# Define infeasibility tests for each method
def get_fushimi_test(config: CorridorInfeasSettings):
    dynamics = cbf.quadruped(config.Ts)
    A = np.zeros((dynamics.ns, dynamics.ns))
    A[1, 1] = -1
    c = 0.5**2
    b = np.zeros(dynamics.ns)

    def fushimi_test(ε, σ):
        # A is hardcoded to be a corridor
        p = get_param_fushimi(ε, σ, c, config.simulation_horizon)
        # We do not explitly solve the controller as these parameters guarantee feasibility
        return p is None

    return fushimi_test


def get_cosner_test(config: CorridorInfeasSettings):
    dynamics = cbf.quadruped(config.Ts)
    A = np.zeros((dynamics.ns, dynamics.ns))
    A[1, 1] = -1
    c = 0.5**2
    b = np.zeros(dynamics.ns)
    qb = QuadraticBarrier(A, b, c)
    base_policy = corridor.base_policy

    def cosner_test(ε, σ):
        α = (1 - ε) ** (1 / config.simulation_horizon)
        Σ = np.zeros((dynamics.ns, dynamics.ns))
        Σ[1, 1] = σ**2
        policy = Cosner(
            base_policy,
            req_conf_level=1 - ε,
            barrier=qb,
            dynamics=dynamics,
            Σ=Σ,
            α=α,
            solver=config.QCQP_solver,
            solver_opts=config.QCQP_solver_options,
        )

        return policy(np.array(config.start)) is None

    return cosner_test


def get_safty_filter_test(config: CorridorInfeasSettings):
    obstacle1 = gp.Polyhedron.from_inequalities(np.array([[0, 1.0]]), np.array([-0.5]))
    obstacle2 = gp.Polyhedron.from_inequalities(np.array([[0, -1.0]]), np.array([-0.5]))
    obstacles = [obstacle1, obstacle2]
    base_policy = corridor.base_policy
    dynamics = cbf.quadruped(config.Ts)

    def safty_filter_test(ε, σ):
        def g(x):
            return dynamics.g(x)[:2]

        Σ = np.zeros((dynamics.ns, dynamics.ns))
        Σ[1, 1] = σ**2
        μ = np.zeros(dynamics.ns)
        policy = cbf.SafetyFilter(
            base_policy,
            (1 - ε) ** (1 / config.simulation_horizon),
            obstacles,
            dynamics,
            corridor.get_gaussian_quantile_estimator(g, μ, Σ),
            state_idx_filter=(0, 1),
            internal_solver=config.QP_solver,
            internal_solver_opts=config.QP_solver_options,
        )
        return policy(np.array(config.start), k=0) is None

    return safty_filter_test


if __name__ == "__main__":
    # Run the infeasibility experiments for each method
    fushimi_test = get_fushimi_test(config)
    cosner_test = get_cosner_test(config)
    safety_filter = get_safty_filter_test(config)
    methods = [fushimi_test, cosner_test, safety_filter]
    infeas_maps = []
    names = ["Fushimi", "Cosner", "Ours"]
    for method in methods:
        infeas_maps.append(main_infeas(config, method))

    # Plot the results
    fig, axs = plt.subplots(1, len(methods), figsize=(15, 5), sharex=True, sharey=True)
    cmap = ListedColormap(["green", "red"])
    sigmas = np.linspace(config.lσ, config.uσ, config.M)
    probs = np.logspace(config.lε, config.uε, config.M)
    X, Y = np.meshgrid(np.array(sigmas), np.array(probs))
    for i, (infeas, name) in enumerate(zip(infeas_maps, names)):
        contour0 = axs[i].contourf(X, Y, infeas, levels=config.M, cmap=cmap)
        axs[i].set_title(f"Infeas. {name}")
        axs[i].set_xlabel("σ")
        axs[i].set_ylabel("ε")
        axs[i].set_yscale("log")
        axs[i].set_ylim([0, 1])
        # fig.colorbar(contour0, ax=axs[0])
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/infeas.png")
    plt.show()
