import geomin as gp
import cbf
import numpy as np
from cbf.literature.fushimi import Fushimi, get_param_fushimi
from cbf.literature.cosner import Cosner
from cbf.literature.base import QuadraticBarrier
from cbf.use_cases.corridor import CorridorSimulationSettings
import cbf.use_cases.corridor as corridor
import cvxpy as cp

# Define simulation configuration (see corridor.py for details)
config = CorridorSimulationSettings(
    simulation_horizon=20,
    n_reps=5000,
    Ts=0.1,
    start=[0.0, 0.0, 0.0],
    probs=(0.8, 0.9, 0.99, 0.999),  # 1-ε
    sigmas=(0.03,),
    seed=100,
    QCQP_solver=cp.ECOS,
    QCQP_solver_options={"feastol": 1e-7, "abstol": 1e-7, "reltol": 1e-7},
    # QCQP_solver_options={'tol_feas':1e-7,'tol_gap_abs':1e-7,'tol_gap_rel':1e-7},
    QP_solver=cp.ECOS,
    QP_solver_options={"feastol": 1e-7, "abstol": 1e-7, "reltol": 1e-7},
)


# Define controllers for each method
def get_fushimi_controller(config: CorridorSimulationSettings, a: float):
    dynamics = cbf.quadruped(config.Ts)
    A = np.zeros((dynamics.ns, dynamics.ns))
    A[1, 1] = -1
    c = 0.5**2
    b = np.zeros(dynamics.ns)
    qb = QuadraticBarrier(A, b, c)
    base_policy = corridor.base_policy

    def fushimi_controller(ε, σ):
        p = get_param_fushimi(ε, σ, c, config.simulation_horizon, a=a)
        if p is None:
            print("No feasibible parameters for Fushimi")
            return None
        Σ = np.zeros((dynamics.ns, dynamics.ns))
        Σ[1, 1] = σ**2
        policy = Fushimi(
            base_policy,
            req_conf_level=1 - ε,
            barrier=qb,
            dynamics=dynamics,
            Σ=Σ,
            a=p[0],
            β=p[1],
            solver=config.QCQP_solver,
            solver_opts=config.QCQP_solver_options,
        )
        return policy

    return fushimi_controller


def get_cosner_controller(config: CorridorSimulationSettings):
    dynamics = cbf.quadruped(config.Ts)
    A = np.zeros((dynamics.ns, dynamics.ns))
    A[1, 1] = -1
    c = 0.5**2
    b = np.zeros(dynamics.ns)
    qb = QuadraticBarrier(A, b, c)
    base_policy = corridor.base_policy

    def cosner_controller(ε, σ):
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
        return policy

    return cosner_controller


def get_safety_filter(config: CorridorSimulationSettings):
    obstacle1 = gp.Polyhedron.from_inequalities(np.array([[0, 1.0]]), np.array([-0.5]))
    obstacle2 = gp.Polyhedron.from_inequalities(np.array([[0, -1.0]]), np.array([-0.5]))
    obstacles = [obstacle1, obstacle2]
    base_policy = corridor.base_policy
    dynamics = cbf.quadruped(config.Ts)

    def safety_filter(ε, σ):
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
        return policy

    return safety_filter


if __name__ == "__main__":
    # Run simulations for different methods with gaussian noise
    fushimi_20_controller = get_fushimi_controller(config, a=20)
    fushimi_35_controller = get_fushimi_controller(config, a=35)
    fushimi_50_controller = get_fushimi_controller(config, a=50)
    cosner_controller = get_cosner_controller(config)
    safety_filter = get_safety_filter(config)

    failure_probs = []
    average_times = []
    infeasibilities = []
    max_times = []

    methods = {
        "Fushimi a=20": fushimi_20_controller,
        "Fushimi a=35": fushimi_35_controller,
        "Fushimi a=50": fushimi_50_controller,
        "Cosner": cosner_controller,
        "Ours": safety_filter,
    }
    for name, method in methods.items():
        print(f"Running method: {name}")
        fp, inf, avgT, maxT = corridor.main_simulation(config=config, get_policy=method)
        infeasibilities.append(inf)
        failure_probs.append(fp)
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
