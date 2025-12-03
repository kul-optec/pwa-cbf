import cbf
from cbf.simulation import SimulationRecord
from cbf.use_cases import path_planning
import numpy as np


def compute_input_error(sim: SimulationRecord):
    u_base = np.array(sim.stats["u_base"])
    u = np.array(sim.u)
    return np.sum((u_base - u) ** 2, axis=1)


if __name__ == "__main__":
    import os
    from pathlib import Path

    SCENE = "corridor-with-obstacles.svg"

    os.chdir(Path(__file__).parent)
    loaded_scene = cbf.Scene.from_svg(f"scenes/{SCENE}", scale=0.1)

    solvers = ["ours", "miqp"]

    barrier_values = dict()
    costs = dict()
    for s in solvers:
        cfg = path_planning.PathPlanningSettings(
            SCENE,
            distribution_scale=0.03,
            n_reps=30,
            simulation_horizon=150,
            max_input_size=5,
            formulation=s,
            internal_MIQPsolver="GUROBI",
            internal_QPsolver="QPALM",
            ignore_safety_filter=True,
            sorting_method=cbf.safety_filter.MAX_H_BASE,
        )

        baseline_sim, sims_cbf = path_planning.main(loaded_scene, cfg)
        barrier_values[s] = np.ravel([sim.stats["barrier"] for sim in sims_cbf])
        costs[s] = np.ravel([compute_input_error(sim.export()) for sim in sims_cbf])

    lowest_costs = np.min([c for c in costs.values()], axis=0)

    method = "ours"

    # Flags
    is_safe = barrier_values[method] >= 0
    filter_active = lowest_costs > 1e-4

    mask = filter_active * is_safe
    relative_suboptimality = (costs[method][mask] - lowest_costs[mask]) / (
        lowest_costs[mask] + 1e-7
    )

    for q in [0.9, 0.95, 0.99, 1]:
        print(
            f"rel subopt {method} (q({q}): {np.quantile(relative_suboptimality, q)} of {len(relative_suboptimality)} points"
        )
