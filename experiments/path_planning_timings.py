import cbf
from cbf.use_cases import path_planning
import numpy as np

if __name__ == "__main__":
    import os
    from pathlib import Path

    SCENE = "corridor-with-obstacles.svg"

    os.chdir(Path(__file__).parent)
    loaded_scene = cbf.Scene.from_svg(f"scenes/{SCENE}", scale=0.1)

    solvers = ["ours", "miqp"]

    times = dict()
    for s in solvers:
        cfg = path_planning.PathPlanningSettings(
            SCENE,
            distribution_scale_x=0.03,
            distribution_scale_y=0.03,
            n_reps=30,
            simulation_horizon=150,
            max_input_size=5,
            formulation=s,
            internal_MIQPsolver="GUROBI",
            internal_QPsolver="QPALM",
            internal_MIQPsolver_opts={},
            internal_QPsolver_opts={},
            ignore_safety_filter=False,
            sorting_method=cbf.safety_filter.MAX_H,
        )

        baseline_sim, sims_cbf = path_planning.main(loaded_scene, cfg)
        times[s] = np.ravel([sim.time for sim in sims_cbf])
        mean_time = np.mean([np.mean(sim.time) for sim in sims_cbf])
        max_time = np.max([np.max(sim.time) for sim in sims_cbf])
        print(
            f"Time {s}: {mean_time / (10**6)} ms (mean) - {max_time / (10**6)} ms (max)"
        )

    speedups = times["miqp"] / times["ours"]
    print(
        f"Speedups: min={speedups.min()},\nmean={speedups.mean()},\nq10={np.quantile(speedups, 0.1)},\nmax={speedups.max()}"
    )
