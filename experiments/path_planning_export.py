import cbf
from cbf.use_cases import path_planning
import numpy as np


if __name__ == "__main__":
    import os
    from pathlib import Path

    SCENE = "corridor-with-obstacles.svg"

    os.chdir(Path(__file__).parent)
    loaded_scene = cbf.Scene.from_svg(f"scenes/{SCENE}", scale=0.1)
    if loaded_scene.moving_obstacles:
        loaded_scene.plot()
        path_planning.plot_moving_obstacles(loaded_scene, 100)
        cfg = path_planning.PathPlanningSettings(SCENE, n_reps=1)
        path_planning.main(loaded_scene, cfg)
    else:
        cfg = path_planning.PathPlanningSettings(
            SCENE,
            distribution_scale_x=0.03,
            distribution_scale_y=0.03,
            n_reps=30,
            simulation_horizon=150,
            max_input_size=1,
            formulation="ours",
            ignore_safety_filter=False,
            sorting_method=cbf.safety_filter.MAX_H_BASE,
        )

        baseline_sim, sims_cbf = path_planning.main(loaded_scene, cfg)
        mean_time = np.mean([np.mean(sim.time) for sim in sims_cbf])
        max_time = np.max([np.max(sim.time) for sim in sims_cbf])
        print(f"Time: {mean_time / (10**6)} ms (mean) - {max_time / (10**6)} ms (max)")
        try:
            path_planning.plot_stats(sims_cbf)
        except KeyError as e:
            print(f"Couldn't plot stats because keys were missing:\n{e}")

        path_planning.export_results(cfg, baseline_sim, sims_cbf)
