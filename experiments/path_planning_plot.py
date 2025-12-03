"""
Generate a matplotlib-based version of Fig. 4 in the paper.
The figure in the paper was generated from the same data using Pykz (see https://pypi.org/project/pykz/).
"""

import cbf
from cbf.use_cases import path_planning
import matplotlib.pyplot as plt


if __name__ == "__main__":
    import os
    from pathlib import Path

    SCENE = "corridor-with-obstacles.svg"

    os.chdir(Path(__file__).parent)
    loaded_scene = cbf.Scene.from_svg(f"scenes/{SCENE}", scale=0.1)
    cfg = path_planning.PathPlanningSettings(
        SCENE,
        distribution_scale_x=0.03,
        distribution_scale_y=0.03,
        n_reps=30,
        simulation_horizon=150,
        max_input_size=5,
        formulation="ours",
        sorting_method=cbf.safety_filter.MAX_H_BASE,
    )

    baseline_sim, sims_cbf = path_planning.main(loaded_scene, cfg)
    plt.figure()
    loaded_scene.plot()
    baseline_sim.plot_state_trajectory(linestyle="--", color="black")
    for sim in sims_cbf:
        sim.plot_state_trajectory(color="forestgreen", alpha=0.1, marker="none")
    plt.axis("equal")
    plt.show()
