import cbf
import os
from pathlib import Path
from cbf.use_cases import path_planning as pp
from dataclasses import dataclass

sorting_method = cbf.safety_filter.MAX_H_BASE


@dataclass
class AniCfg:
    name: str
    path_planning_cfg: pp.PathPlanningSettings
    force_rerun: bool = False
    enable: bool = True
    clear_frame: bool = True


settings = [
    AniCfg(
        "corridor_maxhb",
        pp.PathPlanningSettings(
            "corridor-with-obstacles.svg",
            distribution_scale_x=0.02,
            distribution_scale_y=0.02,
            n_reps=10,
            simulation_horizon=100,
            max_input_size=3,
            total_conf=0.99,
            sorting_method=cbf.safety_filter.MAX_H_BASE,
        ),
        enable=False,
        force_rerun=False,
    ),
    AniCfg(
        "corridor_maxh",
        pp.PathPlanningSettings(
            "corridor-with-obstacles.svg",
            distribution_scale_x=0.02,
            distribution_scale_y=0.02,
            n_reps=10,
            simulation_horizon=100,
            max_input_size=3,
            total_conf=0.99,
            sorting_method=cbf.safety_filter.MAX_H,
        ),
        enable=False,
        force_rerun=False,
    ),
    AniCfg(
        "stresstest",
        pp.PathPlanningSettings(
            "stresstest.svg",
            distribution_scale_x=0.01,
            distribution_scale_y=0.01,
            n_reps=10,
            simulation_horizon=180,
            max_input_size=4,
            total_conf=0.90,
            sorting_method=cbf.safety_filter.MAX_H_BASE,
        ),
        enable=False,
        force_rerun=False,
    ),
    AniCfg(
        "straight_hall_moving",
        pp.PathPlanningSettings(
            "straight-hallway-moving-obstacle.svg",
            distribution_scale_x=0.1,
            distribution_scale_y=0.03,
            simulated_noise_scale_factor=0.05,
            n_reps=20,
            simulation_horizon=100,
            max_input_size=1.0,
            total_conf=0.90,
            sorting_method=cbf.safety_filter.MAX_H_BASE,
        ),
        enable=False,
        force_rerun=False,
    ),
    AniCfg(
        "n-gon",
        pp.PathPlanningSettings(
            "n-gon.svg",
            distribution_scale_x=0.1,
            distribution_scale_y=0.1,
            n_reps=100,
            simulation_horizon=50,
            max_input_size=3,
            total_conf=0.99,
            sorting_method=cbf.safety_filter.MAX_H_BASE,
        ),
        enable=False,
        force_rerun=False,
        clear_frame=True,
    ),
    AniCfg(
        "simple-2",
        pp.PathPlanningSettings(
            "simple_2_obstacles.svg",
            distribution_scale_x=0.1,
            distribution_scale_y=0.1,
            n_reps=20,
            simulation_horizon=150,
            max_input_size=3,
            total_conf=0.99,
            sorting_method=cbf.safety_filter.MAX_H_BASE,
        ),
        enable=True,
        force_rerun=False,
        clear_frame=True,
    ),
]

os.chdir(Path(__file__).parent)
gif_dir = Path("gifs")
os.makedirs(gif_dir, exist_ok=True)
DPI = 80

for setup in settings:
    cfg = setup.path_planning_cfg
    name = setup.name
    clf = "_clf" if setup.clear_frame else ""
    gif_path_base = gif_dir / (name + clf + ".gif")
    gif_path_poly = gif_dir / (name + clf + "_poly.gif")
    if not setup.enable:
        print(f"Skipping `{name}` because `enable` is set to False")
        continue

    skip = gif_path_base.exists() and gif_path_poly.exists()
    skip = False if setup.force_rerun else skip
    if skip:
        print(
            f"Skipping `{name}` because the gif exists. Set force_rerun=True or delete the corresponding gifs to rerun."
        )
        continue

    print(f"Running {cfg.n_reps} simulations...")
    loaded_scene = cbf.Scene.from_svg(f"scenes/{cfg.scene}", scale=0.1)
    print(loaded_scene)
    base_sim, cbf_sims = pp.main(loaded_scene, cfg)

    cbf_sims = [s.export() for s in cbf_sims]

    print(f"Rendering animation for {name} ...")
    animation = cbf.SimAnimation(
        cbf_sims,
        100,
        loaded_scene,
        dpi=DPI,
        base_sim=base_sim.export(),
        clear_frame=setup.clear_frame,
    )
    animation.start(gif_path_base)

    print(f"Rendering poly animation for {name} ...")
    animation = cbf.PolytopeAnimation(
        cbf_sims[0],
        100,
        loaded_scene,
        dpi=DPI,
        base_sim=base_sim.export(),
        clear_frame=setup.clear_frame,
    )

    animation.start(gif_path_poly)
