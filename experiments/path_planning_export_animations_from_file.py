import cbf
import os
from pathlib import Path
import json

sorting_method = cbf.safety_filter.MAX_H_BASE

SCENE = "corridor-with-obstacles.svg"
filename = f"{SCENE}_gaussian(0.030)-ours-{sorting_method}-umax=5-conf=0.9.json"
path = Path("data") / "path_planning" / filename

os.chdir(Path(__file__).parent)
loaded_scene = cbf.Scene.from_svg(f"scenes/{SCENE}", scale=0.1)

with open(path, "r") as f:
    data = json.load(f)

cbf_sims = [cbf.SimulationRecord(**s) for s in data["cbf_sims"]]

animation = cbf.SimAnimation(
    cbf_sims,
    100,
    loaded_scene,
    dpi=80,
)
animation.start()
