"""
simple class-based animation setup using matplotlib
Save as animation_example_typed.py and run: python animation_example_typed.py
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple, Union
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, MovieWriter
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
import geomin as gp

from cbf.simulation import SimulationRecord
from .scene import Scene


class BaseAnimation:
    """
    Minimal class-based animation framework with type hints.
    Subclass and implement `init_plot` and `update(frame)`.
    """

    fig: Figure
    ax: Axes
    interval: int
    frames: Optional[Union[int, Iterable[Any]]]
    _anim: Optional[FuncAnimation]

    def __init__(
        self,
        figsize: Tuple[float, float] = (6, 4),
        interval: int = 30,
        dpi: int = 100,
        clear_frame: bool = False,
        frames: Optional[Union[int, Iterable[Any]]] = None,
    ) -> None:
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.interval = interval
        self.frames = frames
        self._anim = None
        self._clear_frame = clear_frame

    def init_plot(self):
        """Create artists and return them as a sequence. Override in subclass."""
        raise NotImplementedError

    def update(self, frame: Any):
        """Update artists for a given frame and return them. Override in subclass."""
        raise NotImplementedError

    def start(
        self,
        save_path: Optional[Path] = None,
        writer: Optional[MovieWriter] = None,
        blit: bool = True,
        **save_kwargs: Any,
    ) -> FuncAnimation:
        """
        Start animation. If save_path is provided, saves to file.
        Returns the FuncAnimation object.
        """
        self._anim = FuncAnimation(
            self.fig,
            self.update,
            init_func=self.init_plot,
            frames=self.frames,
            interval=self.interval,
            blit=blit,
            repeat=False,
        )
        if save_path:
            if writer is None:
                # default: save as GIF if .gif else try mp4
                if str(save_path).endswith(".gif"):
                    writer = PillowWriter(fps=max(1, 1000 // self.interval))
                else:
                    writer = FFMpegWriter(fps=max(1, 1000 // self.interval))
            # type: ignore[arg-type]  -- MovieWriter accepted by FuncAnimation.save
            self._anim.save(str(save_path), writer=writer, **save_kwargs)
        else:
            plt.show()
        return self._anim


# Example subclass: animated sine wave
class SimAnimation(BaseAnimation):
    def __init__(
        self,
        simulations: list[SimulationRecord],
        interval_ms: int,
        scene: Scene | None = None,
        line_options: dict | None = None,
        base_sim: SimulationRecord | None = None,
        **kwargs,
    ) -> None:
        n_frames = np.max([s.n_steps for s in simulations])
        super().__init__(interval=interval_ms, frames=n_frames, **kwargs)
        self.scene = scene
        self.base_sim = base_sim
        self._line_opts = line_options if line_options is not None else dict()
        self.records = simulations
        self.initialize()

    def initialize(self):
        if self.scene is not None:
            self.scene.plot(ax=self.ax, line_opts={"color": "black", "ls": "--"})
        self.ax.set_xlabel("Position $x$")
        self.ax.set_ylabel("Position $y$")

        line_styles = {"lw": 2, "color": "dodgerblue", "alpha": 0.3}
        line_styles.update(self._line_opts)
        if self.base_sim is not None:
            self.base_traj = (
                self.ax.plot([], [], label="base policy", color="black")[0],
            )
            self.ax.legend()
        else:
            self.base_traj = tuple()
        self.trajectories = tuple(
            [self.ax.plot([], [], **line_styles)[0] for _ in self.records]
        )

    def init_plot(self):
        for line in self.trajectories:
            line.set_data([], [])
        if self.base_traj:
            self.base_traj[0].set_data([], [])
        return self.trajectories + self.base_traj

    def update(self, frame: Any):
        # frame may be an int or a value from an iterable
        # if isinstance(self.frames, int) and isinstance(frame, int):
        #     t = frame / max(1, (self.frames - 1))
        if self._clear_frame:
            self.ax.clear()
            self.initialize()
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        if frame == 0:
            return self.trajectories

        if self.scene is not None:
            for mo in self.scene.moving_obstacles:
                gp.plot_polytope(
                    mo.get_poly_at_time_step(frame - 1), ax=self.ax, color="tomato"
                )

        for record, line in zip(self.records, self.trajectories):
            line.set_data(record.x[:frame, 0], record.x[:frame, 1])

        if self.base_sim is not None:
            self.base_traj[0].set_data(
                self.base_sim.x[:frame, 0], self.base_sim.x[:frame, 1]
            )

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)  # Make sure we don't update the limits
        return self.trajectories + self.base_traj


class PolytopeAnimation(BaseAnimation):
    def __init__(
        self,
        simulation: SimulationRecord,
        interval_ms: int,
        scene: Scene | None = None,
        line_options: dict | None = None,
        base_sim: SimulationRecord | None = None,
        **kwargs,
    ) -> None:
        super().__init__(interval=interval_ms, frames=simulation.n_steps, **kwargs)
        self.scene = scene
        self.base_sim = base_sim
        self._line_opts = line_options if line_options is not None else dict()
        self.records = simulation
        self.initialize()

    def initialize(self):
        if self.scene is not None:
            self.scene.plot(ax=self.ax, line_opts={"color": "black", "ls": "--"})
        self.ax.set_xlabel("Position $x$")
        self.ax.set_ylabel("Position $y$")
        line_styles = {"lw": 2, "color": "dodgerblue", "alpha": 1}
        line_styles.update(self._line_opts)
        if self.base_sim is not None:
            self.base_traj = (
                self.ax.plot([], [], label="base policy", color="black")[0],
            )
            self.ax.legend()
        else:
            self.base_traj = tuple()
        self.trajectory = (self.ax.plot([], [], **line_styles)[0],)

    def init_plot(self):
        self.trajectory[0].set_data([], [])
        if self.base_traj:
            self.base_traj[0].set_data([], [])
        return self.trajectory + self.base_traj

    def update(self, frame: Any):
        import geomin as gp

        if self._clear_frame:
            self.ax.clear()
            self.initialize()
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        if frame == 0:
            return self.trajectory + self.base_traj

        if self.scene is not None:
            for mo in self.scene.moving_obstacles:
                gp.plot_polytope(
                    mo.get_poly_at_time_step(frame - 1), ax=self.ax, color="tomato"
                )
        j_index = self.records.j_selection
        if len(j_index) > frame and self.scene is not None:
            jj = j_index[frame]
            equiv_H = np.vstack(
                [
                    -p.H[j, :][np.newaxis, :]
                    for p, j in zip(self.scene.all_polyhedra(frame - 1), jj)
                ]
            )
            equiv_h = np.array(
                [-p.h[j] for p, j in zip(self.scene.all_polyhedra(frame - 1), jj)]
            )
            dx = xlim[-1] - xlim[0]
            dy = ylim[-1] - ylim[0]
            dxy = np.array([dx, dy])
            c = np.array([np.mean(xlim), np.mean(ylim)])
            box = gp.Box(c - 0.6 * dxy, c + 0.6 * dxy)
            p = gp.Polyhedron.from_inequalities(equiv_H, equiv_h).intersect(box)
            gp.plot_polytope(
                p,
                ax=self.ax,
                color="lightgreen",
                alpha=0.8 if self._clear_frame else 0.1,
            )

        self.trajectory[0].set_data(
            self.records.x[:frame, 0], self.records.x[:frame, 1]
        )

        if self.base_traj and self.base_sim is not None:
            self.base_traj[0].set_data(
                self.base_sim.x[:frame, 0], self.base_sim.x[:frame, 1]
            )
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)  # Make sure we don't update the limits

        return self.trajectory + self.base_traj

    def start(self, *args, **kwargs):
        kwargs["blit"] = False
        return super().start(*args, **kwargs)
