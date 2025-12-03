from __future__ import annotations

import geomin as gp
from typing import List, Optional, TYPE_CHECKING
import numpy as np
from pathlib import Path

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class MovingObstacle:
    def __init__(
        self,
        base_poly: gp.Polyhedron,
        refline: np.ndarray,
        time_steps: int,
        name: str | None = None,
    ):
        self.refline = refline
        self.name = name
        self.poly = base_poly
        self.time_steps = time_steps

    def pathlengths(self) -> np.ndarray:
        diffs = np.diff(
            self.refline, axis=0, prepend=self.refline[0, :].reshape((1, 2))
        )
        return np.linalg.norm(diffs, axis=1)

    def cumul_pathlengths(self) -> np.ndarray:
        return np.cumsum(self.pathlengths())

    def get_poly_at_time_step(self, t: int) -> gp.Polyhedron:
        return self._get_poly_at_t(min(t / self.time_steps, 1))

    def _get_poly_at_t(self, t: float) -> gp.Polyhedron:
        cp = self.cumul_pathlengths()
        progress = cp / cp[-1]
        i = np.searchsorted(progress, t, side="right")
        i = np.clip(i, 0, len(progress) - 1)
        end_t = progress[i]
        if len(self.refline) == 1:
            pos = self.refline[0]
            heading = 0
        else:  # Interpolate
            start_t = progress[i - 1]
            wgt = (t - start_t) / (end_t - start_t)
            assert wgt >= 0
            assert wgt <= 1
            pos = self.refline[i - 1] + wgt * (self.refline[i] - self.refline[i - 1])
            segment_dir = self.refline[i] - self.refline[i - 1]
            heading = np.arctan2(segment_dir[1], segment_dir[0])

        R = np.array(
            [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
        )
        verts = self.poly.vertices()
        vertices = np.vstack([(R @ v + pos)[np.newaxis, :] for v in verts])
        poly = gp.Polyhedron.from_generators(vertices)
        return poly


class Scene:
    def __init__(
        self,
        polyhedra: List[gp.Polyhedron],
        refline: np.ndarray | None = None,
        moving_obstacles: list[MovingObstacle] | None = None,
        name: str | None = None,
    ):
        self.polyhedra = polyhedra
        self.refline = refline
        self.moving_obstacles = [] if moving_obstacles is None else moving_obstacles
        self.name = name if name is not None else "Scene"
        if self.polyhedra:
            np.testing.assert_array_equal(
                [p.dim for p in self.polyhedra],
                len(self.polyhedra) * [self.polyhedra[0].dim],
                f"All polyhedra should have the same underlying dimensions. Got dimensions: {[p.dim for p in self.polyhedra]}",
            )

    def __str__(self) -> str:
        s = f"Scene(name='{self.name}', dim={self.polyhedra[0].dim if self.polyhedra else '∅'}, moving={len(self.moving_obstacles)}, "
        s += f"polyhedra={len(self.polyhedra)}"
        if self.refline is not None:
            s += f", refline_shape={self.refline.shape}"
        s += ")"
        return s

    @property
    def dim(self) -> Optional[int]:
        if not self.polyhedra:
            return
        return self.polyhedra[0].dim

    def plot(
        self,
        ax: Axes | None = None,
        title: str = "",
        show: bool = False,
        line_opts: dict | None = None,
    ) -> Optional[Axes]:
        if not self.polyhedra:
            return

        for p in self.polyhedra:
            ax = gp.plot_polytope(p, ax=ax)

        if ax is None:
            return

        if self.refline is not None:
            options = dict(label="Ref. line")
            if line_opts:
                options.update(line_opts)
            line = ax.plot(*self.refline.T, **options)
            ax.scatter(
                *self.refline[0], marker="x", color=line[0].get_color(), label="start"
            )
            ax.scatter(
                *self.refline[-1], marker="*", color=line[0].get_color(), label="end"
            )
            ax.legend()

        if title and ax is not None:
            ax.set_title(title)

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        return ax

    def all_polyhedra(self, k: int):
        poly = [p for p in self.polyhedra]
        for m in self.moving_obstacles:
            poly.append(m.get_poly_at_time_step(k))
        return poly

    def plot_pykz(
        self,
        *,
        poly_opts: dict | None = None,
        line_opts: dict | None = None,
    ):
        from .plotting_pykz import plot_polytope2d_pykz
        import pykz

        if not self.polyhedra:
            return

        for p in self.polyhedra:
            poly_opts = dict() if poly_opts is None else poly_opts
            plot_polytope2d_pykz(p, **poly_opts)

        if self.refline is not None:
            options = dict(label="Ref. line", color="gray")
            if line_opts:
                options.update(line_opts)
            # line = pykz.plot(*self.refline.T, **options)
            pykz.scatter(
                *self.refline[0], mark="x", color=options["color"], label="start"
            )
            pykz.scatter(
                *self.refline[-1], mark="*", color=options["color"], label="end"
            )
            # ax.legend()

    @classmethod
    def load(cls, filename: str | Path) -> "Scene":
        from .io import load_polygons

        polygons = load_polygons(filename)
        return cls(polygons)

    @classmethod
    def from_svg(cls, file: str | Path, scale: float = 1.0) -> "Scene":
        from .io import get_svg_paths_by_label, get_pathlength_by_label

        file_path = Path(file)

        obstacle_verts = get_svg_paths_by_label(file_path, "obstacle")
        polys = [gp.Polyhedron.from_generators(c * scale) for c in obstacle_verts]
        ref_line = get_svg_paths_by_label(file_path, "refline")
        ref_line = ref_line[0] * scale if len(ref_line) > 0 else None

        moving_obstacles = []
        moving_obstacle_polys = get_svg_paths_by_label(file_path, "moving_obstacle1")
        if len(moving_obstacle_polys) > 1:
            raise NotImplementedError("Multiple moving obstacles not yet supported.")

        for mov in moving_obstacle_polys:
            ref_line_obstacle = get_svg_paths_by_label(file_path, "refline_obstacle1")
            path_length = get_pathlength_by_label(file_path, "refline_obstacle1")
            print(f"Found path length {path_length}")
            assert len(ref_line_obstacle) == 1, (
                "Expected one ref. path for a moving obstacle."
            )
            poly = gp.Polyhedron.from_generators((mov - np.mean(mov, axis=0)) * scale)
            moving_obstacles.append(
                MovingObstacle(poly, ref_line_obstacle[0] * scale, path_length[0])
            )

        return Scene(polys, ref_line, moving_obstacles, name=file_path.name)
