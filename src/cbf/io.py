from dataclasses import dataclass
from pathlib import Path
import geomin as gp
from typing import List
import numpy as np
from svgpathtools import svg2paths
from svgpathtools import Path as SvgPath


def load_polygons(filename: str | Path) -> List[gp.Polyhedron]:
    data = np.load(filename)
    polys = [gp.Polyhedron.from_generators(data[v]) for v in data.files]
    return polys


@dataclass
class SvgPathDescription:
    path: SvgPath
    attr: dict


def filter_svg_by_label(svg_file: Path, label: str) -> list[SvgPathDescription]:
    if not svg_file.is_file():
        raise FileNotFoundError(f"File {svg_file} is not a file.")

    paths, attributes = svg2paths(str(svg_file))

    filtered_paths = []
    for path, attr in zip(paths, attributes):
        lab = attr.get("inkscape:label")
        if lab == label or attr.get("id") == label:
            filtered_paths.append(SvgPathDescription(path, attr))
    return filtered_paths


def get_svg_paths_by_label(svg_file: Path, label: str) -> list[np.ndarray]:
    filtered_paths = filter_svg_by_label(svg_file, label)

    def convert(path: SvgPath):
        coords = [(seg.start.real, seg.start.imag) for seg in path]
        coords.append((path[-1].end.real, path[-1].end.imag))  # include final endpoints
        return np.array(coords)

    return [convert(p.path) for p in filtered_paths]


def get_pathlength_by_label(svg_file: Path, label: str) -> List[int]:
    filtered_paths = filter_svg_by_label(svg_file, label)

    def convert(path: SvgPathDescription):
        return int(path.attr.get("data-time", path.path.length()))

    return [convert(p) for p in filtered_paths]
