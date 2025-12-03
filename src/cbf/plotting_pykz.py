import geomin as gp
import pykz


def plot_polytope2d_pykz(polyhedron: gp.Polyhedron, **style):
    if not isinstance(polyhedron, gp.Polyhedron):
        raise TypeError(f"Expected a Polyhedron to plot. Got a {type(polyhedron)}")

    default_style = dict(fill_opacity=0.5, fill="blue", draw="blue", line_cap="round")
    default_style.update(style)

    if polyhedron.size != 2:
        raise ValueError(
            f"Can only plot polyhedrons of dimension 2, got {polyhedron.size}."
        )
    if not polyhedron.is_compact():
        raise ValueError("Can only plot bounded polyhedron.")

    verts = polyhedron.vertices()
    if len(verts) == 0:
        return

    # faces = polyhedron.get_face_incidence()
    neighbors = polyhedron.get_vertex_adjacency()

    previous, current = -1, 0

    order = []
    for _ in range(len(verts) - 1):
        order.append(current)
        (current, *_), previous = neighbors[current] - set([previous]), current
    order.append(current)

    pykz.plot(verts[order, 0], verts[order, 1], **default_style)
