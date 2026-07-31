"""
STL -> (vertices, faces) loader for the 3D attitude animation.

Ported from the pre-package tree (legacy/STLLoader.py) during the 2026-07
restructure. Lets viz/attitude3d.py render the actual CAD export instead of
the schematic cube -- drop the satellite's STL path into
`deimos animate --stl <path>` (or animate_results(stl_path=...)) and the
mesh replaces the cube with no other change.

Requires numpy-stl (`pip install deimos[viz3d]`); trimesh is optional and
only used to decimate heavy meshes so matplotlib stays interactive.
"""

from __future__ import annotations

import numpy as np


def load_stl_mesh(path, target_extent=1.8, com_offset=None, max_faces=None):
    """
    Load an STL into the (vertices, faces) format AttitudeVisualizer expects.

    com_offset : vector FROM the STL's origin TO the CoM, in the STL's own
        units and frame. Vertices are shifted so the CoM lands at the plot
        origin -- this matters because the body rotates about the CoM, not the
        geometric origin. If None, the geometric centroid is used (visually
        fine for a near-symmetric body, but not the true rotation point if your
        CoM is offset).
    target_extent : largest bounding-box dimension after scaling. Keep it below
        the visualizer's axis limits (currently [-2, 2]), so ~1.8 leaves margin.
    max_faces : if set and trimesh is installed, decimate to ~this many tris.
    """
    try:
        from stl import mesh as stl_mesh          # numpy-stl
    except ImportError as e:
        raise ImportError(
            "Rendering an STL needs numpy-stl -- install the viz3d extra: "
            "pip install deimos[viz3d]") from e

    m = stl_mesh.Mesh.from_file(str(path))
    tris = m.vectors.astype(np.float64)       # (T, 3, 3): T triangles x 3 verts x xyz

    if max_faces is not None and len(tris) > max_faces:
        tris = _try_decimate(tris, max_faces)

    verts = tris.reshape(-1, 3)               # (T*3, 3), one row per triangle vertex
    faces = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(len(verts) // 3)]

    # recenter on the rotation point
    if com_offset is not None:
        verts = verts - np.asarray(com_offset, dtype=np.float64)
    else:
        verts = verts - verts.mean(axis=0)

    # uniform scale to fit existing axis limits
    extent = (verts.max(axis=0) - verts.min(axis=0)).max()
    if extent > 0:
        verts *= (target_extent / extent)

    return verts, faces


def _try_decimate(tris, max_faces):
    try:
        import trimesh
    except ImportError:
        print(f"[stl_mesh] trimesh not installed; rendering all {len(tris)} "
              f"faces (matplotlib will be slow). pip install trimesh to decimate.")
        return tris
    mesh = trimesh.Trimesh(
        vertices=tris.reshape(-1, 3),
        faces=np.arange(len(tris) * 3).reshape(-1, 3),
        process=True,                          # welds duplicate verts
    )
    mesh = mesh.simplify_quadric_decimation(max_faces)  # API name varies by trimesh version
    return mesh.triangles.astype(np.float64)
