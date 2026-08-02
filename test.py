import trimesh, sys, numpy as np


def load(p):
    g = trimesh.load(p, process=False)
    return g.to_geometry() if isinstance(g, trimesh.Scene) else g


a, b = load(sys.argv[1]), load(sys.argv[2])
print(
    "recon :", len(a.vertices), "verts", len(a.faces), "faces", np.round(a.extents, 4)
)
print(
    "final :", len(b.vertices), "verts", len(b.faces), "faces", np.round(b.extents, 4)
)
print(
    "watertight",
    a.is_watertight,
    b.is_watertight,
    "| euler",
    a.euler_number,
    b.euler_number,
)
