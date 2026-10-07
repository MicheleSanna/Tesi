
import meshio
import numpy as np
from scipy.spatial import ConvexHull

def load_tet_mesh(path, scale):
    """Legge la mesh e restituisce (punti Nx3, tetraedri Mx4), eliminando i nodi non usati."""
    mesh = meshio.read(path)
    blocks = [c.data for c in mesh.cells if c.type == "tetra"]
    if not blocks:
        raise RuntimeError(f"'{path}' non contiene tetraedri: rigenera la mesh volumetrica.")
    tets = np.vstack(blocks)

    used = np.unique(tets)
    remap = -np.ones(len(mesh.points), dtype=np.int64)
    remap[used] = np.arange(len(used))
    points = mesh.points[used, :3].astype(float) * scale
    return points, remap[tets]


def mean_edge_length(points, tets):
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return float(np.mean([np.linalg.norm(points[tets[:, i]] - points[tets[:, j]], axis=1)
                          for i, j in pairs]))


def outer_surface_triangles(points, tets, tol=1e-4):
    """Triangoli della superficie esterna, con la normale (regola della mano destra) rivolta
    verso l'interno del materiale.

    Superficie = facce dei tetraedri che appartengono a un solo tetraedro. Tra queste, quelle
    esterne sono quelle che giacciono sull'involucro convesso della mesh (le facce del cubo);
    le pareti del canale stanno all'interno e vengono escluse. tol e' relativa alla diagonale."""
    local = [(1, 2, 3, 0), (0, 2, 3, 1), (0, 1, 3, 2), (0, 1, 2, 3)]   # (faccia, vertice opposto)
    faces = np.vstack([tets[:, [i, j, k]] for i, j, k, _ in local])
    opposite = np.concatenate([tets[:, o] for _, _, _, o in local])
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                 return_index=True, return_counts=True)
    boundary = first[counts == 1]
    faces, opposite = faces[boundary].copy(), opposite[boundary]

    a, b, c = (points[faces[:, i]] for i in range(3))
    normals = np.cross(b - a, c - a)
    outward = np.einsum("ij,ij->i", normals, points[opposite] - a) < 0
    faces[outward] = faces[outward][:, [0, 2, 1]]

    hull = ConvexHull(points)
    centroids = (a + b + c) / 3.0
    dist = -(centroids @ hull.equations[:, :3].T + hull.equations[:, 3]).max(axis=1)
    diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    return faces[dist < tol * diag], len(faces)


def total_volume(points, tets):
    a, b, c, d = (points[tets[:, i]] for i in range(4))
    return float(np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)).sum() / 6.0)
