
import meshio
import numpy as np
from scipy.spatial import ConvexHull

# nomi con cui meshio espone le etichette (physical group / reference) dei vari formati
_TAG_KEYS = ("gmsh:physical", "medit:ref", "CellEntityIds")


def load_tet_mesh(path, scale):
    """Legge la mesh e restituisce (punti Nx3, tetraedri Mx4), eliminando i nodi non usati."""
    mesh = load_labeled_tet_mesh(path, scale)
    return mesh.points, mesh.tets


class LabeledMesh:
    """Mesh tetraedrica con le etichette (eventuali) di tetraedri e triangoli.

    points (N, 3), tets (M, 4); tet_tags (M,) o None; triangles (T, 3) e tri_tags (T,) sono i
    triangoli salvati nel file (le superfici marcate dal generatore di mesh), o None."""

    def __init__(self, points, tets, tet_tags=None, triangles=None, tri_tags=None):
        self.points, self.tets, self.tet_tags = points, tets, tet_tags
        self.triangles, self.tri_tags = triangles, tri_tags


def load_labeled_tet_mesh(path, scale):
    """Come load_tet_mesh, ma restituisce un LabeledMesh con anche le etichette dei tetraedri
    (ad esempio il tessuto) e i triangoli marcati, se il file li contiene."""
    mesh = meshio.read(path)

    def blocks_of(kind):
        """(blocchi di connettivita', blocchi di etichette o None) delle celle di tipo kind."""
        cells = [(i, c.data) for i, c in enumerate(mesh.cells) if c.type == kind]
        key = next((k for k in _TAG_KEYS if k in mesh.cell_data), None)
        tags = [mesh.cell_data[key][i] for i, _ in cells] if key else None
        return [d for _, d in cells], tags

    blocks, tag_blocks = blocks_of("tetra")
    if not blocks:
        raise RuntimeError(f"'{path}' non contiene tetraedri: rigenera la mesh volumetrica.")
    tets = np.vstack(blocks)
    tet_tags = np.concatenate(tag_blocks).astype(int) if tag_blocks else None

    used = np.unique(tets)
    remap = -np.ones(len(mesh.points), dtype=np.int64)
    remap[used] = np.arange(len(used))
    points = mesh.points[used, :3].astype(float) * scale

    triangles = tri_tags = None
    tri_blocks, tri_tag_blocks = blocks_of("triangle")
    if tri_blocks:
        triangles = remap[np.vstack(tri_blocks)]
        tri_tags = np.concatenate(tri_tag_blocks).astype(int) if tri_tag_blocks else None
        keep = (triangles >= 0).all(axis=1)       # triangoli su nodi non usati dai tetraedri
        triangles = triangles[keep]
        tri_tags = tri_tags[keep] if tri_tags is not None else None
    return LabeledMesh(points, remap[tets], tet_tags, triangles, tri_tags)


def trachea_axis(points, tets, tet_tags, trachea_tag, up, length):
    """Asse della trachea vicino all'imboccatura, ricavato dai tetraedri della sua parete.

    Si taglia la parete a fette perpendicolari a 'up' nei primi 'length' metri sotto il punto piu'
    alto della trachea: il baricentro di ogni fetta (un anello di parete) sta sull'asse del lume.
    La retta che meglio approssima i baricentri e' l'asse. Restituisce (punto dell'asse nella
    fetta piu' alta, direzione verso l'interno del corpo, cioe' verso la carena)."""
    up = np.asarray(up, dtype=float) / np.linalg.norm(up)
    wall = tets[tet_tags == trachea_tag]
    if len(wall) == 0:
        raise RuntimeError(f"nessun tetraedro con etichetta {trachea_tag} (trachea)")
    centroids = points[wall].mean(axis=1)
    height = centroids @ up
    top = height.max()
    n_slices = 12
    edges = np.linspace(top - length, top, n_slices + 1)
    centers = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (height >= lo) & (height < hi)
        if sel.sum() > 20:
            centers.append(centroids[sel].mean(axis=0))
    centers = np.array(centers)
    if len(centers) < 3:
        raise RuntimeError("trachea troppo corta per stimarne l'asse: riduci 'length'")
    mean = centers.mean(axis=0)
    direction = np.linalg.svd(centers - mean)[2][0]         # direzione principale dei baricentri
    if direction @ up > 0:
        direction = -direction                              # verso il basso, dentro il corpo
    # punto dell'asse vicino alla fetta piu' alta (le fette sono in ordine di altezza crescente)
    origin = mean + direction * ((centers[-1] - mean) @ direction)
    return origin, direction


def frame_rotation(axis_from, up_from, axis_to, up_to):
    """Rotazione (scipy Rotation) che porta la direzione axis_from su axis_to e, tra le rotazioni
    che lo fanno, porta up_from (resa perpendicolare ad axis_from) su up_to. E' una rotazione
    propria (niente specchiature, che invertirebbero i tetraedri)."""
    from scipy.spatial.transform import Rotation as R

    def frame(axis, up):
        a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        u = np.asarray(up, dtype=float) - (np.asarray(up, dtype=float) @ a) * a
        if np.linalg.norm(u) < 1e-9:
            raise ValueError("la direzione 'up' e' parallela all'asse")
        u /= np.linalg.norm(u)
        return np.column_stack([a, u, np.cross(a, u)])     # colonne: asse, alto, terzo asse

    return R.from_matrix(frame(axis_to, up_to) @ frame(axis_from, up_from).T)


def nodal_values(n_points, tets, element_values):
    """Media, su ogni nodo, dei valori dei tetraedri che lo contengono (n_points,)."""
    total = np.zeros(n_points)
    count = np.zeros(n_points)
    for i in range(4):
        np.add.at(total, tets[:, i], element_values)
        np.add.at(count, tets[:, i], 1.0)
    return total / np.maximum(count, 1.0)


def nodal_edge_length(points, tets):
    """Lunghezza media degli spigoli dei tetraedri attorno a ogni nodo (n_points,)."""
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    total = np.zeros(len(points))
    count = np.zeros(len(points))
    for i, j in pairs:
        length = np.linalg.norm(points[tets[:, i]] - points[tets[:, j]], axis=1)
        for k in (i, j):
            np.add.at(total, tets[:, k], length)
            np.add.at(count, tets[:, k], 1.0)
    return total / np.maximum(count, 1.0)


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


def boundary_triangles(tets, points):
    """Tutti i triangoli di superficie (esterna e pareti del canale), orientati con la normale
    (regola della mano destra) che ESCE dal materiale: sulle pareti del canale punta quindi verso
    il lume, cioe' verso la sonda. Usati dalla telecamera endoscopica (endoscope_camera.py).

    Una faccia e' di superficie se appartiene a un solo tetraedro; il suo orientamento si
    sistema guardando il vertice opposto del tetraedro, che sta dalla parte del materiale."""
    local = [(1, 2, 3, 0), (0, 2, 3, 1), (0, 1, 3, 2), (0, 1, 2, 3)]   # (faccia, vertice opposto)
    faces = np.vstack([tets[:, [i, j, k]] for i, j, k, _ in local])
    opposite = np.concatenate([tets[:, o] for _, _, _, o in local])
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                 return_index=True, return_counts=True)
    boundary = first[counts == 1]
    faces, opposite = faces[boundary].copy(), opposite[boundary]

    a, b, c = (points[faces[:, i]] for i in range(3))
    inward = np.einsum("ij,ij->i", np.cross(b - a, c - a), points[opposite] - a) > 0
    faces[inward] = faces[inward][:, [0, 2, 1]]   # normale verso il materiale: si inverte
    return faces


def total_volume(points, tets):
    a, b, c, d = (points[tets[:, i]] for i in range(4))
    return float(np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)).sum() / 6.0)
