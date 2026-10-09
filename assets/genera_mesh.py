"""
Genera due varianti della mesh tetraedrica del labirinto a partire da maze4.1.stl
(la mesh originale maze4.1.msh, prodotta da tetra.py, non viene toccata):

  maze4.1_coarse.msh  grossolana ovunque (tetgen): la piu' veloce, ma deformazioni meno
                      dettagliate e materiale un po' piu' rigido del reale
  maze4.1_graded.msh  graduata (Gmsh): fine vicino al canale, dove la sonda tocca,
                      grossolana lontano, dove la deformazione conta poco

La graduata usa Gmsh perche' in tetgen 0.8.4 l'opzione per le dimensioni variabili
(mesh di sfondo, parametro bgmesh) manda in crash il processo.

Con l'argomento "polmoni" genera invece polmoni3.0_graded.msh da polmoni3.0.stl
(polmoni3.0.msh, prodotta da tetra.py, non viene toccata): fine vicino alle pareti
di trachea e bronchi, grossolana nel resto dei polmoni. Con "polmoni_coarse" genera
polmoni3.0_coarse.msh, stessa tecnica con elementi piu' grandi vicino ai canali (una
mesh grossolana ovunque come quella del labirinto non si puo' fare: le pareti della
trachea e il bordo affilato dei polmoni sono piu' sottili dei tetraedri).

USO
  python genera_mesh.py                   (labirinto)
  python genera_mesh.py polmoni           (polmoni, graduata)
  python genera_mesh.py polmoni_coarse    (polmoni, grossolana)
"""

import math
import os
import sys
import tempfile

import numpy as np
import pyvista as pv
import tetgen
import meshio
import gmsh
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
STL_FILE = os.path.join(HERE, "maze4.1.stl")
ORIGINAL_MESH = os.path.join(HERE, "maze4.1.msh")

# Dimensioni in unita' della mesh (il cubo va da -1 a 1, il canale e' largo 0.112).
# Per confronto, l'originale ha tetraedri di lato ~0.06 vicino al canale e ~0.3 lontano.
GRADED_SIZE_NEAR = 0.06       # lato dei tetraedri sulle pareti del canale
GRADED_BAND = 0.03            # entro questa distanza dal canale si resta a GRADED_SIZE_NEAR
GRADED_GROWTH = 1.5           # quanto cresce il lato per unita' di distanza oltre la banda
GRADED_SIZE_FAR = 0.8         # lato massimo lontano dal canale

# Polmoni (unita' della mesh: trachea alta ~0.21 e larga ~0.45, pareti 0.07-0.13).
# Graduata e grossolana differiscono solo per il lato vicino alle pareti dei canali:
#   0.07 -> graduata, ~14k nodi, canali fedeli (scarto <= 0.18 tranne l'imboccatura)
#   0.1  -> grossolana, ~8k nodi, due fondi di bronco accorciati di ~0.3
#   0.12 -> ~6k nodi, ma i fondi dei bronchi si accorciano fino a ~0.5: scartata
LUNGS_STL = os.path.join(HERE, "polmoni3.0.stl")
LUNGS_SIZE_NEAR = 0.07
LUNGS_COARSE_SIZE_NEAR = 0.1
LUNGS_BAND = 0.03
LUNGS_GROWTH = 1.5            # con crescita 3 Gmsh fallisce (superficie autointersecante)
LUNGS_SIZE_FAR = 0.3          # con 0.4-0.6 il bordo affilato dei polmoni viene tagliato o
                              # le sue due facce si incrociano e Gmsh fallisce


def report_and_save(name, points, tets):
    a, b, c, d = (points[tets[:, i]] for i in range(4))
    vol = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0
    edges = np.stack([b - a, c - a, d - a, c - b, d - b, d - c], axis=1)
    quality = 6 * np.sqrt(2) * np.abs(vol) / np.linalg.norm(edges, axis=2).max(axis=1) ** 3
    meshio.write(os.path.join(HERE, name), meshio.Mesh(points, [("tetra", tets)]),
                 file_format="gmsh22", binary=False)
    print(f"{name}: {len(points)} nodi, {len(tets)} tetraedri, "
          f"volumi negativi: {(vol < 0).sum()}, qualita' minima {quality.min():.3f}")


def coarse_mesh():
    """tetgen con vincoli di qualita' permissivi -> pochi punti aggiunti all'interno."""
    tet = tetgen.TetGen(pv.read(STL_FILE))
    points, tets, _, _ = tet.tetrahedralize(order=1, mindihedral=10, minratio=2.0)
    return points, tets


def graded_mesh():
    """Gmsh con un campo di dimensione che dipende dalla distanza dalle pareti del canale."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.merge(STL_FILE)
        # divide l'STL in superfici piane e le rende rimeshabili, poi chiude il volume
        gmsh.model.mesh.classifySurfaces(20 * math.pi / 180, True, True, math.pi)
        gmsh.model.mesh.createGeometry()
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop(surfaces)])
        gmsh.model.geo.synchronize()

        # pareti del canale = superfici che non giacciono su una faccia esterna del cubo
        channel = []
        for tag in surfaces:
            box = np.array(gmsh.model.getBoundingBox(2, tag))
            lo, hi = box[:3], box[3:]
            if not any(abs(lo[i] - hi[i]) < 1e-6 and abs(abs(lo[i]) - 1.0) < 1e-6 for i in range(3)):
                channel.append(tag)

        field = gmsh.model.mesh.field
        field.add("Distance", 1)
        field.setNumbers(1, "SurfacesList", channel)
        field.setNumber(1, "Sampling", 40)
        field.add("Threshold", 2)
        field.setNumber(2, "InField", 1)
        field.setNumber(2, "SizeMin", GRADED_SIZE_NEAR)
        field.setNumber(2, "SizeMax", GRADED_SIZE_FAR)
        field.setNumber(2, "DistMin", GRADED_BAND)
        field.setNumber(2, "DistMax",
                        GRADED_BAND + (GRADED_SIZE_FAR - GRADED_SIZE_NEAR) / GRADED_GROWTH)
        field.setAsBackgroundMesh(2)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        return generate_tets()
    finally:
        gmsh.finalize()


def generate_tets():
    """Genera e ottimizza la mesh di volume del modello Gmsh corrente e restituisce
    (punti, tetraedri), eliminando i nodi non usati dai tetraedri."""
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    tags, coords, _ = gmsh.model.mesh.getNodes()
    points = coords.reshape(-1, 3)
    index = {int(t): i for i, t in enumerate(tags)}
    types, _, nodes = gmsh.model.mesh.getElements(3)
    tets = np.vstack([np.array(n, dtype=np.int64).reshape(-1, 4)
                      for t, n in zip(types, nodes) if t == 4])
    tets = np.vectorize(index.get)(tets)

    used = np.unique(tets)
    remap = -np.ones(len(points), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return points[used], remap[tets]


def cavity_faces(surf):
    """Facce che delimitano i canali interni: il raggio lungo la normale uscente
    incontra di nuovo il modello (sulle facce esterne invece esce nel vuoto)."""
    surf = surf.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
    centers = surf.cell_centers().points
    normals = surf["Normals"]
    diag = np.linalg.norm(np.ptp(surf.points, axis=0))
    hit = [len(surf.ray_trace(c + n * 1e-4, c + n * diag, first_point=True)[0]) > 0
           for c, n in zip(centers, normals)]
    return surf.extract_cells(np.where(hit)[0]).extract_surface()


def graded_lungs_mesh(size_near):
    """Gmsh con dimensione che dipende dalla distanza dalle pareti dei canali (trachea e
    bronchi): size_near sulle pareti, fino a LUNGS_SIZE_FAR lontano. La superficie di
    partenza e' quella rimeshata da tetra.py: l'STL originale ha triangoli degeneri che
    Gmsh non riesce a riparametrizzare."""
    from tetra import fix_degenerate, remesh_surface

    surf = remesh_surface(fix_degenerate(pv.read(LUNGS_STL)))
    # triangoli orientati tutti verso l'esterno: con l'orientamento lasciato da
    # remesh_surface Gmsh puo' fallire (superficie autointersecante)
    surf = surf.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
    cavity = cavity_faces(surf)
    tree = cKDTree(cavity.subdivide(2, "linear").points)   # campioni fitti sulle pareti
    print(f"pareti dei canali: {cavity.n_cells} triangoli su {surf.n_cells}")

    dist_max = LUNGS_BAND + (LUNGS_SIZE_FAR - size_near) / LUNGS_GROWTH

    def size(dim, tag, x, y, z, lc):
        d = tree.query((x, y, z))[0]
        return float(np.clip(size_near + (d - LUNGS_BAND) * LUNGS_GROWTH,
                             size_near, LUNGS_SIZE_FAR)) if d < dist_max else LUNGS_SIZE_FAR

    with tempfile.TemporaryDirectory() as tmp:
        stl = os.path.join(tmp, "superficie.stl")
        surf.save(stl)
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.merge(stl)
            # con angoli piu' piccoli Gmsh non riesce a parametrizzare le superfici curve
            gmsh.model.mesh.classifySurfaces(90 * math.pi / 180, True, True, math.pi)
            gmsh.model.mesh.createGeometry()
            surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
            gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop(surfaces)])
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.setSizeCallback(size)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            return generate_tets()
        finally:
            gmsh.finalize()


def main():
    lungs = {"polmoni": ("polmoni3.0_graded.msh", LUNGS_SIZE_NEAR),
             "polmoni_coarse": ("polmoni3.0_coarse.msh", LUNGS_COARSE_SIZE_NEAR)}
    if len(sys.argv) > 1 and sys.argv[1] in lungs:
        name, size_near = lungs[sys.argv[1]]
        report_and_save(name, *graded_lungs_mesh(size_near))
        orig = meshio.read(os.path.join(HERE, "polmoni3.0.msh"))
        print(f"polmoni3.0.msh (uniforme): {len(orig.points)} nodi, "
              f"{len(orig.cells_dict['tetra'])} tetraedri")
        return

    report_and_save("maze4.1_coarse.msh", *coarse_mesh())
    report_and_save("maze4.1_graded.msh", *graded_mesh())

    orig = meshio.read(ORIGINAL_MESH)
    print(f"maze4.1.msh (originale): {len(orig.points)} nodi, "
          f"{len(orig.cells_dict['tetra'])} tetraedri")


if __name__ == "__main__":
    main()
