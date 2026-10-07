"""
Genera due varianti della mesh tetraedrica del labirinto a partire da maze4.1.stl
(la mesh originale maze4.1.msh, prodotta da tetra.py, non viene toccata):

  maze4.1_coarse.msh  grossolana ovunque (tetgen): la piu' veloce, ma deformazioni meno
                      dettagliate e materiale un po' piu' rigido del reale
  maze4.1_graded.msh  graduata (Gmsh): fine vicino al canale, dove la sonda tocca,
                      grossolana lontano, dove la deformazione conta poco

La graduata usa Gmsh perche' in tetgen 0.8.4 l'opzione per le dimensioni variabili
(mesh di sfondo, parametro bgmesh) manda in crash il processo.

USO
  python genera_mesh.py
"""

import math
import os

import numpy as np
import pyvista as pv
import tetgen
import meshio
import gmsh

HERE = os.path.dirname(os.path.abspath(__file__))
STL_FILE = os.path.join(HERE, "maze4.1.stl")
ORIGINAL_MESH = os.path.join(HERE, "maze4.1.msh")

# Dimensioni in unita' della mesh (il cubo va da -1 a 1, il canale e' largo 0.112).
# Per confronto, l'originale ha tetraedri di lato ~0.06 vicino al canale e ~0.3 lontano.
GRADED_SIZE_NEAR = 0.06       # lato dei tetraedri sulle pareti del canale
GRADED_BAND = 0.03            # entro questa distanza dal canale si resta a GRADED_SIZE_NEAR
GRADED_GROWTH = 1.5           # quanto cresce il lato per unita' di distanza oltre la banda
GRADED_SIZE_FAR = 0.8         # lato massimo lontano dal canale


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

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")

        tags, coords, _ = gmsh.model.mesh.getNodes()
        points = coords.reshape(-1, 3)
        index = {int(t): i for i, t in enumerate(tags)}
        types, _, nodes = gmsh.model.mesh.getElements(3)
        tets = np.vstack([np.array(n, dtype=np.int64).reshape(-1, 4)
                          for t, n in zip(types, nodes) if t == 4])
        tets = np.vectorize(index.get)(tets)
    finally:
        gmsh.finalize()

    # elimina i nodi non usati dai tetraedri
    used = np.unique(tets)
    remap = -np.ones(len(points), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return points[used], remap[tets]


def main():
    report_and_save("maze4.1_coarse.msh", *coarse_mesh())
    report_and_save("maze4.1_graded.msh", *graded_mesh())

    orig = meshio.read(ORIGINAL_MESH)
    print(f"maze4.1.msh (originale): {len(orig.points)} nodi, "
          f"{len(orig.cells_dict['tetra'])} tetraedri")


if __name__ == "__main__":
    main()
