import os
import numpy as np
import pyvista as pv
import tetgen
import meshio


def fix_degenerate(surf, tol=1e-6):
    """Rimuove i triangoli ad area nulla (T-junction: un vertice giace sul lato
    opposto), che TetGen segnala come auto-intersezioni. Il triangolo degenere
    viene eliminato e il vicino che condivide il lato lungo viene spezzato sul
    vertice intermedio: la geometria resta identica."""
    surf = surf.clean()
    P = surf.points.astype(float)
    F = surf.faces.reshape(-1, 4)[:, 1:].tolist()
    while True:
        T = np.array(F)
        a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
        bad = np.where(area < tol)[0]
        if len(bad) == 0:
            break
        k = bad[0]
        tri = F[k]
        # vertice intermedio = opposto al lato più lungo
        L = [np.linalg.norm(P[tri[(i + 1) % 3]] - P[tri[(i + 2) % 3]]) for i in range(3)]
        i = int(np.argmax(L))
        m, u, v = tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]
        # triangolo vicino che condivide il lato lungo (u, v)
        nb = [j for j, f in enumerate(F) if j != k and u in f and v in f][0]
        f = F[nb]
        # spezza il vicino mantenendo l'orientamento
        t1 = [m if w == v else w for w in f]
        t2 = [m if w == u else w for w in f]
        for j in sorted([k, nb], reverse=True):
            F.pop(j)
        F += [t1, t2]
        print(f"rimosso triangolo degenere {tri}, spezzato vicino {f}")
    faces = np.hstack([np.full((len(F), 1), 3), np.array(F)]).ravel()
    return pv.PolyData(P, faces)


def remesh_surface(surf, voxel=0.04, n_clusters=10000):
    """Rifà la superficie con triangoli uniformi, eliminando i triangoli lunghi
    e sottili e le zeppe lasciate dai booleani di Blender.
    1) campo di distanza su griglia voxel + marching cubes (le zeppe più sottili
       di `voxel` spariscono)  2) smoothing Taubin per togliere la scalettatura
    3) clustering ACVD (~2*n_clusters triangoli quasi equilateri)
    4) pymeshfix chiude gli eventuali buchi lasciati dal clustering."""
    import pyacvd
    import pymeshfix
    b = np.array(surf.bounds).reshape(3, 2)
    pad = 3 * voxel
    dims = np.ceil((b[:, 1] - b[:, 0] + 2 * pad) / voxel).astype(int) + 1
    img = pv.ImageData(dimensions=dims, spacing=(voxel,) * 3, origin=b[:, 0] - pad)
    img = img.compute_implicit_distance(surf)
    iso = img.contour([0.0], scalars="implicit_distance").triangulate().clean()
    iso = iso.smooth_taubin(n_iter=20)
    cl = pyacvd.Clustering(iso)
    cl.subdivide(2)
    cl.cluster(n_clusters)
    mf = pymeshfix.MeshFix(cl.create_mesh().clean())
    mf.repair()
    out = mf.mesh
    print(f"remesh: {surf.n_cells} -> {out.n_cells} triangoli, volume {surf.volume:.3f} -> {out.volume:.3f}")
    return out


HERE = os.path.dirname(os.path.abspath(__file__))    # cartella assets/

if __name__ == "__main__":
    surf = pv.read(os.path.join(HERE, "polmoni3.0.stl"))  # esportato da Blender, triangolato
    surf = fix_degenerate(surf)
    # rimesh necessaria: senza, TetGen va in stack overflow sui triangoli lunghi e
    # sottili della trachea. Controllare sempre che la topologia non cambi (su
    # polmoni2.0 apriva un buco nella giunzione centrale, pareti di ~0.01)
    REMESH = True
    if REMESH:
        surf = remesh_surface(surf)
    tet = tetgen.TetGen(surf)
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    grid = tet.grid
    print(grid.n_cells, "tetraedri")
    grid.plot(show_edges=True)               # controllo visivo
    #grid.save("maze4.1.vtk")               # SOFA lo legge con MeshVTKLoader




    tets = grid.cells_dict[10]          # 10 = tetraedri
    print(len(tets), "tetraedri")

    meshio.write(
        os.path.join(HERE, "polmoni3.0.msh"),
        meshio.Mesh(grid.points, [("tetra", tets)]),
        file_format="gmsh22",
        binary=False,
    )
