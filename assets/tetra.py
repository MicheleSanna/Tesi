import pyvista as pv
import tetgen
import meshio

surf = pv.read("maze_ultra_complex.stl")          # esportato da Blender, triangolato
tet = tetgen.TetGen(surf)
tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
grid = tet.grid
print(grid.n_cells, "tetraedri")
grid.plot(show_edges=True)               # controllo visivo
#grid.save("maze4.1.vtk")               # SOFA lo legge con MeshVTKLoader




tets = grid.cells_dict[10]          # 10 = tetraedri
print(len(tets), "tetraedri")

meshio.write(
    "maze_ultra_complex.msh",
    meshio.Mesh(grid.points, [("tetra", tets)]),
    file_format="gmsh22",
    binary=False,
)