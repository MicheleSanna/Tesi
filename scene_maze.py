import sys
import os
import mesh_utilities
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import windows_keyboard
import controller
# =============================================================================
# CONFIGURAZIONE - modifica qui
# =============================================================================
# Mesh tetraedrica del labirinto (le varianti si rigenerano con genera_mesh.py):
#   "fine"    maze4.1.msh         1601 nodi - la piu' accurata, la piu' lenta
#   "graded"  maze4.1_graded.msh   769 nodi - fine vicino al canale, grossolana lontano:
#                                             quasi identica alla fine, ~4x piu' veloce
#   "coarse"  maze4.1_coarse.msh   488 nodi - la piu' veloce, materiale ~13% piu' rigido
#   "ultra"   maze_ultra_complex.msh  31662 nodi - superficie esterna ondulata, stesso canale
MESH_VARIANT = "fine"
MESH_FILES = {"fine": "maze4.1.msh", "graded": "maze4.1_graded.msh", "coarse": "maze4.1_coarse.msh",
              "ultra": "maze_ultra_complex.msh"}
# percorso relativo a questo script, cosi' la scena si carica anche se runSofa viene lanciato
# da un'altra cartella
MESH_FILE = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"), MESH_FILES[MESH_VARIANT])
# maze4.1 e' un cubo normalizzato in [-1, 1]^3 con un canale interno a sezione quadrata
# (lato 0.112 unita' mesh) che entra al centro della faccia -Y e procede verso +Y nel piano z = 0.
# Con MESH_SCALE = 0.05 il cubo e' largo 10 cm e il canale 5.6 mm.
MESH_SCALE = 0.05             # fattore per portare la mesh in metri (0.001 se e' in mm)
UP_AXIS = 2                   # asse verticale: 2 = Z (default di Blender), 1 = Y

YOUNG_MODULUS = 3.0e4         # [Pa] rigidezza del materiale (~silicone morbido)
POISSON_RATIO = 0.45          # quasi incomprimibile, tipico dei tessuti
DENSITY = 1000.0              # [kg/m^3]
FIX_BOTTOM_FRACTION = 0.02    # vincola i nodi nel 2% piu' basso del labirinto (la "base")

# Pressione uniforme e costante sulla superficie ESTERNA del labirinto (non sulle pareti del
# canale), sempre perpendicolare alla superficie anche quando si deforma.
# > 0 comprime, < 0 tira verso l'esterno, 0 = disattivata. Riferimento: E = 30 kPa.
EXTERNAL_PRESSURE = 1000.0    # [Pa]  (1 kPa ~ 7.5 mmHg)

DT = 0.01                     # [s] time step

# Solutore del sistema lineare del labirinto:
#   "direct"    = SparseLDLSolver + GenericConstraintCorrection: esatto, contatti accurati, lento
#   "iterative" = CGLinearSolver + UncoupledConstraintCorrection: approssimato, molto piu' veloce
LINEAR_SOLVER = "iterative"
CG_ITERATIONS = 50            # solo per "iterative": iterazioni massime del gradiente coniugato
USE_MULTITHREADING = False     # versioni parallele (plugin MultiThreading) di FEM e collisioni

# Sonda (scritta in unita' mesh * MESH_SCALE, cosi' resta coerente se cambi scala).
# None = valore automatico proporzionale alla dimensione del labirinto (stampato a terminale)
PROBE_RADIUS = 0.03 * MESH_SCALE   # [m] < meta' larghezza canale (0.056 unita' mesh)!
PROBE_LENGTH = 0.1 * MESH_SCALE    # [m]
PROBE_START = [0.0, -1.06 * MESH_SCALE, 0.0]  # [m] punta appena fuori dall'ingresso sulla faccia -Y
PROBE_START_YAW_DEG = 90.0    # orientamento iniziale: 0 = verso +X, 90 = verso +Y (dentro il canale)

TRANSLATION_STEP = None       # [m] spostamento per passo con tasto premuto (None = meta' raggio)
ROTATION_STEP_DEG = 2.0       # [gradi] rotazione per passo con tasto premuto
FRICTION = 0.1                # coefficiente di attrito sonda/labirinto

# Come viene mossa la sonda: "force" (corpo rigido tirato da una molla) o "kinematic"
PROBE_MODE = "force"
PROBE_MASS = 0.01            # [kg] ~ un'asta d'acciaio di 3 mm x 4 cm
# Molla bersaglio -> sonda. Forza massima = rigidezza * scostamento massimo:
#   traslazione: 20 N/m * 2.5 mm = 0.05 N;  rotazione: 0.004 N*m/rad * 15 deg = ~1 mN*m
# (per confronto, 0.05 N sulla punta di 3 mm schiaccia il silicone da 30 kPa di ~1 mm)
COUPLING_STIFFNESS = 40000.0             # [N/m]
COUPLING_ANGULAR_STIFFNESS = 8   # [N*m/rad]
COUPLING_MAX_OFFSET = 2.5e-3          # [m] distanza massima bersaglio-sonda
COUPLING_MAX_ANGLE_DEG = 15.0         # [gradi] rotazione massima bersaglio-sonda

PRINT_EVERY = 10              # stampa la posa della sonda ogni N passi (0 = mai)


def createScene(root, with_visual=True):
    # --- Mesh e parametri derivati ---------------------------------------------
    points, tets = mesh_utilities.load_tet_mesh(MESH_FILE, MESH_SCALE)
    bb_min, bb_max = points.min(axis=0), points.max(axis=0)
    size = bb_max - bb_min
    diag = float(np.linalg.norm(size))

    up = np.zeros(3)
    up[UP_AXIS] = 1.0

    radius = PROBE_RADIUS or 0.02 * diag
    length = PROBE_LENGTH or 0.4 * diag
    trans_step = TRANSLATION_STEP or 0.5 * radius
    rot_step = math.radians(ROTATION_STEP_DEG)

    if PROBE_START is None:
        start = (bb_min + bb_max) / 2.0
        start[0] = bb_min[0] - 2.0 * radius
    else:
        start = np.asarray(PROBE_START, dtype=float)
    q0 = R.from_rotvec(up * math.radians(PROBE_START_YAW_DEG)).as_quat()
    start_pose = list(start) + list(q0)

    height = points[:, UP_AXIS]
    fixed = np.where(height <= bb_min[UP_AXIS] + FIX_BOTTOM_FRACTION * size[UP_AXIS])[0]
    if len(fixed) == 0:
        raise RuntimeError("Nessun nodo da vincolare: aumenta FIX_BOTTOM_FRACTION o controlla UP_AXIS.")

    mass = DENSITY * mesh_utilities.total_volume(points, tets)

    print("=== Labirinto ===")
    print(f"  nodi: {len(points)}  tetraedri: {len(tets)}  nodi vincolati: {len(fixed)}")
    print(f"  bounding box min: {bb_min}  max: {bb_max}  (dimensioni {size})")
    print(f"  massa: {mass:.4g} kg")
    print("=== Sonda ===")
    print(f"  raggio: {radius:.4g}  lunghezza: {length:.4g}  passo: {trans_step:.4g}")
    print(f"  punta iniziale: {start}")

    # --- Root: plugin, loop, contatti --------------------------------------------
    plugins = [
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.Collision.Detection.Algorithm",
        "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Geometry",
        "Sofa.Component.Collision.Response.Contact",
        "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.Constraint.Lagrangian.Solver",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.Mapping.Linear",
        "Sofa.Component.Mapping.NonLinear",
        "Sofa.Component.Mass",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Topology.Mapping",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Metis",
    ]
    if USE_MULTITHREADING:
        plugins.append("MultiThreading")
    if with_visual:
        plugins += ["Sofa.Component.Visual",
                    "Sofa.GL.Component.Rendering2D",
                    "Sofa.GL.Component.Rendering3D"]
    root.addObject("RequiredPlugin", name="plugins", pluginName=plugins)

    labels = []
    if with_visual:
        root.addObject("VisualStyle", displayFlags="showVisualModels")
        # testo a schermo con la posa della sonda (aggiornato dal controller a ogni passo)
        for i in range(4 if PROBE_MODE == "force" else 3):
            labels.append(root.addObject("OglLabel", name=f"poseLabel{i}", label="",
                                         x=10, y=10 + 22 * i, fontsize=14,
                                         selectContrastingColor=True))

    root.dt = DT
    root.gravity = list(-9.81 * up)

    root.addObject("FreeMotionAnimationLoop")
    root.addObject("BlockGaussSeidelConstraintSolver", maxIterations=200, tolerance=1e-6)

    par = "Parallel" if USE_MULTITHREADING else ""
    root.addObject("CollisionPipeline")
    root.addObject(par + "BruteForceBroadPhase")
    root.addObject(par + "BVHNarrowPhase")
    root.addObject("MinProximityIntersection",
                   alarmDistance=0.5 * radius, contactDistance=0.1 * radius)
    root.addObject("CollisionResponse",
                   response="FrictionContactConstraint", responseParams=f"mu={FRICTION}")

    # --- Labirinto deformabile ---------------------------------------------------
    maze = root.addChild("Labirinto")
    maze.addObject("EulerImplicitSolver", rayleighStiffness=0.1, rayleighMass=0.1)
    if LINEAR_SOLVER == "direct":
        maze.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")
        maze.addObject("MetisOrderingMethod")  # riordino: fattorizzazione LDL molto piu' veloce
    elif LINEAR_SOLVER == "iterative":
        maze.addObject("CGLinearSolver", iterations=CG_ITERATIONS, tolerance=1e-9, threshold=1e-12)
    else:
        raise ValueError(f"LINEAR_SOLVER sconosciuto: {LINEAR_SOLVER!r}")
    maze.addObject("TetrahedronSetTopologyContainer", name="topo",
                   position=points.tolist(), tetrahedra=tets.tolist())
    maze.addObject("TetrahedronSetTopologyModifier")
    maze.addObject("MechanicalObject", name="dofs", template="Vec3d", position=points.tolist())
    maze.addObject("UniformMass", totalMass=mass)
    maze.addObject(par + "TetrahedronFEMForceField", method="large",
                   youngModulus=YOUNG_MODULUS, poissonRatio=POISSON_RATIO)
    maze.addObject("FixedProjectiveConstraint", indices=fixed.tolist())
    if LINEAR_SOLVER == "direct":
        maze.addObject("GenericConstraintCorrection")
    else:
        # ogni nodo viene trattato come indipendente dagli altri, con una cedevolezza
        # (spostamento per unita' di forza) stimata come 1 / (E * lunghezza media degli spigoli)
        maze.addObject("UncoupledConstraintCorrection",
                       defaultCompliance=1.0 / (YOUNG_MODULUS * mesh_utilities.mean_edge_length(points, tets)))

    # superficie estratta dai tetraedri: usata per collisione e grafica
    surf = maze.addChild("Superficie")
    surf.addObject("TriangleSetTopologyContainer", name="topo")
    surf.addObject("TriangleSetTopologyModifier")
    surf.addObject("Tetra2TriangleTopologicalMapping", input="@../topo", output="@topo")
    surf.addObject("TriangleCollisionModel")
    surf.addObject("LineCollisionModel")
    surf.addObject("PointCollisionModel")
    if with_visual:
        surf.addObject("OglModel", name="ogl", color=[0.85, 0.35, 0.35, 0.35])  # semitrasparente
        surf.addObject("IdentityMapping", input="@../dofs", output="@ogl")

    # pressione uniforme sulla superficie esterna: i triangoli usano direttamente gli indici dei
    # nodi del labirinto, quindi le forze agiscono sui dofs del nodo padre senza mapping
    if EXTERNAL_PRESSURE:
        outer, n_boundary = mesh_utilities.outer_surface_triangles(points, tets)
        a, b, c = (points[outer[:, i]] for i in range(3))
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
        print(f"  pressione esterna: {EXTERNAL_PRESSURE:g} Pa su {len(outer)} triangoli esterni "
              f"(su {n_boundary} di superficie totale), area {area * 1e4:.1f} cm^2")
        pressure = maze.addChild("Pressione")
        pressure.addObject("TriangleSetTopologyContainer", name="topo",
                           position=points.tolist(), triangles=outer.tolist())
        # useTangentStiffness=False: la matrice resta simmetrica, come richiesto da CG e LDL
        pressure.addObject("SurfacePressureForceField", pressure=EXTERNAL_PRESSURE,
                           useTangentStiffness=False)

    # --- Sonda ----------------------------------------------------------------------
    if PROBE_MODE not in ("force", "kinematic"):
        raise ValueError(f"PROBE_MODE sconosciuto: {PROBE_MODE!r}")
    dynamic = PROBE_MODE == "force"

    # bersaglio comandato da tastiera/RL (solo "force"): non ha solutore, si muove solo via codice
    target_dofs = None
    if dynamic:
        target = root.addChild("Bersaglio")
        target_dofs = target.addObject("MechanicalObject", name="dofs", template="Rigid3d",
                                       position=[start_pose], showObject=with_visual,
                                       showObjectScale=1.5 * radius)

    probe = root.addChild("Sonda")
    if with_visual:
        probe.addObject("VisualStyle", displayFlags="showCollisionModels showBehaviorModels"
                        + (" showForceFields" if dynamic else ""))
    if dynamic:
        # corpo rigido con la sua dinamica; per un sistema di 6 incognite il solutore diretto
        # e la correzione esatta dei contatti costano praticamente nulla
        probe.addObject("EulerImplicitSolver", rayleighStiffness=0.0, rayleighMass=0.0)
        probe.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixd")
    probe_dofs = probe.addObject("MechanicalObject", name="dofs", template="Rigid3d",
                                 position=[start_pose], showObject=True,
                                 showObjectScale=2.0 * radius)
    if dynamic:
        # asta di raggio r e lunghezza L ruotata attorno alla punta (dove sta il nodo rigido):
        # inerzia / massa = r^2/2 attorno all'asse, L^2/3 + r^2/4 attorno agli assi trasversali
        i_axis = radius ** 2 / 2.0
        i_side = length ** 2 / 3.0 + radius ** 2 / 4.0
        probe.addObject("UniformMass", vertexMass=f"{PROBE_MASS} 1 {i_axis} 0 0 0 {i_side} 0 0 0 {i_side}")
        # la sonda e' sorretta dall'operatore: si compensa il suo peso
        probe.addObject("ConstantForceField", indices=[0],
                        forces=[list(PROBE_MASS * 9.81 * up) + [0.0, 0.0, 0.0]])
        probe.addObject("RestShapeSpringsForceField", name="molla", points=[0],
                        external_rest_shape="@../Bersaglio/dofs", external_points=[0],
                        stiffness=[COUPLING_STIFFNESS], angularStiffness=[COUPLING_ANGULAR_STIFFNESS],
                        drawSpring=with_visual)
        probe.addObject("GenericConstraintCorrection")

    # catena di sfere lungo l'asse locale X (punta nell'origine, corpo verso -X)
    n_spheres = max(2, int(math.ceil(length / radius)) + 1)
    local_pts = [[-i * length / (n_spheres - 1), 0.0, 0.0] for i in range(n_spheres)]

    col = probe.addChild("Collisione")
    col_dofs = col.addObject("MechanicalObject", name="dofs", template="Vec3d", position=local_pts)
    if dynamic:
        col.addObject("SphereCollisionModel", radius=radius)
    else:
        col.addObject("SphereCollisionModel", radius=radius, moving=True, simulated=False)
    col.addObject("RigidMapping", input="@../dofs", output="@dofs", globalToLocalCoords=False)

    coupling = None
    if dynamic:
        coupling = dict(stiffness=COUPLING_STIFFNESS, angular_stiffness=COUPLING_ANGULAR_STIFFNESS,
                        max_offset=COUPLING_MAX_OFFSET, max_angle=math.radians(COUPLING_MAX_ANGLE_DEG))
        print(f"  modalita' force: forza max {COUPLING_STIFFNESS * COUPLING_MAX_OFFSET * 1000:.1f} mN, "
              f"coppia max {coupling['angular_stiffness'] * coupling['max_angle'] * 1000:.2f} mN*m")

    # tastiera solo con l'interfaccia grafica (in headless / RL la sonda si comanda con set_command)
    keyboard = windows_keyboard.WindowsKeyboard() if with_visual and sys.platform == "win32" else None
    root.addObject(controller.ProbeController(probe_dofs, up, trans_step, rot_step,
                                   target_dofs=target_dofs, coupling=coupling,
                                   col_dofs=None if dynamic else col_dofs, col_local=local_pts,
                                   labels=labels, print_every=PRINT_EVERY, keyboard=keyboard,
                                   name="ProbeController"))
    return root