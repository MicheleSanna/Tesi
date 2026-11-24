import sys
import os
import mesh_utilities
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import windows_keyboard
import linux_keyboard
import controller
import bronchoscope
import contact_monitor
# =============================================================================
# CONFIGURAZIONE - modifica qui
# =============================================================================
# Mesh tetraedrica del labirinto (le varianti si rigenerano con genera_mesh.py):
#   "fine"    maze4.1.msh         1601 nodi - la piu' accurata, la piu' lenta
#   "graded"  maze4.1_graded.msh   769 nodi - fine vicino al canale, grossolana lontano:
#                                             quasi identica alla fine, ~4x piu' veloce
#   "coarse"  maze4.1_coarse.msh   488 nodi - la piu' veloce, materiale ~13% piu' rigido
#   "ultra"   maze_ultra_complex.msh  31662 nodi - superficie esterna ondulata, stesso canale
# Polmoni (tetra.py e genera_mesh.py polmoni):
#   "polmoni"         polmoni3.0.msh         20773 nodi - uniforme
#   "polmoni_graded"  polmoni3.0_graded.msh  13982 nodi - fine vicino a trachea e bronchi
#   "polmoni_coarse"  polmoni3.0_coarse.msh   7954 nodi - come graded ma piu' grossolana,
#                                             due fondi di bronco accorciati di ~0.3 unita'
MESH_VARIANT = "polmoni_graded"
MESH_FILES = {"fine": "maze4.1.msh", "graded": "maze4.1_graded.msh", "coarse": "maze4.1_coarse.msh",
              "ultra": "maze_ultra_complex.msh", "polmoni": "polmoni3.0.msh",
              "polmoni_graded": "polmoni3.0_graded.msh", "polmoni_coarse": "polmoni3.0_coarse.msh"}
# percorso relativo a questo script, cosi' la scena si carica anche se runSofa viene lanciato
# da un'altra cartella
MESH_FILE = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"), MESH_FILES[MESH_VARIANT])
UP_AXIS = 2                   # asse verticale: 2 = Z (default di Blender), 1 = Y

# Scala, rotazione e posa iniziale del broncoscopio dipendono dalla mesh.
# MESH_SCALE: fattore per portare la mesh in metri (0.001 se e' in mm)
# MESH_ROTATION_DEG: rotazione della mesh attorno all'asse verticale, applicata al caricamento
# Misure dello strumento scritte in unita' mesh * MESH_SCALE, cosi' restano coerenti se cambi scala.
# PROBE_RADIUS = raggio esterno del broncoscopio; PROBE_START = posizione iniziale della punta,
#   appena fuori dall'imboccatura (None = automatico, proporzionale alla dimensione della mesh)
# SCOPE_LENGTH = lunghezza inseribile dello strumento: deve bastare a raggiungere il fondo del
#   canale / dei bronchi (all'inizio e' tutta fuori, dentro il tubo introduttore)
if os.path.basename(MESH_FILE).startswith("maze"):
    # maze4.1 e' un cubo normalizzato in [-1, 1]^3 con un canale interno a sezione quadrata
    # (lato 0.112 unita' mesh) che entra al centro della faccia -Y e procede verso +Y nel piano z = 0.
    # Con MESH_SCALE = 0.05 il cubo e' largo 10 cm e il canale 5.6 mm.
    MESH_SCALE = 0.05
    MESH_ROTATION_DEG = 0.0
    MESH_UPSIDE_DOWN = False
    PROBE_RADIUS = 0.03 * MESH_SCALE   # [m] < meta' larghezza canale (0.056 unita' mesh)!
    PROBE_START = [0.0, -1.06 * MESH_SCALE, 0.0]  # [m] punta appena fuori dall'ingresso sulla faccia -Y
    SCOPE_LENGTH = 0.12                # [m] il canale e' lungo ~8 cm
else:
    # polmoni3.0: la trachea e' un canale a sezione rettangolare lungo +X (prima della rotazione),
    # largo 0.43 e alto 0.215 unita' mesh, con asse in y = 0, z ~ 0.318 e imboccatura a x ~ -4.55.
    # La rotazione di 90 gradi porta l'imboccatura su -Y e la trachea lungo +Y, come il canale
    # del labirinto (stessa convenzione per la sonda e per il reward +Y dell'ambiente RL).
    # Scala: diametro della sonda = meta' del lato minore della trachea, con la sonda della stessa
    # misura di quella del labirinto (diametro 3 mm) -> trachea 6 x 12 mm, polmoni ~25 cm.
    # MESH_UPSIDE_DOWN capovolge il modello (rotazione di 180 gradi attorno all'asse della trachea,
    # non uno specchio, che invertirebbe i tetraedri): la trachea passa da z ~ +0.318 a z ~ -0.318,
    # nella meta' bassa del modello, e i due polmoni si scambiano di lato.
    TRACHEA_SIZE = 0.215               # lato minore del canale della trachea [unita' mesh]
    MESH_SCALE = 4 * 0.0015 / TRACHEA_SIZE
    MESH_ROTATION_DEG = 90.0
    MESH_UPSIDE_DOWN = True
    PROBE_RADIUS = TRACHEA_SIZE / 4 * MESH_SCALE   # [m] diametro = meta' della trachea
    trachea_z = -0.318 if MESH_UPSIDE_DOWN else 0.318
    PROBE_START = [0.0, -4.66 * MESH_SCALE, trachea_z * MESH_SCALE]  # [m] punta appena fuori dall'imboccatura
    SCOPE_LENGTH = 0.25                # [m] trachea ~13 cm + bronchi fino a ~10 cm

YOUNG_MODULUS = 3.0e4         # [Pa] rigidezza del materiale (~silicone morbido)
POISSON_RATIO = 0.45          # quasi incomprimibile, tipico dei tessuti
DENSITY = 1000.0              # [kg/m^3]
FIX_BOTTOM_FRACTION = 0.27   # vincola i nodi nel 2% piu' basso del labirinto (la "base")

# Pressione uniforme e costante sulla superficie ESTERNA del labirinto (non sulle pareti del
# canale), sempre perpendicolare alla superficie anche quando si deforma.
# > 0 comprime, < 0 tira verso l'esterno, 0 = disattivata. Riferimento: E = 30 kPa.
EXTERNAL_PRESSURE = 1000.0    # [Pa]  (1 kPa ~ 7.5 mmHg)

DT = 0.01                     # [s] time step

# Solutore dei vincoli (contatti + cavi), nome del componente SOFA senza "ConstraintSolver":
#   "BlockGaussSeidel" = Gauss-Seidel a blocchi (quello usato con la sonda rigida)
#   "NNCG"             = gradiente coniugato non lineare non liscio: converge molto prima quando
#                        tanti contatti sono accoppiati da uno strumento rigido (lo strumento
#                        incastrato e inflesso nelle vie aeree)
CONSTRAINT_SOLVER = "BlockGaussSeidel"
CONSTRAINT_ITERATIONS = 500
CONSTRAINT_TOLERANCE = 1e-6

# Solutore del sistema lineare del labirinto:
#   "direct"    = SparseLDLSolver + GenericConstraintCorrection: esatto, contatti accurati, lento
#   "iterative" = CGLinearSolver + UncoupledConstraintCorrection: approssimato, molto piu' veloce
LINEAR_SOLVER = "iterative"
CG_ITERATIONS = 50            # solo per "iterative": iterazioni massime del gradiente coniugato
USE_MULTITHREADING = False     # versioni parallele (plugin MultiThreading) di FEM e collisioni

# FEM del labirinto su GPU (plugin SofaCUDA, precisione singola):
#   "auto" = usa CUDA solo se il plugin SofaCUDA e una GPU NVIDIA sono disponibili
#   True / False = forza GPU / CPU
# Si puo' scegliere anche senza modificare il file: SOFA_USE_CUDA=auto|1|0 python main.py
# Con CUDA restano su CPU collisioni, contatti, pressione esterna e sonda; conviene solo con
# mesh grandi (decine di migliaia di nodi). Richiede LINEAR_SOLVER = "iterative".
USE_CUDA =  True #os.environ.get("SOFA_USE_CUDA", "auto")
print("CUDA: ", USE_CUDA)
CUDA_CG_TOLERANCE = 1e-6      # in precisione singola la tolleranza 1e-9 del CG non si raggiunge

# --- Broncoscopio (bronchoscope.py) --------------------------------------------------------
# Raggio, lunghezza e punto di partenza sono impostati sopra in base alla mesh.
# Valori di riferimento: un broncoscopio vero da 5-6 mm ha una sezione di curvatura di 3-4 cm che
# si piega di 180 gradi su e 130 giu'. Qui lo strumento e' scalato come la mesh (diametro 3 mm).
PROBE_START_YAW_DEG = 90.0    # direzione di inserimento: 0 = verso +X, 90 = verso +Y (dentro il canale)

SCOPE_BENDING_LENGTH = 0.015          # [m] sezione di curvatura distale
SCOPE_ELEMENT_LENGTH = 0.005          # [m] lunghezza delle travi dello shaft
SCOPE_BENDING_ELEMENTS = 6            # travi ("vertebre") della sezione di curvatura
# Rigidezza a flessione EI [N*m^2]. Lo shaft deve essere abbastanza rigido da poter essere spinto
# (Euler: un tratto libero di 2 cm regge ~25 N di punta) ma adattarsi alle curve dei bronchi; la
# sezione di curvatura e' 20 volte piu' morbida, cosi' i cavi la piegano senza piegare lo shaft.
# Non va resa molto piu' morbida: il suo carico di punta critico pi^2 EI / (4 L^2) (0.55 N con
# questi valori) deve stare vicino alla spinta massima della mano. Con EI = 2e-5 (0.22 N) la
# punta spinta contro la carena si ripiegava di scatto e la simulazione divergeva.
SHAFT_BENDING_STIFFNESS = 1.0e-3
BENDING_SECTION_STIFFNESS = 5.0e-5
CHANNEL_RADIUS = None                 # [m] canale di lavoro (None = 0.4 * raggio)
SCOPE_DENSITY = 1100.0                # [kg/m^3]
SCOPE_GRAVITY = False                 # peso dello strumento (trascurabile, vedi bronchoscope.py)

# Cavi tendinei (CableConstraint di SoftRobots)
CABLE_WAYS = 4                # 2 = su/giu' (broncoscopio standard), 4 = anche sinistra/destra (robotico)
MAX_BEND_DEG = 180.0          # curvatura con la leva a fondo corsa
MAX_CABLE_FORCE = 20.0        # [N] tensione massima di un cavo (curvatura libera a 180 gradi: ~10 N)
CABLE_SLACK_DEG = 10.0        # gioco della leva attorno al neutro (punta "molle" a leva neutra)

# Mano dell'operatore: molla mano -> base dello strumento. Spinta massima = rigidezza * offset.
HAND_STIFFNESS = 1000.0               # [N/m]
HAND_MAX_PUSH_OFFSET = 1.0e-3         # [m]  -> spinta massima 1 N (in broncoscopia ~1-5 N, qui le
                                      #         vie aeree sono scalate a ~1/3)
HAND_ANGULAR_STIFFNESS = 5.0          # [N*m/rad] alta: la base regge una leva di decine di cm
HAND_MAX_ROLL_OFFSET_DEG = 0.3        # [gradi] -> coppia massima di rotazione ~26 mN*m
MAX_ROLL_DEG = 180.0                  # rotazione massima dello strumento in ciascun verso

# Comandi (tastiera / RL): incrementi per passo con comando = 1
TRANSLATION_STEP = None       # [m] avanzamento per passo (None = meta' raggio)
ROTATION_STEP_DEG = 2.0       # [gradi] rotazione dello strumento per passo
BEND_STEP = 0.02              # frazione della corsa della leva per passo (0.02 -> 50 passi = 0.5 s da 0
                              # a fondo corsa; piu' veloce la punta "frusta", vedi bronchoscope.py)
FRICTION = 0.1                # coefficiente di attrito strumento/pareti (mucosa bagnata: 0.05-0.2)

INTRODUCER_GAP = 1.0e-3       # [m] distanza tra la fine del tubo introduttore e il tessuto

# Contatti strumento-pareti misurati per il reward (contact_monitor.py)
CONTACT_TIP_LENGTH = None     # [m] parte distale che conta come "punta" (None = sezione di curvatura)

PRINT_EVERY = 10              # stampa la posa della sonda ogni N passi (0 = mai)

# --- Endoscopio: telecamera simulata sulla punta della sonda (endoscope_camera.py) -------------
# Con runSofa apre una finestra separata con l'immagine dell'endoscopio, aggiornata a ogni passo.
# Nel training viene creata solo se l'ambiente usa OBS_MODE = "camera" (vedi env.py).
CAMERA_ENABLED = True            # True = telecamera attiva quando c'e' l'interfaccia grafica
CAMERA_RESOLUTION = 64           # [px] lato dell'immagine vista dall'agente RL
CAMERA_FOV_DEG = 120.0           # campo visivo (endoscopi reali: 100-140 gradi), ottica fisheye
CAMERA_LIGHT_RANGE = None        # [m] distanza a cui una parete appare ben illuminata (None = 4 raggi sonda)
CAMERA_TEXTURE_SCALE = None      # [m] dimensione delle maglie dei vasi (None = 2 raggi sonda)
CAMERA_NOISE = 0.01              # rumore del sensore (deviazione standard, luminosita' in [0, 1])
CAMERA_SEED = 0                  # seme della texture dei vasi (cambiarlo = un altro "paziente")
CAMERA_SHOW_WINDOW = True        # finestra OpenCV con anteprima, vista dell'agente e profondita'
CAMERA_PREVIEW_RESOLUTION = 320  # [px] lato dell'anteprima ad alta risoluzione nella finestra
CAMERA_SAVE_DIR = None           # cartella dove salvare le immagini (None = non salvare), es.
                                 # "endoscope_frames"; si usa anche se la finestra non si apre
CAMERA_SAVE_EVERY = 1            # salva un'immagine ogni N passi


def _cuda_device_available():
    """True se il driver NVIDIA vede almeno una GPU (senza dipendere da torch o altro)."""
    import ctypes
    try:
        cuda = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
        count = ctypes.c_int(0)
        return cuda.cuInit(0) == 0 and cuda.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
    except (OSError, AttributeError):
        return False


def resolve_use_cuda():
    """Interpreta USE_CUDA ("auto", True/False o stringhe come "1"/"0") e restituisce un bool."""
    choice = str(USE_CUDA).strip().lower()
    if choice in ("0", "false", "no", "off"):
        return False
    forced = choice in ("1", "true", "yes", "on")
    if not forced and choice != "auto":
        raise ValueError(f"USE_CUDA sconosciuto: {USE_CUDA!r} (usa auto, 1 o 0)")

    if LINEAR_SOLVER != "iterative":
        if forced:
            raise ValueError('USE_CUDA richiede LINEAR_SOLVER = "iterative" '
                             "(SparseLDLSolver non ha una versione CUDA)")
        print(f'[CUDA] disattivato: LINEAR_SOLVER = "{LINEAR_SOLVER}" non e\' supportato su GPU')
        return False

    import Sofa.Core
    import SofaRuntime
    reason = None
    if not _cuda_device_available():
        reason = "nessuna GPU NVIDIA o driver CUDA trovato"
    elif not SofaRuntime.importPlugin("SofaCUDA.Component"):
        reason = "plugin SofaCUDA non disponibile in questa installazione di SOFA"
    elif "CudaVec3f" not in Sofa.Core.ObjectFactory.getComponent("MechanicalObject").templates:
        reason = "il plugin SofaCUDA non registra il template CudaVec3f"
    if reason:
        if forced:
            raise RuntimeError(f"USE_CUDA forzato ma {reason}")
        print(f"[CUDA] disattivato: {reason}")
        return False
    return True


def create_keyboard():
    """Lettore della tastiera adatto al sistema operativo (None se non supportato)."""
    if sys.platform == "win32":
        return windows_keyboard.WindowsKeyboard()
    if sys.platform.startswith("linux"):
        return linux_keyboard.LinuxKeyboard()
    print(f"[tastiera] sistema {sys.platform} non supportato: comandi da tastiera disattivati")
    return None


def scope_params(radius):
    """ScopeParams del broncoscopio a partire dalle impostazioni in cima al file."""
    return bronchoscope.ScopeParams(
        radius=radius, length=SCOPE_LENGTH, bending_length=SCOPE_BENDING_LENGTH,
        shaft_element_length=SCOPE_ELEMENT_LENGTH, bending_elements=SCOPE_BENDING_ELEMENTS,
        shaft_bending_stiffness=SHAFT_BENDING_STIFFNESS,
        bending_section_stiffness=BENDING_SECTION_STIFFNESS,
        channel_radius=CHANNEL_RADIUS if CHANNEL_RADIUS is not None else 0.4 * radius,
        density=SCOPE_DENSITY, gravity=SCOPE_GRAVITY, cable_ways=CABLE_WAYS, max_bend_deg=MAX_BEND_DEG,
        max_cable_force=MAX_CABLE_FORCE, cable_slack_deg=CABLE_SLACK_DEG,
        hand_stiffness=HAND_STIFFNESS, hand_angular_stiffness=HAND_ANGULAR_STIFFNESS,
        max_push_offset=HAND_MAX_PUSH_OFFSET, max_roll_offset_deg=HAND_MAX_ROLL_OFFSET_DEG)


def createScene(root, with_visual=True, with_camera=None):
    """Costruisce la scena: tessuto deformabile (labirinto o polmoni), broncoscopio a cavi con il
    suo tubo introduttore, controller dei comandi, misura dei contatti ed eventuale telecamera.

    with_camera: True / False = aggiungi / non aggiungere la telecamera endoscopica;
    None = solo se CAMERA_ENABLED e c'e' l'interfaccia grafica (il training la chiede con True)."""
    if with_camera is None:
        with_camera = CAMERA_ENABLED and with_visual

    # --- Mesh e parametri derivati ---------------------------------------------
    up = np.zeros(3)
    up[UP_AXIS] = 1.0

    points, tets = mesh_utilities.load_tet_mesh(MESH_FILE, MESH_SCALE)
    if MESH_ROTATION_DEG:
        points = R.from_rotvec(up * math.radians(MESH_ROTATION_DEG)).apply(points)
    if MESH_UPSIDE_DOWN:
        # 180 gradi attorno all'asse Y (quello lungo cui entra la sonda)
        points = R.from_rotvec([0.0, math.pi, 0.0]).apply(points)
    bb_min, bb_max = points.min(axis=0), points.max(axis=0)
    size = bb_max - bb_min
    diag = float(np.linalg.norm(size))

    radius = PROBE_RADIUS or 0.02 * diag
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
    print(f"=== Punta del broncoscopio all'inizio: {start}, avanzamento {trans_step * 1000:.2f} mm/passo ===")

    use_cuda = resolve_use_cuda()
    print(f"=== FEM del labirinto su {'GPU (SofaCUDA, CudaVec3f)' if use_cuda else 'CPU (Vec3d)'} ===")

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
        "Sofa.Component.Topology.Container.Constant",
        "Sofa.Metis",
        "BeamAdapter",      # travi dello strumento (BeamInterpolation, AdaptiveBeamForceFieldAndMass)
        "SoftRobots",       # cavi tendinei (CableConstraint)
    ]
    if USE_MULTITHREADING:
        plugins.append("MultiThreading")
    # SofaCUDA.Component non va in questa lista: lo carica gia' resolve_use_cuda(), e un
    # RequiredPlugin su un plugin gia' caricato stampa un avviso inutile
    if with_visual:
        plugins += ["Sofa.Component.Visual",
                    "Sofa.GL.Component.Rendering2D",
                    "Sofa.GL.Component.Rendering3D"]
    root.addObject("RequiredPlugin", name="plugins", pluginName=plugins)

    labels = []
    if with_visual:
        root.addObject("VisualStyle", displayFlags="showVisualModels")
        # testo a schermo: 4 righe sullo strumento (BronchoscopeController) + 1 sui contatti
        # (ContactMonitor), aggiornate a ogni passo
        for i in range(5):
            labels.append(root.addObject("OglLabel", name=f"poseLabel{i}", label="",
                                         x=10, y=10 + 22 * i, fontsize=14,
                                         selectContrastingColor=True))

    root.dt = DT
    root.gravity = list(-9.81 * up)

    root.addObject("FreeMotionAnimationLoop")
    # computeConstraintForces: espone le forze dei vincoli (contatti e cavi), che servono a
    # ContactMonitor per misurare il contatto con le pareti
    constraint_solver = root.addObject(CONSTRAINT_SOLVER + "ConstraintSolver", name="constraintSolver",
                                       maxIterations=CONSTRAINT_ITERATIONS,
                                       tolerance=CONSTRAINT_TOLERANCE, computeConstraintForces=True)

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
        maze.addObject("CGLinearSolver", iterations=CG_ITERATIONS,
                       tolerance=CUDA_CG_TOLERANCE if use_cuda else 1e-9, threshold=1e-12)
    else:
        raise ValueError(f"LINEAR_SOLVER sconosciuto: {LINEAR_SOLVER!r}")
    maze.addObject("TetrahedronSetTopologyContainer", name="topo",
                   position=points.tolist(), tetrahedra=tets.tolist())
    maze.addObject("TetrahedronSetTopologyModifier")
    # con CUDA massa, FEM, vincoli e correzione dei contatti prendono il template CudaVec3f
    # dal MechanicalObject; il FEM su GPU non ha la variante Parallel (MultiThreading)
    maze_dofs = maze.addObject("MechanicalObject", name="dofs",
                               template="CudaVec3f" if use_cuda else "Vec3d", position=points.tolist())
    maze.addObject("UniformMass", totalMass=mass)
    maze.addObject(("" if use_cuda else par) + "TetrahedronFEMForceField", method="large",
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
    if use_cuda:
        # collisioni su CPU: copia Vec3d dei nodi del labirinto (PointCollisionModel non esiste
        # in CUDA); le forze di contatto tornano sulla GPU attraverso la mapping
        surf.addObject("MechanicalObject", name="dofs", template="Vec3d")
        surf.addObject("IdentityMapping", input="@../dofs", output="@dofs")
    # gruppo 1 = stesso gruppo del tubo introduttore: tubo e tessuto non vengono testati tra
    # loro (nessuna collisione possibile tra modelli dello stesso gruppo).
    # Solo triangoli: lo strumento e' fatto di sfere, e il test sfera-triangolo usa gia' il punto
    # del triangolo piu' vicino (anche su spigoli e vertici). Con anche LineCollisionModel e
    # PointCollisionModel lo stesso contatto veniva creato fino a 3 volte: in prova, spingendo lo
    # strumento contro la carena, 1500 vincoli in gran parte doppi, Gauss-Seidel che non
    # convergeva e simulazione che divergeva.
    surf.addObject("TriangleCollisionModel", group=1)
    # MechanicalObject su cui agiscono i contatti con il tessuto (per ContactMonitor)
    wall_dofs = surf.getObject("dofs") if use_cuda else maze_dofs
    if with_visual:
        # con CUDA il nodo ha gia' un MechanicalObject (uno solo per nodo): grafica in un figlio,
        # mappata dalla copia Vec3d; senza CUDA resta qui, mappata dai nodi del labirinto
        vis = surf.addChild("Visuale") if use_cuda else surf
        vis.addObject("OglModel", name="ogl", color=[0.85, 0.35, 0.35, 0.35])  # semitrasparente
        vis.addObject("IdentityMapping", input="@../dofs", output="@ogl")

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
        if use_cuda:
            # SurfacePressureForceField non esiste in CUDA: agisce su una copia Vec3d dei nodi
            # (stessi indici, quindi i triangoli restano validi) collegata alla GPU dalla mapping
            pressure.addObject("MechanicalObject", name="dofs", template="Vec3d")
            pressure.addObject("IdentityMapping", input="@../dofs", output="@dofs")
        # useTangentStiffness=False: la matrice resta simmetrica, come richiesto da CG e LDL
        pressure.addObject("SurfacePressureForceField", pressure=EXTERNAL_PRESSURE,
                           useTangentStiffness=False)

    # --- Broncoscopio ------------------------------------------------------------------
    # strumento + mano + tubo introduttore (bronchoscope.py); la punta parte in "start" e
    # l'inserimento avviene lungo l'asse +X locale di q0 (la direzione PROBE_START_YAW_DEG)
    params = scope_params(radius)
    axis = R.from_quat(q0).apply([1.0, 0.0, 0.0])
    # il tubo introduttore arriva fino a INTRODUCER_GAP dal tessuto (nessun tratto libero)
    tube_end = bronchoscope.introducer_end_for_mesh(points, start, axis, INTRODUCER_GAP)
    scope = bronchoscope.add_bronchoscope(root, start, q0, params, with_visual=with_visual,
                                          collision_group=1, introducer_end=tube_end)

    # tastiera solo con l'interfaccia grafica (in headless / RL si comanda con set_command)
    keyboard = create_keyboard() if with_visual else None
    probe_ctrl = root.addObject(controller.BronchoscopeController(
        scope, up, trans_step, rot_step, BEND_STEP, max_roll=math.radians(MAX_ROLL_DEG),
        labels=labels[:4], print_every=PRINT_EVERY, keyboard=keyboard, name="ScopeController"))

    # misura dei contatti strumento-pareti a fine passo (per il reward e per il testo a schermo)
    root.addObject(contact_monitor.ContactMonitor(
        constraint_solver, scope.col_dofs, wall_dofs, scope.geometry.sphere_abscissa,
        CONTACT_TIP_LENGTH or SCOPE_BENDING_LENGTH,
        label=labels[4] if labels else None, name="ContactMonitor"))

    # --- Endoscopio --------------------------------------------------------------------
    if with_camera:
        import endoscope_camera   # import qui: Warp serve solo se la telecamera e' attiva
        # la telecamera vede tutti i triangoli di superficie (pareti del canale ed esterno),
        # orientati con la normale verso il lume
        camera = endoscope_camera.EndoscopeCamera(
            points, mesh_utilities.boundary_triangles(tets, points),
            fov_deg=CAMERA_FOV_DEG,
            light_range=CAMERA_LIGHT_RANGE or 4.0 * radius,
            texture_scale=CAMERA_TEXTURE_SCALE or 2.0 * radius,
            max_distance=diag,          # oltre la diagonale della mesh non c'e' niente da vedere
            lens_offset=radius,         # la posa e' il centro dell'ultima sfera: lente sulla sua faccia
            noise=CAMERA_NOISE, seed=CAMERA_SEED)
        print(f"=== Endoscopio: {CAMERA_RESOLUTION}x{CAMERA_RESOLUTION} px, campo visivo "
              f"{CAMERA_FOV_DEG:g} gradi, raycasting su {camera.device} ===")
        root.addObject(controller.EndoscopeController(
            camera, maze_dofs, probe_ctrl, resolution=CAMERA_RESOLUTION,
            show_window=with_visual and CAMERA_SHOW_WINDOW,
            preview_resolution=CAMERA_PREVIEW_RESOLUTION,
            save_dir=CAMERA_SAVE_DIR, save_every=CAMERA_SAVE_EVERY,
            name="EndoscopeController"))
    return root