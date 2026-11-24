"""
Broncoscopio flessibile a cavi (soft robot) per la scena SOFA.

COSA MODELLA (e come lo fa uno strumento vero)
  Un broncoscopio flessibile ha tre parti meccanicamente distinte:

    mano dell'operatore         tubo di inserimento ("catetere")            sezione di curvatura   punta
    (inserisce e ruota)  ===================================================~~~~~~~~~~~~~~~~~~~~~[cam]
                         ^ base                                             ^ inizio curvatura    ^ tip
                         |-------------- shaft: rigido a flessione ---------|-- morbida, a vertebre --|

  1. TUBO DI INSERIMENTO (shaft): lungo, abbastanza rigido a flessione da poter essere spinto,
     ma che si adatta alla forma delle vie aeree. Trasmette alla punta la spinta e la rotazione
     date dalla mano dell'operatore alla base.
  2. SEZIONE DI CURVATURA: gli ultimi 1-3 cm, fatti di vertebre incernierate: molto cedevoli a
     flessione ma praticamente incomprimibili in senso assiale. Due (o quattro) cavi corrono
     paralleli all'asse, a distanza d dall'asse, e sono ancorati alla punta. La leva sul manico
     avvolge i cavi su una puleggia: un cavo si accorcia di delta, l'antagonista si allunga di
     delta, e la sezione si curva verso il cavo tirato con angolo ~ delta / d.
  3. PUNTA: telecamera e luce (endoscope_camera.py), canale di lavoro.

  Fuori dal paziente lo strumento scorre dentro un TUBO INTRODUTTORE rigido (come un tubo
  endotracheale o un bite-block): senza, il lungo tratto libero tra la mano e la bocca si
  inflesserebbe (instabilita' di Eulero) appena la punta incontra resistenza. Il tubo arriva
  fino a ~1 mm dal tessuto: in prova, con un tratto libero di 2-3 cm tra la fine del tubo e la
  trachea, lo strumento spinto contro la carena faceva un'ansa li', passava attorno al bordo del
  tubo e la simulazione esplodeva.

COME E' FATTO IN SOFA
  - Shaft + sezione di curvatura: una catena di nodi Rigid3d (posizione + orientamento) uniti da
    travi di Timoshenko del plugin BeamAdapter (BeamInterpolation + AdaptiveBeamForceFieldAndMass)
    con massa concentrata sui nodi. Raggio, modulo di Young e Poisson sono dati trave per trave,
    quindi shaft e sezione di curvatura hanno rigidezze diverse nello stesso force field.
    L'asse locale +X di ogni nodo e' l'asse dello strumento, +Z e' "su" (l'alto dell'immagine
    dell'endoscopio), +Y e' "sinistra".
    NB: NON si usa BeamFEMForceField di SOFA. La sua formulazione corotazionale (trave lineare nel
    sistema di riferimento del primo nodo) bilancia i momenti sulla geometria indeformata: con la
    forte compressione assiale data dai cavi e rotazioni di vari gradi dentro ogni trave nasce un
    momento spurio ~ T * l * angolo, che in prova piegava tutto lo shaft (di ~10 cm) anche se il
    carico dei cavi e' interno allo strumento. BeamAdapter non ha questo problema (verificato:
    shaft dritto, angolo e tensione del cavo come da teoria).
  - Rigidezza della sezione di curvatura: si sceglie la rigidezza a flessione EI voluta e la si
    ottiene con una sezione circolare piena di raggio piccolo ("raggio strutturale"): cosi'
    EA / EI = 4 / r^2 e' grande e la sezione resta quasi incomprimibile sotto la tensione dei cavi,
    come la catena di vertebre vera. Il raggio "di contatto" (quello che tocca le pareti) e'
    un'altra cosa: lo da' il modello di collisione.
  - Cavi: CableConstraint del plugin SoftRobots. I punti del cavo (occhielli) sono fissati ai nodi
    della sezione di curvatura con una RigidMapping, spostati di d dall'asse. Il primo punto,
    all'inizio della sezione di curvatura, fa da punto di tiro (hasPullPoint=False): il cavo
    esce li' dalla guaina che lo porta fino al manico, quindi la reazione e' tutta interna allo
    strumento e lo shaft non viene piegato dai cavi (come con la guaina a spirale vera).
    valueType="displacement": si impone l'accorciamento del cavo, come fa la leva.
  - Contatto: una fila di sfere (SphereCollisionModel) lungo l'asse, mappate sui nodi con
    RigidMapping. Le sfere si sovrappongono, quindi formano un tubo continuo di raggio r.
  - Mano: un nodo Rigid3d senza dinamica (comandato dal controller) collegato alla base da una
    molla (RestShapeSpringsForceField). Lo scostamento mano-base e' limitato dal controller,
    quindi la forza di spinta e la coppia di rotazione restano limitate (vedi controller.py).

USO (da scene_maze.py)
    params = ScopeParams(radius=..., length=..., ...)
    scope = add_bronchoscope(root, tip_start, q0, params, with_visual)
    scope.dofs, scope.hand_dofs, scope.col_dofs, scope.cables, ...  -> vedi la classe Bronchoscope
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


# =============================================================================
# Parametri
# =============================================================================
@dataclass
class ScopeParams:
    """Parametri fisici del broncoscopio (unita' SI: metri, newton, pascal, kg)."""

    radius: float                    # raggio esterno: contatto con le pareti e grafica
    length: float                    # lunghezza totale inseribile (shaft + sezione di curvatura)
    bending_length: float            # lunghezza della sezione di curvatura distale
    shaft_element_length: float      # lunghezza delle travi dello shaft
    bending_elements: int = 6        # numero di travi (vertebre) della sezione di curvatura

    # rigidezza a flessione EI [N*m^2]: e' la grandezza che si misura sugli strumenti veri
    shaft_bending_stiffness: float = 1.0e-3
    bending_section_stiffness: float = 5.0e-5
    channel_radius: float = 0.0      # raggio del canale di lavoro (lo shaft e' un tubo)
    bending_core_radius: float = 4.0e-4   # raggio strutturale della sezione di curvatura (vedi sopra)
    poisson_ratio: float = 0.3
    density: float = 1100.0          # [kg/m^3] densita' media (polimero + fibre + cavi)
    # Peso dello strumento. E' trascurabile (~0.08 N/m: 0.008 N su 10 cm inseriti) rispetto alle
    # forze di contatto, ma lo farebbe appoggiare per tutta la lunghezza sul fondo del tubo
    # introduttore e delle vie aeree: centinaia di contatti in piu' (simulazione piu' lenta) e un
    # contatto "di fondo" sempre presente nella penalita' del reward. False = senza peso.
    gravity: bool = False
    # Smorzamento. Il Rayleigh di rigidezza dell'integratore (beta * K) va lasciato a 0: si
    # applicherebbe anche alla molla della mano, che non ha un secondo nodo simulato, e
    # diventerebbe un attrito viscoso beta * k sulla base (con 0.1 * 1000 N/m: 100 N*s/m, cioe'
    # 2.5 N di freno a 2.5 cm/s). Le travi pero' hanno bisogno di smorzamento (senza, la sezione
    # di curvatura oscilla e diverge quando i cavi tirano): lo si mette solo sul loro force field
    # (beam_damping, che e' il rayleighStiffness di AdaptiveBeamForceFieldAndMass).
    rayleigh_stiffness: float = 0.0  # Rayleigh di rigidezza dell'integratore (tutto il nodo)
    rayleigh_mass: float = 0.1       # Rayleigh di massa dell'integratore
    beam_damping: float = 0.1        # Rayleigh di rigidezza delle sole travi

    # cavi
    cable_ways: int = 2              # 2 = su/giu' (broncoscopio standard), 4 = anche sinistra/destra
    cable_offset: float = None       # distanza cavo-asse [m] (None = 0.7 * radius)
    max_bend_deg: float = 180.0      # curvatura massima con la leva a fondo corsa
    # Gioco della leva (come negli strumenti veri, dove la punta a leva neutra e' "molle"): ogni
    # cavo e' lasco finche' non si allunga di questo angolo equivalente. Senza gioco, a leva
    # neutra entrambi i cavi sono tesi: se la punta viene torta o piegata da una parete si
    # allungano tutti e due, tirano insieme, comprimono la sezione di curvatura e in prova la
    # simulazione divergeva (spingendo contro la carena).
    cable_slack_deg: float = 10.0
    # Variazione massima dell'accorciamento di un cavo in un passo, come frazione della corsa.
    # Imporre l'accorciamento e' un vincolo cinematico sulla sezione di curvatura, che ha massa
    # minuscola: se cambia troppo in fretta la punta "frusta" e la simulazione diverge (in prova:
    # stabile con la corsa intera in 50 passi, al limite in 25, divergente in 12).
    max_cable_rate: float = 0.02

    # contatto
    sphere_spacing: float = None     # distanza tra le sfere di collisione (None = radius)

    # mano dell'operatore (molla mano -> base)
    # La rigidezza angolare deve essere alta: la base regge un braccio di leva di decine di cm e
    # con una molla debole lo strumento ruoterebbe tutto attorno alla mano. Il limite di coppia
    # sulla rotazione si ottiene limitando l'angolo mano-base (max_roll_offset_deg), non la rigidezza.
    hand_stiffness: float = 1000.0          # [N/m]
    hand_angular_stiffness: float = 5.0     # [N*m/rad]
    max_push_offset: float = 1.0e-3         # [m]    -> forza di spinta massima = rigidezza * offset
    max_roll_offset_deg: float = 0.3        # [gradi] -> coppia massima = rigidezza ang. * angolo
    max_cable_force: float = 20.0           # [N] tensione massima di un cavo (la curvatura libera a
                                            # 180 gradi ne richiede ~10); oltre, il cavo "slitta"

    # tubo introduttore (fuori dal paziente)
    # Il gioco tra strumento e tubo deve superare la distanza di allarme delle collisioni
    # (0.5 * radius in scene_maze.py), altrimenti ogni sfera nel tubo genera contatti inutili
    introducer_radius: float = None  # raggio interno (None = 2.2 * radius)
    introducer_sides: int = 16       # lati del poligono che approssima il tubo
    introducer_ring_spacing: float = None   # distanza tra gli anelli della mesh (None = 6 * radius)

    def __post_init__(self):
        if self.cable_ways not in (2, 4):
            raise ValueError(f"cable_ways deve essere 2 o 4, non {self.cable_ways!r}")
        if self.bending_length >= self.length:
            raise ValueError("la sezione di curvatura deve essere piu' corta dello strumento")
        self.cable_offset = self.cable_offset or 0.7 * self.radius
        self.sphere_spacing = self.sphere_spacing or self.radius
        self.introducer_radius = self.introducer_radius or 2.2 * self.radius
        self.introducer_ring_spacing = self.introducer_ring_spacing or 6.0 * self.radius

    # --- grandezze derivate --------------------------------------------------------
    @property
    def shaft_length(self):
        """Lunghezza del tubo di inserimento (tutto tranne la sezione di curvatura) [m]."""
        return self.length - self.bending_length

    @property
    def shaft_young_modulus(self):
        """Modulo di Young equivalente dello shaft, dalla rigidezza EI e dalla sezione a tubo."""
        inertia = math.pi / 4.0 * (self.radius ** 4 - self.channel_radius ** 4)
        return self.shaft_bending_stiffness / inertia

    @property
    def bending_young_modulus(self):
        """Modulo di Young equivalente della sezione di curvatura (sezione piena, raggio strutturale)."""
        inertia = math.pi / 4.0 * self.bending_core_radius ** 4
        return self.bending_section_stiffness / inertia

    @property
    def max_cable_displacement(self):
        """Accorciamento del cavo [m] che da' la curvatura massima: arco = d * angolo."""
        return self.cable_offset * math.radians(self.max_bend_deg)

    @property
    def cable_slack(self):
        """Gioco di ogni cavo [m]: allungamento consentito prima che il cavo vada in tensione."""
        return self.cable_offset * math.radians(self.cable_slack_deg)

    @property
    def max_push_force(self):
        """Forza massima con cui la mano spinge o tira lo strumento [N]."""
        return self.hand_stiffness * self.max_push_offset

    @property
    def max_roll_torque(self):
        """Coppia massima con cui la mano ruota lo strumento [N*m]."""
        return self.hand_angular_stiffness * math.radians(self.max_roll_offset_deg)


# Direzione di ogni cavo nel piano della sezione (coordinate locali y, z del nodo).
# Il cavo "su" sta a +Z locale: tirarlo accorcia quel lato e la punta si piega verso l'alto
# dell'immagine dell'endoscopio. Ogni coppia (cavo, antagonista) e' comandata da una leva.
CABLE_DIRECTIONS = {
    "su": (0.0, +1.0), "giu": (0.0, -1.0),            # leva 0: su/giu'
    "sinistra": (+1.0, 0.0), "destra": (-1.0, 0.0),   # leva 1: sinistra/destra (solo 4 vie)
}
# leva -> (cavo tirato quando la leva e' positiva, cavo tirato quando e' negativa)
LEVERS = (("su", "giu"), ("sinistra", "destra"))


# =============================================================================
# Geometria a riposo
# =============================================================================
@dataclass
class ScopeGeometry:
    """Discretizzazione dello strumento dritto, dalla base (indice 0) alla punta (ultimo nodo)."""

    node_abscissa: np.ndarray    # (N,) ascissa curvilinea di ogni nodo, 0 = base [m]
    edges: np.ndarray            # (N-1, 2) travi tra nodi consecutivi
    shaft_edges: list            # indici delle travi dello shaft
    bending_edges: list          # indici delle travi della sezione di curvatura
    bending_start_node: int      # primo nodo della sezione di curvatura (dove escono i cavi)
    sphere_abscissa: np.ndarray  # (S,) ascissa di ogni sfera di collisione [m]
    sphere_node: np.ndarray      # (S,) nodo a cui e' attaccata ogni sfera
    sphere_local: np.ndarray     # (S, 3) posizione della sfera nel sistema locale del suo nodo

    @property
    def n_nodes(self):
        """Numero di nodi Rigid3d dello strumento."""
        return len(self.node_abscissa)


def build_geometry(p: ScopeParams) -> ScopeGeometry:
    """Ascisse dei nodi e delle sfere di collisione.

    Lo shaft e' diviso in travi lunghe ~shaft_element_length, la sezione di curvatura in
    bending_elements travi piu' corte (serve risoluzione dove la curvatura e' alta). Ogni sfera
    si attacca al nodo piu' vicino, con uno spostamento lungo l'asse locale X."""
    n_shaft = max(1, int(math.ceil(p.shaft_length / p.shaft_element_length)))
    shaft_s = np.linspace(0.0, p.shaft_length, n_shaft + 1)
    bend_s = p.shaft_length + np.linspace(0.0, p.bending_length, p.bending_elements + 1)[1:]
    node_s = np.concatenate([shaft_s, bend_s])
    n_nodes = len(node_s)
    edges = np.stack([np.arange(n_nodes - 1), np.arange(1, n_nodes)], axis=1)

    n_spheres = int(math.ceil(p.length / p.sphere_spacing)) + 1
    sphere_s = np.linspace(0.0, p.length, n_spheres)
    sphere_node = np.abs(sphere_s[:, None] - node_s[None, :]).argmin(axis=1)
    sphere_local = np.zeros((n_spheres, 3))
    sphere_local[:, 0] = sphere_s - node_s[sphere_node]

    return ScopeGeometry(node_abscissa=node_s, edges=edges,
                         shaft_edges=list(range(n_shaft)),
                         bending_edges=list(range(n_shaft, n_nodes - 1)),
                         bending_start_node=n_shaft,
                         sphere_abscissa=sphere_s, sphere_node=sphere_node,
                         sphere_local=sphere_local)


def rest_poses(geom: ScopeGeometry, length, tip_start, q0):
    """Pose Rigid3d [x, y, z, qx, qy, qz, qw] dei nodi con lo strumento dritto, la punta in
    tip_start e l'asse lungo la direzione locale +X dell'orientamento q0."""
    axis = R.from_quat(q0).apply([1.0, 0.0, 0.0])
    positions = np.asarray(tip_start, dtype=float) + np.outer(geom.node_abscissa - length, axis)
    return np.hstack([positions, np.tile(q0, (geom.n_nodes, 1))])


# =============================================================================
# Tubo introduttore e tubo grafico
# =============================================================================
def tube_mesh(start, end, radius, sides, ring_spacing, inward=True):
    """Superficie laterale di un cilindro da start a end: anelli di 'sides' vertici distanti
    circa ring_spacing. Tanti triangoli corti invece di pochi lunghissimi: la BVH della rilevazione
    delle collisioni scarta subito i triangoli lontani dallo strumento.

    inward=True orienta le normali verso l'asse: e' il lato da cui si avvicina lo strumento
    che scorre dentro il tubo introduttore. Restituisce (punti, triangoli)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = np.linalg.norm(end - start)
    axis = (end - start) / length
    u = np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = 2.0 * math.pi * np.arange(sides) / sides
    ring = radius * (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v))
    n_rings = max(2, int(math.ceil(length / ring_spacing)) + 1)
    points = np.vstack([start + axis * t + ring for t in np.linspace(0.0, length, n_rings)])
    tris = []
    for r in range(n_rings - 1):
        for i in range(sides):
            a, b = r * sides + i, r * sides + (i + 1) % sides   # anello r
            c, d = a + sides, b + sides                         # anello r + 1
            tris += [[a, b, d], [a, d, c]]      # normali verso l'esterno (regola della mano destra)
    tris = np.array(tris)
    if inward:
        tris = tris[:, [0, 2, 1]]
    return points, tris


def scope_skin(geom: ScopeGeometry, radius, sides=12):
    """Superficie grafica dello strumento: un anello di vertici per ogni sfera di collisione,
    nel sistema locale del nodo a cui e' attaccata la sfera, piu' un tappo sulla punta.
    Restituisce (punti locali, indice del nodo per punto, triangoli)."""
    angles = 2.0 * math.pi * np.arange(sides) / sides
    ring = np.stack([np.zeros(sides), radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    local, node = [], []
    for s_local, s_node in zip(geom.sphere_local, geom.sphere_node):
        local.append(ring + s_local)
        node += [s_node] * sides
    # tappo: un vertice al centro della faccia della punta (la lente)
    local.append(geom.sphere_local[-1][None, :] + [[radius, 0.0, 0.0]])
    node.append(geom.sphere_node[-1])
    local = np.vstack(local)

    n_rings = len(geom.sphere_local)
    tris = []
    for r in range(n_rings - 1):
        for i in range(sides):
            a, b = r * sides + i, r * sides + (i + 1) % sides
            tris += [[a, b, b + sides], [a, b + sides, a + sides]]
    tip_center, last = len(local) - 1, (n_rings - 1) * sides
    for i in range(sides):
        tris.append([last + i, last + (i + 1) % sides, tip_center])
    return local, np.array(node), np.array(tris)


# =============================================================================
# Costruzione nella scena
# =============================================================================
@dataclass
class Bronchoscope:
    """Riferimenti agli oggetti SOFA del broncoscopio, usati dai controller."""

    params: ScopeParams
    geometry: ScopeGeometry
    node: object                 # nodo SOFA "Broncoscopio"
    dofs: object                 # MechanicalObject Rigid3d dei nodi (0 = base, -1 = punta)
    hand_dofs: object            # MechanicalObject Rigid3d della mano (comandata dal controller)
    col_dofs: object             # MechanicalObject Vec3d dei centri delle sfere di collisione
    cables: dict                 # nome del cavo -> CableConstraint
    start_poses: np.ndarray      # (N, 7) pose iniziali dei nodi
    axis: np.ndarray             # direzione di inserimento (asse del tubo introduttore)
    mouth: np.ndarray            # posizione iniziale della punta, appena fuori dal paziente
    introducer_group: int = 1    # gruppo di collisione del tubo introduttore


def add_bronchoscope(root, tip_start, q0, p: ScopeParams, with_visual=True, collision_group=1,
                     introducer_end=None):
    """Aggiunge a root il broncoscopio (con la punta in tip_start, asse lungo +X di q0), la mano
    che lo spinge e il tubo introduttore.

    collision_group: gruppo del tubo introduttore; va dato lo stesso gruppo alla mesh dei
        polmoni, cosi' tubo e polmoni non vengono testati tra loro.
    introducer_end: distanza [m] da tip_start, lungo l'asse, a cui finisce il tubo (> 0 = oltre
        la punta iniziale, che allora parte dentro il tubo). Conviene farlo finire appena prima
        del tessuto (vedi introducer_end_for_mesh). None = finisce sulla punta iniziale."""
    geom = build_geometry(p)
    poses = rest_poses(geom, p.length, tip_start, q0)
    axis = R.from_quat(q0).apply([1.0, 0.0, 0.0])

    # --- mano dell'operatore: nessun solutore, la muove solo il controller --------------
    hand = root.addChild("Mano")
    hand_dofs = hand.addObject("MechanicalObject", name="dofs", template="Rigid3d",
                               position=[poses[0].tolist()], showObject=with_visual,
                               showObjectScale=3.0 * p.radius)

    # --- strumento -----------------------------------------------------------------------
    scope = root.addChild("Broncoscopio")
    if with_visual:
        scope.addObject("VisualStyle", displayFlags="showVisualModels hideCollisionModels "
                                                    "hideBehaviorModels showInteractionForceFields")
    # una catena di ~50 nodi Rigid3d (~300 incognite): il solutore diretto costa pochissimo e
    # da' una correzione dei vincoli (contatti + cavi) esatta
    scope.addObject("EulerImplicitSolver", rayleighStiffness=p.rayleigh_stiffness,
                    rayleighMass=p.rayleigh_mass)
    scope.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixd")
    scope.addObject("EdgeSetTopologyContainer", name="topo", edges=geom.edges.tolist())
    dofs = scope.addObject("MechanicalObject", name="dofs", template="Rigid3d",
                           position=poses.tolist(), showObject=False)

    # travi (BeamAdapter): proprieta' trave per trave, prima lo shaft poi la sezione di curvatura.
    # Le travi dello shaft sono tubi (canale di lavoro), quelle della curvatura hanno la sezione
    # piena "strutturale" (vedi sopra).
    n_shaft, n_bend = len(geom.shaft_edges), len(geom.bending_edges)
    scope.addObject("BeamInterpolation", name="interpolation", straight=True,
                    radius=[p.radius] * n_shaft + [p.bending_core_radius] * n_bend,
                    innerRadius=[p.channel_radius] * n_shaft + [0.0] * n_bend,
                    defaultYoungModulus=[p.shaft_young_modulus] * n_shaft
                                        + [p.bending_young_modulus] * n_bend,
                    defaultPoissonRatio=[p.poisson_ratio] * (n_shaft + n_bend))
    # Massa: concentrata sui nodi (UniformMass), non quella consistente di BeamAdapter
    # (computeMass=False). La massa consistente distribuisce il peso con dei momenti che dipendono
    # dalla configurazione, e la compensazione del peso (vedi sotto) non riusciva ad annullarlo
    # (in prova restava ~12% del peso). Con masse ai nodi peso e compensazione sono esattamente
    # opposti. Massa per unita' di lunghezza = densita' * sezione piena di raggio radius.
    # SOFA stampa un WARNING ("Only one BaseMass is permitted in a Node... replaced"): e' atteso.
    # AdaptiveBeamForceFieldAndMass e' sia force field sia massa; UniformMass lo sostituisce solo
    # come massa e le travi continuano a funzionare (verificato: freccia di una mensola caricata
    # in punta entro l'1% dalla teoria, curvatura con i cavi invariata).
    scope.addObject("AdaptiveBeamForceFieldAndMass", name="beams", interpolation="@interpolation",
                    computeMass=False, rayleighStiffness=p.beam_damping)
    node_mass = p.density * math.pi * p.radius ** 2 * p.length / geom.n_nodes
    # inerzia per unita' di massa (formato di UniformMass Rigid3d) di un tratto di asta lungo l:
    # r^2/2 attorno all'asse, l^2/12 + r^2/4 attorno agli assi trasversali
    l = p.length / (geom.n_nodes - 1)
    i_axis, i_side = p.radius ** 2 / 2.0, l ** 2 / 12.0 + p.radius ** 2 / 4.0
    scope.addObject("UniformMass", name="massa",
                    vertexMass=f"{node_mass} 1 {i_axis} 0 0 0 {i_side} 0 0 0 {i_side}")
    if not p.gravity:
        _add_gravity_compensation(scope, geom.n_nodes, node_mass,
                                  np.asarray(root.gravity.value, dtype=float))

    # molla mano -> base: tutti i 6 gradi di liberta' della base seguono la mano
    scope.addObject("RestShapeSpringsForceField", name="mano", points=[0],
                    external_rest_shape="@../Mano/dofs", external_points=[0],
                    stiffness=[p.hand_stiffness], angularStiffness=[p.hand_angular_stiffness],
                    drawSpring=with_visual)
    scope.addObject("GenericConstraintCorrection")

    # --- cavi (SoftRobots) ---------------------------------------------------------------
    cables = _add_cables(scope, geom, p, poses, root.dt.value)

    # --- collisione: sfere sovrapposte lungo l'asse ---------------------------------------
    # Posizioni date in coordinate GLOBALI (la mapping ricava quelle locali): sono anche le
    # posizioni a cui Sofa.Simulation.reset() riporta il MechanicalObject. Con coordinate locali
    # dopo un reset le sfere finivano per un passo vicino all'origine, cioe' dentro i polmoni,
    # e la rilevazione delle collisioni creava contatti enormi.
    col = scope.addChild("Collisione")
    col_dofs = col.addObject("MechanicalObject", name="dofs", template="Vec3d",
                             position=_to_world(poses, geom.sphere_node, geom.sphere_local).tolist())
    col.addObject("SphereCollisionModel", radius=p.radius)
    col.addObject("RigidMapping", input="@../dofs", output="@dofs", globalToLocalCoords=True,
                  rigidIndexPerPoint=geom.sphere_node.tolist())

    # --- grafica: un tubo che segue i nodi -------------------------------------------------
    if with_visual:
        local, node_of, tris = scope_skin(geom, p.radius)
        skin = scope.addChild("Visuale")
        skin.addObject("OglModel", name="ogl", position=_to_world(poses, node_of, local).tolist(),
                       triangles=tris.tolist(), color=[0.15, 0.15, 0.18, 1.0])
        skin.addObject("RigidMapping", input="@../dofs", output="@ogl", globalToLocalCoords=True,
                       rigidIndexPerPoint=node_of.tolist())

    # --- tubo introduttore: dalla base iniziale (con un po' di margine) fino al tessuto -------
    mouth = np.asarray(tip_start, dtype=float)
    _add_introducer(root, mouth - axis * (p.length + 4.0 * p.radius),
                    mouth + axis * (introducer_end or 0.0), p, collision_group, with_visual)

    print("=== Broncoscopio ===")
    print(f"  lunghezza {p.length * 1000:.0f} mm (curvatura {p.bending_length * 1000:.0f} mm), "
          f"diametro {2000 * p.radius:.1f} mm, {geom.n_nodes} nodi, {len(geom.sphere_abscissa)} sfere")
    print(f"  EI shaft {p.shaft_bending_stiffness:.2g} N*m^2 (E = {p.shaft_young_modulus:.3g} Pa), "
          f"EI curvatura {p.bending_section_stiffness:.2g} N*m^2 (E = {p.bending_young_modulus:.3g} Pa)")
    print(f"  cavi: {p.cable_ways} vie, a {p.cable_offset * 1000:.2f} mm dall'asse, curvatura max "
          f"{p.max_bend_deg:g} gradi (corsa {p.max_cable_displacement * 1000:.2f} mm)")
    print(f"  mano: spinta max {p.max_push_force:.2f} N, coppia max {p.max_roll_torque * 1000:.1f} mN*m")

    return Bronchoscope(params=p, geometry=geom, node=scope, dofs=dofs, hand_dofs=hand_dofs,
                        col_dofs=col_dofs, cables=cables, start_poses=poses, axis=axis,
                        mouth=mouth, introducer_group=collision_group)


def _add_gravity_compensation(scope, n_nodes, node_mass, gravity):
    """Annulla il peso dello strumento con una forza costante -m * g su ogni nodo.

    Non basta mettere gravity = 0 nel nodo dello strumento: SOFA ricopia a ogni passo la gravita'
    del nodo padre nei figli (in prova il valore tornava -9.81 dopo ogni passo)."""
    forces = np.zeros((n_nodes, 6))                 # Rigid3d: forza (3) + momento (3)
    forces[:, 0:3] = -node_mass * gravity
    scope.addObject("ConstantForceField", name="compensazionePeso",
                    indices=list(range(n_nodes)), forces=forces.tolist())


def _add_cables(scope, geom: ScopeGeometry, p: ScopeParams, poses, dt):
    """Crea i cavi tendinei: per ogni cavo un occhiello su ogni nodo della sezione di curvatura,
    a distanza cable_offset dall'asse. Tutti i punti di tutti i cavi stanno in un solo
    MechanicalObject (nodo "Cavi"); ogni CableConstraint usa i propri indici.

    Gli occhielli si danno in coordinate GLOBALI (globalToLocalCoords=True, la mapping ricava da
    sola quelle locali): CableConstraint misura la lunghezza a riposo del cavo durante l'init,
    prima che la mapping abbia calcolato le posizioni, e con coordinate locali vedrebbe tutti gli
    occhielli nello stesso punto (lunghezza 0 -> forze enormi e simulazione che diverge)."""
    names = [n for lever in LEVERS[:p.cable_ways // 2] for n in lever]
    cable_nodes = list(range(geom.bending_start_node, geom.n_nodes))   # dalla base della curvatura alla punta

    local, node_of, indices = [], [], {}
    for name in names:
        dy, dz = CABLE_DIRECTIONS[name]
        indices[name] = list(range(len(local), len(local) + len(cable_nodes)))
        local += [[0.0, dy * p.cable_offset, dz * p.cable_offset]] * len(cable_nodes)
        node_of += cable_nodes

    local, node_of = np.asarray(local), np.asarray(node_of)
    world = _to_world(poses, node_of, local)

    node = scope.addChild("Cavi")
    node.addObject("MechanicalObject", name="dofs", template="Vec3d", position=world.tolist())
    cables = {}
    for name in names:
        # hasPullPoint=False: il primo occhiello (inizio della curvatura) e' il punto di tiro,
        # quindi il cavo e' interno allo strumento. value = accorciamento imposto [m].
        # minForce=0: un cavo puo' solo tirare; l'antagonista, che la puleggia allunga, resta
        # lasco se la curvatura vera e' minore di quella comandata (es. punta bloccata da una
        # parete). Va impostato esplicitamente: se non e' "set" SoftRobots non lo applica.
        # maxForce: SoftRobots lo confronta con la grandezza risolta dal solutore dei vincoli, che
        # e' un impulso (forza * dt), quindi la forza massima va moltiplicata per dt (con 30 e
        # dt = 0.01 il limite reale era 3000 N)
        cables[name] = node.addObject("CableConstraint", name=f"cavo_{name}",
                                      indices=indices[name], hasPullPoint=False,
                                      valueType="displacement", value=[-p.cable_slack],
                                      minForce=0.0, maxForce=p.max_cable_force * dt,
                                      maxDispVariation=p.max_cable_rate
                                                       * (p.max_cable_displacement + p.cable_slack),
                                      drawPoints=False, drawPullPoint=False)
    node.addObject("RigidMapping", input="@../dofs", output="@dofs", globalToLocalCoords=True,
                   rigidIndexPerPoint=node_of.tolist())
    return cables


def _to_world(poses, node_of, local):
    """Coordinate globali di punti dati nel sistema locale dei nodi: poses (N, 7) pose dei nodi,
    node_of (P,) nodo di ogni punto, local (P, 3) coordinate locali."""
    node_of, local = np.asarray(node_of), np.asarray(local, dtype=float)
    return poses[node_of, 0:3] + R.from_quat(poses[node_of, 3:7]).apply(local)


def introducer_end_for_mesh(points, tip_start, axis, gap):
    """Distanza da tip_start, lungo axis, a cui far finire il tubo introduttore: 'gap' metri prima
    del primo punto della mesh del tessuto (la mesh deve stare tutta davanti alla punta)."""
    first = float(((np.asarray(points) - np.asarray(tip_start)) @ np.asarray(axis)).min())
    return first - gap


def _add_introducer(root, start, end, p: ScopeParams, group, with_visual):
    """Tubo rigido e fisso (non simulato) attorno al tratto di strumento fuori dal paziente."""
    points, tris = tube_mesh(start, end, p.introducer_radius, p.introducer_sides,
                             p.introducer_ring_spacing, inward=True)
    tube = root.addChild("TuboIntroduttore")
    tube.addObject("MeshTopology", name="topo", position=points.tolist(), triangles=tris.tolist())
    tube.addObject("MechanicalObject", name="dofs", template="Vec3d", position=points.tolist())
    tube.addObject("TriangleCollisionModel", moving=False, simulated=False, group=group)
    if with_visual:
        tube.addObject("OglModel", name="ogl", position=points.tolist(), triangles=tris.tolist(),
                       color=[0.7, 0.75, 0.8, 0.25])
