"""
Telecamera endoscopica simulata, montata sulla punta della sonda.

Il modulo NON dipende da SOFA: riceve solo
  - la mesh di superficie del labirinto/organo (posizioni dei nodi + triangoli),
  - la posa della punta della sonda,
e restituisce l'immagine che vedrebbe un endoscopio. Si puo' quindi riusare identico nelle
versioni Genesis e Newton.

COME FUNZIONA (raycasting su GPU con NVIDIA Warp)
  Per ogni pixel si lancia un raggio dalla lente sulla punta della sonda e si cerca il primo
  triangolo della superficie che il raggio colpisce. La ricerca usa una BVH (Bounding Volume
  Hierarchy: un albero di scatole che racchiudono gruppi di triangoli), quindi costa
  ~log(numero di triangoli) invece di provare tutti i triangoli. Quando il tessuto si deforma la
  BVH non si ricostruisce da zero: si "rifitta" (si aggiornano solo le scatole), che e' molto
  piu' veloce. Con 64x64 pixel e ~20-30k triangoli il tutto costa ~1 ms per immagine.

  Dal punto colpito si ricavano:
    - la DISTANZA (immagine di profondita'),
    - la NORMALE della superficie (per l'illuminazione),
    - la posizione A RIPOSO del punto (per la texture, che cosi' resta "attaccata" al tessuto e
      si deforma con lui invece di scivolarci sopra).

COSA RENDE L'IMMAGINE SIMILE A QUELLA DI UN VERO ENDOSCOPIO
  1. Luce sulla punta, coassiale alla telecamera: la luminosita' cala con 1/distanza^2, quindi le
     pareti vicine sono chiare e il lume lontano e' buio (l'effetto "tunnel" tipico delle
     endoscopie). Se la punta tocca la parete l'immagine diventa quasi tutta rossa e saturata.
  2. Ottica grandangolare fisheye (100-140 gradi) con immagine circolare e angoli neri.
  3. Riflessi speculari (la mucosa e' bagnata): termine di Blinn-Phong.
  4. Texture procedurale rosata con una rete di vasi sanguigni (rumore di Perlin 3D).
  5. Vignettatura (bordi piu' scuri), rumore del sensore, correzione gamma.

USO
    cam = EndoscopeCamera(rest_points, triangles)      # una volta sola
    cam.update_mesh(points)                             # a ogni passo: posizioni deformate
    rgb, depth = cam.render(tip_pose, resolution=64)    # rgb uint8 (H, W, 3), depth float32 [m]

  tip_pose = [x, y, z, qx, qy, qz, qw] (come Rigid3d di SOFA). Convenzione degli assi della sonda
  (la stessa di scene_maze.py): asse locale +X = direzione in cui guarda la sonda, +Z = "alto"
  dell'immagine, quindi +Y locale = sinistra dell'immagine.
"""

import math

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R

wp.config.quiet = True   # niente messaggi di Warp a ogni compilazione dei kernel


# =============================================================================
# Parametri ottici, raccolti in una struct per passarli ai kernel in un colpo solo
# =============================================================================
@wp.struct
class Optics:
    half_fov: float        # meta' del campo visivo [rad]
    max_distance: float    # oltre questa distanza un raggio non "vede" niente (pixel nero) [m]
    light_power: float     # = light_range^2: una parete perpendicolare a light_range ha luminosita' 1
    ambient: float         # luce diffusa minima, indipendente dalla distanza (rimbalzi nel lume)
    specular: float        # intensita' dei riflessi speculari (mucosa bagnata)
    shininess: float       # esponente di Blinn-Phong: piu' alto = riflessi piu' piccoli e netti
    vignette: float        # quanto si scuriscono i bordi dell'immagine (0 = niente)
    noise: float           # deviazione standard del rumore del sensore (luminosita' in [0, 1])
    texture_scale: float   # dimensione tipica delle maglie dei vasi sanguigni [m]
    texture_seed: int      # seme della texture (cambiarlo = un altro "paziente")


# =============================================================================
# Normali dei vertici (per un'illuminazione morbida, senza vedere le faccette dei triangoli)
# =============================================================================
@wp.kernel
def _accumulate_normals(points: wp.array(dtype=wp.vec3), tris: wp.array(dtype=wp.int32),
                        normals: wp.array(dtype=wp.vec3)):
    """Un thread per triangolo: somma la normale del triangolo sui suoi tre vertici.
    Il prodotto vettoriale non normalizzato e' lungo 2 * area: cosi' i triangoli grandi pesano
    di piu' (media pesata sull'area, la scelta standard)."""
    f = wp.tid()
    i = tris[3 * f + 0]
    j = tris[3 * f + 1]
    k = tris[3 * f + 2]
    n = wp.cross(points[j] - points[i], points[k] - points[i])
    wp.atomic_add(normals, i, n)   # atomic: piu' triangoli scrivono sullo stesso vertice in parallelo
    wp.atomic_add(normals, j, n)
    wp.atomic_add(normals, k, n)


@wp.kernel
def _normalize(normals: wp.array(dtype=wp.vec3)):
    v = wp.tid()
    n = normals[v]
    length = wp.length(n)
    if length > 0.0:
        normals[v] = n / length


# =============================================================================
# Texture del tessuto
# =============================================================================
@wp.func
def _tissue_albedo(p: wp.vec3, scale: float, seed: int):
    """Colore del tessuto nel punto p (posizione A RIPOSO, in metri).

    - Fondo rosato con macchie ampie e leggere (variazioni di colore della mucosa).
    - Rete di vasi: il rumore di Perlin e' una funzione liscia che vale 0 su superfici
      "ondulate"; dove queste superfici tagliano la parete si ottengono linee curve e ramificate,
      che sembrano capillari. Due reti: vasi piu' grandi e marcati, capillari piu' fini e tenui."""
    q = p / scale

    # macchie a bassa frequenza: noise in [-1, 1] -> luminosita' del fondo +-15%
    mottle = wp.noise(wp.rand_init(seed), q * 0.35)

    # vasi grandi: pixel "vicini allo zero" del rumore. Lo spessore della linea varia lungo il
    # vaso (un secondo rumore lento lo modula), cosi' i vasi si assottigliano e spariscono come
    # quelli veri invece di sembrare curve di livello tutte uguali
    n1 = wp.noise(wp.rand_init(seed + 1), q)
    width1 = 0.03 + 0.07 * wp.max(wp.noise(wp.rand_init(seed + 3), q * 0.5), 0.0)
    vessel_big = 1.0 - wp.smoothstep(0.0, width1, wp.abs(n1))
    # capillari: frequenza 2.7 volte piu' alta, linee piu' sottili e tenui, solo in alcune zone
    n2 = wp.noise(wp.rand_init(seed + 2), q * 2.7)
    width2 = 0.04 * wp.max(wp.noise(wp.rand_init(seed + 4), q * 0.8), 0.0)
    vessel_small = 1.0 - wp.smoothstep(0.0, width2 + 1.0e-4, wp.abs(n2))
    vessels = wp.max(0.85 * vessel_big, 0.35 * vessel_small)

    base = wp.vec3(0.82, 0.40, 0.34) * (1.0 + 0.15 * mottle)   # mucosa rosa-salmone
    vessel_color = wp.vec3(0.45, 0.05, 0.07)                   # sangue
    return base * (1.0 - vessels) + vessel_color * vessels


# =============================================================================
# Kernel principale: un thread per pixel
# =============================================================================
@wp.kernel
def _render(mesh: wp.uint64,
            rest: wp.array(dtype=wp.vec3), tris: wp.array(dtype=wp.int32),
            normals: wp.array(dtype=wp.vec3),
            origin: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3,
            opt: Optics, frame_seed: int,
            rgb: wp.array2d(dtype=wp.vec3), depth: wp.array2d(dtype=float)):
    i, j = wp.tid()          # i = riga (0 = in alto), j = colonna (0 = a sinistra)
    height = rgb.shape[0]
    width = rgb.shape[1]

    # --- 1. pixel -> coordinate normalizzate (x, y) in [-1, 1], con y verso l'alto ---
    x = 2.0 * (float(j) + 0.5) / float(width) - 1.0
    y = 1.0 - 2.0 * (float(i) + 0.5) / float(height)
    r = wp.sqrt(x * x + y * y)   # distanza dal centro dell'immagine (1 = bordo del cerchio)

    if r > 1.0:
        # fuori dal cerchio dell'ottica: nero, come negli endoscopi veri
        rgb[i, j] = wp.vec3(0.0, 0.0, 0.0)
        depth[i, j] = opt.max_distance
        return

    # --- 2. direzione del raggio: proiezione fisheye "equidistante" ---
    # l'angolo dall'asse ottico cresce linearmente con la distanza dal centro dell'immagine
    # (r = 1 -> theta = meta' del campo visivo). Con una prospettiva normale non si potrebbe
    # superare ~120 gradi senza deformare enormemente i bordi.
    theta = r * opt.half_fov
    lateral = wp.vec3(0.0, 0.0, 0.0)
    if r > 1.0e-8:
        lateral = (right * x + up * y) / r      # direzione "di lato" corrispondente al pixel
    direction = fwd * wp.cos(theta) + lateral * wp.sin(theta)   # gia' di lunghezza 1

    # --- 3. primo triangolo colpito dal raggio (ricerca nella BVH) ---
    hit = wp.mesh_query_ray(mesh, origin, direction, opt.max_distance)

    color = wp.vec3(0.0, 0.0, 0.0)   # nessun bersaglio entro max_distance -> nero (nessuna luce torna)
    distance = opt.max_distance
    if hit.result:
        distance = hit.t   # direzione di lunghezza 1 -> t e' la distanza in metri
        if hit.sign > 0.0:
            # il raggio colpisce il lato "davanti" del triangolo, cioe' la parete vista dal lume
            # (i triangoli sono orientati con la normale che esce dal materiale)

            # vertici del triangolo e coordinate baricentriche (u, v, 1-u-v) del punto colpito
            a = tris[3 * hit.face + 0]
            b = tris[3 * hit.face + 1]
            c = tris[3 * hit.face + 2]
            w = 1.0 - hit.u - hit.v

            # normale interpolata dai vertici: superficie liscia invece che a faccette
            n = wp.normalize(normals[a] * hit.u + normals[b] * hit.v + normals[c] * w)
            cos_i = wp.max(wp.dot(n, -direction), 0.0)   # coseno dell'angolo di incidenza

            # luce puntiforme sulla punta: intensita' ~ 1 / distanza^2
            falloff = opt.light_power / (distance * distance)

            # colore del tessuto dalla posizione a riposo del punto colpito
            p_rest = rest[a] * hit.u + rest[b] * hit.v + rest[c] * w
            albedo = _tissue_albedo(p_rest, opt.texture_scale, opt.texture_seed)

            # diffusione (Lambert) + riflesso speculare (Blinn-Phong) + un minimo di luce diffusa.
            # Luce e telecamera coincidono, quindi il "vettore a meta' strada" di Blinn-Phong e'
            # proprio -direction e il riflesso si vede dove la parete e' perpendicolare al raggio.
            diffuse = albedo * (cos_i * falloff)
            spec = opt.specular * wp.pow(cos_i, opt.shininess) * falloff
            color = diffuse + wp.vec3(spec, spec, spec) + albedo * opt.ambient
        else:
            # lato "dietro": la lente e' dentro il materiale (sonda che ha penetrato la parete)
            color = wp.vec3(0.04, 0.0, 0.0)

    # --- 4. effetti del sensore e dell'ottica ---
    color = color * (1.0 - opt.vignette * r * r)   # vignettatura: bordi piu' scuri

    # "tone mapping": comprime le luci troppo forti invece di tagliarle di netto
    # (1 -> 0.63, 3 -> 0.95), come fa l'esposizione automatica di una telecamera. Si comprime la
    # luminosita' (il canale piu' forte) e si scalano i tre canali dello stesso fattore: cosi' il
    # colore resta saturo; comprimendo i canali separatamente il rosso, che satura per primo,
    # verrebbe raggiunto dagli altri e il tessuto vicino diventerebbe bianco-rosa.
    peak = wp.max(color[0], wp.max(color[1], color[2]))
    if peak > 1.0e-6:
        color = color * ((1.0 - wp.exp(-peak)) / peak)

    # correzione gamma (i monitor si aspettano valori "percettivi", non lineari)
    color = wp.vec3(wp.pow(color[0], 1.0 / 2.2), wp.pow(color[1], 1.0 / 2.2), wp.pow(color[2], 1.0 / 2.2))

    # rumore del sensore, diverso a ogni immagine (frame_seed) e a ogni pixel
    state = wp.rand_init(frame_seed, i * width + j)
    color = color + wp.vec3(wp.randn(state), wp.randn(state), wp.randn(state)) * opt.noise

    rgb[i, j] = wp.vec3(wp.clamp(color[0], 0.0, 1.0), wp.clamp(color[1], 0.0, 1.0),
                        wp.clamp(color[2], 0.0, 1.0))
    depth[i, j] = distance


# =============================================================================
# Interfaccia Python
# =============================================================================
class EndoscopeCamera:
    """Endoscopio virtuale. Vedi la docstring del modulo per l'uso.

    Parametri (lunghezze in metri):
      rest_points    (N, 3) posizioni a riposo di TUTTI i nodi (anche interni: non danno fastidio)
      triangles      (M, 3) triangoli della superficie, con la normale (regola della mano destra)
                     che esce dal materiale, cioe' punta verso il lume
      fov_deg        campo visivo totale (endoscopi reali: 100-140 gradi)
      light_range    distanza a cui una parete perpendicolare appare con luminosita' "piena"
      texture_scale  dimensione delle maglie dei vasi
      max_distance   distanza massima visibile (oltre: nero)
      lens_offset    distanza della lente dal punto della posa lungo l'asse della sonda
                     (la posa di SOFA e' il centro della sfera di punta: la lente sta davanti)
      ambient, specular, shininess, vignette, noise: aspetto dell'immagine (vedi Optics)
      seed           seme della texture
      device         "cuda:0", "cpu" o None (= GPU se disponibile)
    """

    def __init__(self, rest_points, triangles, fov_deg=120.0, light_range=0.006,
                 texture_scale=0.003, max_distance=0.3, lens_offset=0.0,
                 ambient=0.01, specular=0.6, shininess=25.0, vignette=0.35, noise=0.01,
                 seed=0, device=None):
        wp.init()
        self.device = device or ("cuda:0" if wp.is_cuda_available() else "cpu")
        rest_points = np.asarray(rest_points, dtype=np.float32)
        triangles = np.asarray(triangles, dtype=np.int32)

        # dati sulla GPU: posizioni correnti (aggiornate a ogni passo), posizioni a riposo
        # (per la texture), indici dei triangoli e normali dei vertici
        self.points = wp.array(rest_points, dtype=wp.vec3, device=self.device)
        self.rest = wp.array(rest_points, dtype=wp.vec3, device=self.device)
        self.tris = wp.array(triangles.ravel(), dtype=wp.int32, device=self.device)
        self.normals = wp.zeros(len(rest_points), dtype=wp.vec3, device=self.device)
        self.n_tris = len(triangles)

        # la wp.Mesh costruisce la BVH sui triangoli; tiene un riferimento a self.points, quindi
        # basta aggiornare quell'array e chiamare refit()
        self.mesh = wp.Mesh(points=self.points, indices=self.tris)
        self._update_normals()

        self.optics = Optics()
        self.optics.half_fov = math.radians(fov_deg) / 2.0
        self.optics.max_distance = max_distance
        self.optics.light_power = light_range ** 2
        self.optics.ambient = ambient
        self.optics.specular = specular
        self.optics.shininess = shininess
        self.optics.vignette = vignette
        self.optics.noise = noise
        self.optics.texture_scale = texture_scale
        self.optics.texture_seed = int(seed)
        self.lens_offset = lens_offset

        self.frame = 0       # contatore delle immagini: cambia il rumore del sensore a ogni scatto
        self._buffers = {}   # immagini di output gia' allocate, una coppia per risoluzione

    # --- aggiornamento della geometria -------------------------------------------
    def update_mesh(self, points):
        """Nuove posizioni (N, 3) dei nodi, nello stesso ordine di rest_points."""
        self.points.assign(np.asarray(points, dtype=np.float32))   # copia CPU -> GPU (in place)
        self.mesh.refit()                                          # aggiorna le scatole della BVH
        self._update_normals()

    def _update_normals(self):
        self.normals.zero_()
        wp.launch(_accumulate_normals, dim=self.n_tris, inputs=[self.points, self.tris, self.normals],
                  device=self.device)
        wp.launch(_normalize, dim=len(self.normals), inputs=[self.normals], device=self.device)

    # --- scatto ------------------------------------------------------------------
    def camera_frame(self, tip_pose):
        """(origine della lente, avanti, destra, alto) nel sistema globale, dalla posa della punta."""
        pose = np.asarray(tip_pose, dtype=float)
        rot = R.from_quat(pose[3:7])            # SOFA e scipy: quaternione (x, y, z, w)
        fwd = rot.apply([1.0, 0.0, 0.0])        # asse della sonda
        up = rot.apply([0.0, 0.0, 1.0])         # "alto" dell'immagine
        right = np.cross(fwd, up)               # x avanti, z alto -> destra = -y locale
        origin = pose[0:3] + fwd * self.lens_offset
        return origin, fwd, right, up

    def render(self, tip_pose, resolution=64):
        """Scatta un'immagine quadrata resolution x resolution.
        Restituisce (rgb uint8 (H, W, 3), depth float32 (H, W) in metri)."""
        if resolution not in self._buffers:
            self._buffers[resolution] = (
                wp.zeros((resolution, resolution), dtype=wp.vec3, device=self.device),
                wp.zeros((resolution, resolution), dtype=float, device=self.device))
        rgb, depth = self._buffers[resolution]

        origin, fwd, right, up = (wp.vec3(*map(float, v)) for v in self.camera_frame(tip_pose))
        self.frame += 1
        wp.launch(_render, dim=(resolution, resolution),
                  inputs=[self.mesh.id, self.rest, self.tris, self.normals,
                          origin, fwd, right, up, self.optics, self.frame],
                  outputs=[rgb, depth], device=self.device)

        # .numpy() aspetta la fine del kernel e copia GPU -> CPU (64x64x3 float: ~50 kB)
        image = (rgb.numpy() * 255.0 + 0.5).astype(np.uint8)
        return image, depth.numpy()


# =============================================================================
# Utilita' per visualizzare/salvare
# =============================================================================
def depth_to_image(depth, max_distance):
    """Profondita' [m] -> immagine uint8 in scala di grigi: vicino = chiaro, lontano = scuro.
    Scala logaritmica, perche' nel lume le distanze vanno da decimi di mm a vari cm."""
    near = max_distance * 1e-3
    d = np.clip(depth, near, max_distance)
    value = 1.0 - np.log(d / near) / np.log(max_distance / near)
    return (value * 255.0 + 0.5).astype(np.uint8)
