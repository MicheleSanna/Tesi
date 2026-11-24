"""
Ambiente gymnasium per la scena del broncoscopio (scene_maze.py):
reset() -> (obs, info), step(action) -> (obs, reward, terminated, truncated, info), close().

AZIONE (4 valori in [-1, 1], come BronchoscopeController.set_command; sono incrementi per passo)
  [avanzamento, rotazione attorno all'asse, leva su/giu', leva sinistra/destra]
  Con CABLE_WAYS = 2 (broncoscopio standard) l'ultimo valore viene ignorato.

OSSERVAZIONE: si sceglie con OBS_MODE (sotto)
  "state"  -> 18 valori:
              [0:7]   posa della punta: x, y, z [m], qx, qy, qz, qw
              [7:11]  comandi: inserimento [m], rotazione [rad], leva su/giu', leva sx/dx [-1, 1]
              [11:13] spinta [N] e coppia [N*m] della mano sullo strumento (resistenza sentita)
              [13:15] forza di contatto con le pareti [N]: parte distale, resto dello strumento
              [15:18] vettore punta -> GOAL [m] nel sistema di riferimento della punta
  "camera" -> dizionario con due campi, come vedrebbe un endoscopio reale:
              "image"  uint8 (canali, H, W): le ultime FRAME_STACK immagini dell'endoscopio
                       (impilate lungo i canali, la piu' recente per ultima: dalla differenza tra
                       immagini consecutive la rete puo' capire come si sta muovendo)
              "vector" float32 [spinta [N], coppia [N*m], leva su/giu', leva sx/dx]: quello che
                       l'operatore sente con la mano e la posizione delle leve
              Niente posizione ne' goal: l'agente deve orientarsi solo con quello che vede.

RICOMPENSA
  progresso - penalita' di contatto:
    progresso = spostamento della punta lungo +Y in questo passo, in millimetri
    penalita' = contact_monitor.contact_penalty() con i pesi di CONTACT_PENALTY (sotto): forza
                delle pareti sulla parte distale e sullo shaft oltre una tolleranza, e
                opzionalmente lunghezza strisciata lungo le pareti
  Se una singola sfera dello strumento preme piu' di MAX_CONTACT_FORCE l'episodio termina
  (rischio di lesione) con un'ulteriore penalita' UNSAFE_PENALTY.

INFO
  info["tip_pose"]: posa della punta [x, y, z, qx, qy, qz, qw] (per stampe e analisi)
  info["progress"], info["contact_penalty"], info["contact_terms"]: parti del reward
  info["contact_force"], info["tip_contact_force"], info["max_contact_force"]: forze [N]

USO
  python env.py                   -> un episodio con azioni casuali
"""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

import Sofa
import Sofa.Core
import Sofa.Simulation

import scene_maze
from contact_monitor import contact_penalty, unsafe_contact

# fondo del canale di maze4.1 (il canale e' cieco: l'unica apertura e' l'ingresso su -Y)
GOAL = np.array([0.05, 0.51, 0.0]) * scene_maze.MESH_SCALE   # [m]
MAX_STEPS = 500

# --- Penalita' per il contatto con le pareti (vedi contact_monitor.contact_penalty) ----------
# Il progresso vale al massimo ~0.75 mm/passo (TRANSLATION_STEP = meta' raggio): con questi pesi
# premere 0.5 N con la punta costa quanto avanzare di 1 mm.
CONTACT_PENALTY = dict(
    tip_weight=2.0,          # per N di forza sulla parte distale (sezione di curvatura)
    tip_tolerance=0.01,      # [N] forza sulla punta tollerata (appoggio leggero)
    shaft_weight=0.5,        # per N di forza sul resto dello strumento
    shaft_tolerance=0.05,    # [N] lo shaft appoggia per forza nelle curve: si tollera un po'
    length_weight=0.0,       # per mm di strumento a contatto con le pareti (0 = disattivato)
    exponent=1.0,            # 1 = lineare, 2 = penalizza molto di piu' i picchi di forza
)
MAX_CONTACT_FORCE = 1.0      # [N] forza su una sfera oltre la quale l'episodio termina (None = mai)
UNSAFE_PENALTY = 10.0        # penalita' aggiuntiva quando l'episodio termina per forza eccessiva

# --- Osservazione ---------------------------------------------------------------------------
OBS_MODE = "state"     # "state" (12 valori) oppure "camera" (immagini dell'endoscopio + forza)
CAMERA_IMAGE = "gray"  # solo "camera": "gray" (1 canale), "rgb" (3 canali) o "depth" (profondita')
FRAME_STACK = 2        # solo "camera": quante immagini consecutive vede la rete


class ObservationBuilder:
    """Costruisce l'osservazione dal BronchoscopeController (e dall'EndoscopeController in modo "camera").
    Usato sia da MazeEnv sia da view_agent.py, cosi' l'agente vede esattamente la stessa cosa in
    training e in visualizzazione."""

    def __init__(self, mode=None, image=None, stack=None, goal=GOAL):
        # None = valori in cima a questo file (letti adesso, cosi' si possono cambiare anche da
        # un altro script con env.OBS_MODE = "camera" prima di creare l'ambiente)
        mode = mode or OBS_MODE
        image = image or CAMERA_IMAGE
        stack = stack or FRAME_STACK
        if mode not in ("state", "camera"):
            raise ValueError(f"OBS_MODE sconosciuto: {mode!r} (usa 'state' o 'camera')")
        if image not in ("gray", "rgb", "depth"):
            raise ValueError(f"CAMERA_IMAGE sconosciuto: {image!r} (usa 'gray', 'rgb' o 'depth')")
        self.mode = mode
        self.image = image
        self.goal = np.asarray(goal, dtype=float)
        self.frames = deque(maxlen=stack)   # ultime immagini (si svuota a ogni reset)

        if mode == "state":
            self.space = spaces.Box(-np.inf, np.inf, (18,), dtype=np.float32)
        else:
            channels = (3 if image == "rgb" else 1) * stack
            size = scene_maze.CAMERA_RESOLUTION
            self.space = spaces.Dict({
                "image": spaces.Box(0, 255, (channels, size, size), dtype=np.uint8),
                "vector": spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
            })

    def reset(self):
        """Nuovo episodio: dimentica le immagini precedenti."""
        self.frames.clear()

    def __call__(self, probe_ctrl, camera_ctrl=None, contacts=None):
        """Osservazione dello stato attuale. probe_ctrl: BronchoscopeController; camera_ctrl:
        EndoscopeController (solo modo "camera"); contacts: ContactMonitor (solo modo "state",
        None = forze di contatto a zero)."""
        if self.mode == "state":
            return observation(probe_ctrl, self.goal, contacts)

        rgb, depth = camera_ctrl.capture()
        frame = self._frame(rgb, depth, camera_ctrl.camera.optics.max_distance)
        if not self.frames:
            # primo passo dell'episodio: non ci sono immagini precedenti, si ripete la prima
            self.frames.extend([frame] * self.frames.maxlen)
        else:
            self.frames.append(frame)
        force, torque = probe_ctrl.get_coupling_load()
        _, _, levers = probe_ctrl.get_controls()
        return {"image": np.concatenate(list(self.frames), axis=0),   # (canali, H, W)
                "vector": np.array([force, torque, levers[0], levers[1]], dtype=np.float32)}

    def _frame(self, rgb, depth, max_distance):
        """Un'immagine (H, W, 3) -> (canali, H, W) uint8 nel formato scelto."""
        if self.image == "rgb":
            return np.ascontiguousarray(rgb.transpose(2, 0, 1))   # torch vuole i canali per primi
        if self.image == "gray":
            # luminanza percepita (pesi standard ITU-R BT.601)
            gray = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            return (gray + 0.5).astype(np.uint8)[None]
        from endoscope_camera import depth_to_image
        return depth_to_image(depth, max_distance)[None]


class MazeEnv(gym.Env):
    """Ambiente RL: l'agente comanda il broncoscopio (inserimento, rotazione, leve) e viene
    premiato per l'avanzamento della punta e penalizzato per il contatto con le pareti."""

    metadata = {"render_modes": []}

    def __init__(self, goal=GOAL, max_steps=MAX_STEPS, obs_mode=None, contact_penalty=None,
                 max_contact_force=MAX_CONTACT_FORCE):
        """contact_penalty: pesi per contact_monitor.contact_penalty (None = CONTACT_PENALTY);
        max_contact_force: [N] forza oltre la quale l'episodio termina (None = mai)."""
        self.max_steps = max_steps
        self.obs = ObservationBuilder(mode=obs_mode, goal=goal)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.observation_space = self.obs.space

        self.root = Sofa.Core.Node("root")
        # la telecamera si crea solo se serve (Warp e il raycasting costano qualche secondo all'avvio)
        scene_maze.createScene(self.root, with_visual=False, with_camera=self.obs.mode == "camera")
        Sofa.Simulation.init(self.root)
        self.ctrl = self.root.getObject("ScopeController")
        self.ctrl.print_every = 0     # niente stampe a terminale durante il training
        self.contacts = self.root.getObject("ContactMonitor")
        self.penalty_weights = dict(CONTACT_PENALTY if contact_penalty is None else contact_penalty)
        self.max_contact_force = max_contact_force
        # telecamera (solo in modo "camera"; in modo "state" non esiste)
        self.camera = self.root.getObject("EndoscopeController") if self.obs.mode == "camera" else None

        self.steps = 0
        self.prev_y = 0.0

    def reset(self, seed=None, options=None):
        """Riporta tessuto, strumento, mano e leve allo stato iniziale."""
        super().reset(seed=seed)   # inizializza self.np_random
        if seed is not None:
            self.action_space.seed(seed)
        Sofa.Simulation.reset(self.root)   # riporta i nodi (tessuto, strumento, mano) all'inizio
        self.ctrl.reset_state()            # ... e lo stato del controller (inserimento, leve, cavi)
        self.contacts.reset()
        self.obs.reset()
        self.steps = 0
        pose = self.ctrl.get_tip_pose()
        self.prev_y = pose[1]
        return self.obs(self.ctrl, self.camera, self.contacts), {"tip_pose": pose}

    def step(self, action):
        """Applica l'azione per un passo di simulazione e calcola il reward."""
        self.ctrl.set_command(*np.clip(np.asarray(action, dtype=float), -1.0, 1.0))
        Sofa.Simulation.animate(self.root, self.root.dt.value)
        self.steps += 1

        obs = self.obs(self.ctrl, self.camera, self.contacts)
        pose = self.ctrl.get_tip_pose()
        progress = (pose[1] - self.prev_y) * 1000.0
        self.prev_y = pose[1]

        contact = self.contacts.last
        penalty, terms = contact_penalty(contact, **self.penalty_weights)
        reward = progress - penalty

        unsafe = unsafe_contact(contact, self.max_contact_force)
        if unsafe:
            reward -= UNSAFE_PENALTY
        diverged = not np.all(np.isfinite(pose))   # simulazione divergente
        terminated = bool(diverged or unsafe)
        truncated = self.steps >= self.max_steps
        info = {"tip_pose": pose, "progress": progress, "contact_penalty": penalty,
                "contact_terms": terms, "contact_force": contact.total_force,
                "tip_contact_force": contact.tip_force, "max_contact_force": contact.max_force,
                "unsafe_contact": unsafe}
        return obs, float(reward), terminated, truncated, info

    def close(self):
        """Libera la scena SOFA."""
        Sofa.Simulation.unload(self.root)


def observation(ctrl, goal, contacts=None):
    """Osservazione "state" a 18 valori (vedi la docstring del modulo).
    ctrl: BronchoscopeController; contacts: ContactMonitor (None = forze di contatto a zero)."""
    pose = ctrl.get_tip_pose()
    insertion, roll, levers = ctrl.get_controls()
    force, torque = ctrl.get_coupling_load()
    info = contacts.last if contacts is not None else None
    contact = [info.tip_force, info.shaft_force] if info is not None else [0.0, 0.0]
    goal_rel = R.from_quat(pose[3:7]).inv().apply(goal - pose[0:3])
    return np.concatenate([pose, [insertion, roll], levers, [force, torque], contact,
                           goal_rel]).astype(np.float32)


if __name__ == "__main__":
    env = MazeEnv(max_steps=100)
    obs, info = env.reset(seed=0)
    total = 0.0
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        done = terminated or truncated
    shape = {k: v.shape for k, v in obs.items()} if isinstance(obs, dict) else obs
    print(f"passi: {env.steps}  ricompensa totale: {total:.3f}  osservazione finale: {shape}")
    print(f"ultimo passo: progresso {info['progress']:.3f} mm, penalita' contatto "
          f"{info['contact_penalty']:.3f}, forza sulle pareti {info['contact_force']:.3f} N")
    env.close()
