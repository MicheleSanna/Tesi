import Sofa
import Sofa.Core
import Sofa.Simulation
import numpy as np
import atexit
import math
import os
import shutil
import struct
import subprocess
import sys
import threading
from scipy.spatial.transform import Rotation as R

from bronchoscope import LEVERS


class BronchoscopeController(Sofa.Core.Controller):
    """Comandi del broncoscopio (bronchoscope.py), gli stessi che ha un operatore:

      avanzamento   la mano spinge/tira la base dello strumento lungo l'asse del tubo introduttore
      rotazione     la mano ruota lo strumento attorno al proprio asse (cosi' si sceglie in che
                    direzione si piega la punta)
      leva su/giu'  la leva del manico accorcia il cavo "su" e allunga di altrettanto il cavo
                    "giu'" (puleggia), o viceversa: la punta si piega
      leva sx/dx    seconda leva, solo con 4 cavi (broncoscopi robotici)

    Da tastiera in runSofa oppure da codice con set_command() (RL). I comandi sono INCREMENTI:
    con un tasto premuto (o un comando costante) lo strumento avanza, ruota o si piega a velocita'
    costante, come quando un operatore spinge o sposta la leva con continuita'.

    La mano e' collegata alla base da una molla: se lo strumento e' bloccato la mano non puo'
    allontanarsi dalla base piu' di max_push_offset / max_roll_offset_deg, quindi la forza di
    spinta e la coppia di rotazione non superano mai i massimi di ScopeParams (l'operatore
    "sente" la resistenza e non forza oltre)."""

    KEYMAP = {  # (avanzamento, rotazione, leva su/giu', leva sinistra/destra)
        "W": (+1, 0, 0, 0), "S": (-1, 0, 0, 0),   # avanti / indietro
        "A": (0, +1, 0, 0), "D": (0, -1, 0, 0),   # ruota antiorario / orario (visto dalla mano)
        "R": (0, 0, +1, 0), "F": (0, 0, -1, 0),   # punta su / giu'
        "Q": (0, 0, 0, +1), "E": (0, 0, 0, -1),   # punta a sinistra / destra (solo 4 cavi)
    }

    def __init__(self, scope, up, trans_step, rot_step, bend_step, max_insertion=None,
                 max_roll=math.pi, labels=None, print_every=0, keyboard=None, *args, **kwargs):
        """scope: bronchoscope.Bronchoscope. trans_step [m], rot_step [rad] e bend_step (frazione
        della corsa della leva) sono gli incrementi per passo con comando = 1. max_insertion [m]:
        inserimento massimo (None = finche' la base resta nel tubo introduttore); max_roll [rad]:
        rotazione massima in ciascun verso (il polso dell'operatore non gira all'infinito)."""
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.scope = scope
        self.params = scope.params
        self.dofs = scope.dofs                  # nodi dello strumento (0 = base, -1 = punta)
        self.hand = scope.hand_dofs             # mano (bersaglio della molla)
        self.axis = np.asarray(scope.axis, dtype=float)
        self.up = np.asarray(up, dtype=float)
        self.hand_start = np.array(scope.start_poses[0], dtype=float)
        self.rot0 = R.from_quat(self.hand_start[3:7])
        self.n_levers = self.params.cable_ways // 2
        self.trans_step = trans_step
        self.rot_step = rot_step
        self.bend_step = bend_step
        p = self.params
        # la base (dove la tiene la mano) deve restare fuori dal paziente, dentro il tubo
        # introduttore: al massimo si inserisce tutto tranne la sezione di curvatura e un margine
        self.max_insertion = (max_insertion if max_insertion is not None
                              else p.length - p.bending_length - 2.0 * p.radius)
        self.max_roll = max_roll

        # stato dei comandi (la "mano" e le leve)
        self.insertion = 0.0                    # [m] avanzamento della mano dalla posa iniziale
        self.roll = 0.0                         # [rad] rotazione della mano attorno all'asse
        self.levers = np.zeros(2)               # posizione delle leve in [-1, 1]

        self.command = np.zeros(4)              # comando one-shot da codice (set_command)
        self.keyboard = keyboard                # WindowsKeyboard / LinuxKeyboard, o None (RL)
        self.held_keys = set()                  # tasti attualmente premuti in runSofa
        self.x_was_down = False
        self.reset_requested = False
        self.labels = labels or []              # OglLabel a schermo (solo con interfaccia grafica)
        self.print_every = print_every
        self.step_count = 0

        # assi di riferimento orizzontali per get_tip_angles(): yaw = 0 quando la punta guarda +X
        self.ref_fwd = np.array([1.0, 0.0, 0.0])
        self.ref_left = np.cross(self.up, self.ref_fwd)

    # --- API per codice esterno / RL -----------------------------------------
    def set_command(self, advance=0.0, rotate=0.0, bend_ud=0.0, bend_lr=0.0):
        """Comando per il PROSSIMO passo, ogni valore in [-1, 1] (scalato dagli step).
        Con 2 cavi bend_lr viene ignorato."""
        self.command[:] = (advance, rotate, bend_ud, bend_lr)

    def get_tip_pose(self):
        """Posa della punta (ultimo nodo): [x, y, z, qx, qy, qz, qw]. Asse locale +X = direzione
        in cui guarda la telecamera, +Z = alto dell'immagine."""
        return np.array(self.dofs.position.value[-1], dtype=float)

    def get_base_pose(self):
        """Posa della base dello strumento (dove lo tiene la mano)."""
        return np.array(self.dofs.position.value[0], dtype=float)

    def get_target_pose(self):
        """Posa comandata, cioe' quella della mano."""
        return np.array(self.hand.position.value[0], dtype=float)

    def get_centerline(self):
        """Posizioni (N, 3) dei nodi lungo l'asse dello strumento, dalla base alla punta."""
        return np.array(self.dofs.position.value, dtype=float)[:, 0:3]

    def get_coupling_load(self):
        """(forza di spinta [N], coppia di rotazione [N*m]) che la mano esercita sullo strumento,
        con segno: forza > 0 = spinge in avanti, coppia > 0 = nel verso di rotazione positivo.
        Quando lo strumento e' fermo e' la resistenza che oppone (pareti, attrito, introduttore)."""
        axial, twist = self._hand_offsets()
        return (self.params.hand_stiffness * axial, self.params.hand_angular_stiffness * twist)

    def get_cable_tensions(self):
        """Tensione di ogni cavo [N] (0 = lasco). CableConstraint restituisce un impulso (la
        grandezza risolta dal solutore dei vincoli in un passo), quindi si divide per dt."""
        dt = self.getContext().dt.value
        return {name: float(cable.force.value) / dt for name, cable in self.scope.cables.items()}

    def get_controls(self):
        """Stato dei comandi: inserimento [m], rotazione [rad], leve [-1, 1] (su/giu', sx/dx)."""
        return self.insertion, self.roll, self.levers.copy()

    def get_tip_angles(self):
        """(yaw, pitch, roll) della punta in gradi.
        yaw: rotazione attorno all'asse verticale (0 = verso +X); pitch: > 0 punta verso l'alto;
        roll: rotazione attorno all'asse della punta."""
        rot = R.from_quat(self.get_tip_pose()[3:7])
        fwd = rot.apply([1.0, 0.0, 0.0])
        yaw = math.atan2(fwd @ self.ref_left, fwd @ self.ref_fwd)
        pitch = math.asin(float(np.clip(fwd @ self.up, -1.0, 1.0)))

        # roll: angolo tra l'asse Y locale e l'asse "sinistra" orizzontale
        left = np.cross(self.up, fwd)
        n = np.linalg.norm(left)
        if n > 1e-6:
            left /= n
            y_body = rot.apply([0.0, 1.0, 0.0])
            roll = math.atan2(np.cross(left, y_body) @ fwd, left @ y_body)
        else:
            roll = 0.0  # punta verticale: roll e yaw non sono distinguibili
        return np.degrees([yaw, pitch, roll])

    def pose_text(self):
        """Righe di testo con lo stato dello strumento (a schermo e nel terminale)."""
        p = self.get_tip_pose()
        yaw, pitch, roll = self.get_tip_angles()
        mm = p[0:3] * 1000.0
        force, torque = self.get_coupling_load()
        tensions = "  ".join(f"{name} {t:5.2f}" for name, t in self.get_cable_tensions().items())
        levers = f"leva su/giu' {self.levers[0]:+.2f}"
        if self.n_levers == 2:
            levers += f"  sx/dx {self.levers[1]:+.2f}"
        return (f"punta [mm]: x={mm[0]:+8.2f}  y={mm[1]:+8.2f}  z={mm[2]:+8.2f}   "
                f"[deg] yaw={yaw:+7.2f}  pitch={pitch:+7.2f}  roll={roll:+7.2f}",
                f"inserimento {self.insertion * 1000:6.1f} mm  rotazione {math.degrees(self.roll):+7.1f} deg  "
                + levers,
                f"spinta {force:+6.3f} N  coppia {torque * 1000:+6.2f} mN*m",
                f"cavi [N]: {tensions}")

    def reset_state(self):
        """Riporta a zero comandi, leve e cavi e la mano nella posa iniziale. Da chiamare dopo
        Sofa.Simulation.reset() (che riporta indietro i nodi ma non lo stato di questo controller)."""
        self.insertion, self.roll = 0.0, 0.0
        self.levers[:] = 0.0
        self.command[:] = 0.0
        self._write_hand()
        self._write_cables()

    def reset_scope(self):
        """Reset completo dello strumento (tasto X): comandi a zero, nodi dritti nella posa
        iniziale e fermi. Il tessuto non viene toccato."""
        self.reset_state()
        start = self.scope.start_poses
        for data in (self.dofs.position, self.dofs.free_position):
            with data.writeableArray() as pos:
                pos[:] = start
        for data in (self.dofs.velocity, self.dofs.free_velocity):
            with data.writeableArray() as vel:
                vel[:] = 0.0
        self.scope.update_guide()

    # --- Mano e cavi --------------------------------------------------------------
    def _hand_offsets(self):
        """(scostamento assiale [m], scostamento di rotazione attorno all'asse [rad]) mano - base.
        Sono piccoli per costruzione (li limita _apply_limits), quindi la rotazione relativa non
        ha i salti di +-180 gradi che avrebbe un angolo assoluto."""
        base, hand = self.get_base_pose(), self.get_target_pose()
        axial = float((hand[0:3] - base[0:3]) @ self.axis)
        rel = (R.from_quat(hand[3:7]) * R.from_quat(base[3:7]).inv()).as_rotvec()
        return axial, float(rel @ self.axis)

    def _apply_limits(self):
        """Tiene la mano entro lo scostamento massimo dalla base (limite di forza e coppia),
        dentro la corsa consentita (tubo introduttore) e dentro la rotazione massima."""
        axial, twist = self._hand_offsets()
        base_insertion = self.insertion - axial     # dove si trova davvero la base
        base_roll = self.roll - twist
        p = self.params
        max_twist = math.radians(p.max_roll_offset_deg)
        self.insertion = float(np.clip(self.insertion, base_insertion - p.max_push_offset,
                                       base_insertion + p.max_push_offset))
        self.roll = float(np.clip(self.roll, base_roll - max_twist, base_roll + max_twist))
        self.insertion = float(np.clip(self.insertion, 0.0, self.max_insertion))
        self.roll = float(np.clip(self.roll, -self.max_roll, self.max_roll))

    def _write_hand(self):
        """Scrive la posa della mano: traslata lungo l'asse e ruotata attorno all'asse.
        Si aggiorna anche free_position, che FreeMotionAnimationLoop usa per il passo libero."""
        pose = self.hand_start.copy()
        pose[0:3] += self.axis * self.insertion
        pose[3:7] = (R.from_rotvec(self.axis * self.roll) * self.rot0).as_quat()
        for data in (self.hand.position, self.hand.free_position):
            if len(data.value) != 1:
                # dopo Sofa.Simulation.reset() free_position della mano (che non ha solutore)
                # e' vuoto: si riassegna per intero
                data.value = [pose.tolist()]
                continue
            with data.writeableArray() as pos:
                pos[0] = pose

    def _write_cables(self):
        """Leve -> accorciamento imposto ai cavi. Puleggia: leva > 0 accorcia di delta il primo
        cavo della coppia e allunga di delta il secondo (che resta lasco se non serve).
        Con il gioco g: accorciamento = leva * (corsa + g) - g, cioe' a leva neutra entrambi i
        cavi possono allungarsi di g senza tirare, e a fondo corsa si ha comunque la curvatura
        massima."""
        stroke = self.params.max_cable_displacement
        slack = self.params.cable_slack
        for lever, (positive, negative) in zip(self.levers[:self.n_levers], LEVERS):
            self.scope.cables[positive].value = [float(lever * (stroke + slack) - slack)]
            self.scope.cables[negative].value = [float(-lever * (stroke + slack) - slack)]

    # --- Tastiera -----------------------------------------------------------------
    def _poll_keyboard(self):
        """Aggiorna held_keys leggendo i tasti premuti in questo momento (solo con runSofa in primo piano)."""
        if self.keyboard is None:
            return
        focused = self.keyboard.has_focus()
        pressed = {k for k in self.KEYMAP if focused and self.keyboard.is_down(k)}
        for key in sorted(pressed - self.held_keys):
            print(f"[broncoscopio] tasto {key} premuto")
        self.held_keys = pressed

        x_down = focused and self.keyboard.is_down("X")
        if x_down and not self.x_was_down:   # solo sul fronte di pressione, non a ogni passo
            self.reset_requested = True
            print("[broncoscopio] reset alla posizione iniziale")
        self.x_was_down = x_down

    # --- Eventi SOFA ------------------------------------------------------------
    def onAnimateBeginEvent(self, event):
        """Inizio passo: legge i comandi (codice + tastiera), muove mano e leve, aggiorna i cavi."""
        self._poll_keyboard()
        if self.reset_requested:
            self.reset_scope()
            self.reset_requested = False
        command = self.command + sum((np.array(self.KEYMAP[k], dtype=float) for k in self.held_keys),
                                     np.zeros(4))
        self.command[:] = 0.0
        advance, rotate, bend_ud, bend_lr = np.clip(command, -1.0, 1.0)

        self.insertion += advance * self.trans_step
        self.roll += rotate * self.rot_step
        self.levers += np.array([bend_ud, bend_lr]) * self.bend_step
        self.levers[self.n_levers:] = 0.0           # con 2 cavi la seconda leva non esiste
        self.levers = np.clip(self.levers, -1.0, 1.0)

        self._apply_limits()
        self._write_hand()
        self._write_cables()
        self.scope.update_guide()       # nodi entrati / usciti dal tubo introduttore

    def onAnimateEndEvent(self, event):
        """Fine passo: aggiorna il testo a schermo e, ogni print_every passi, il terminale."""
        self.step_count += 1
        lines = None
        if self.labels:
            lines = self.pose_text()
            for label, text in zip(self.labels, lines):
                label.label.value = text
        if self.print_every and self.step_count % self.print_every == 0:
            lines = lines or self.pose_text()
            t = self.getContext().getTime()
            print(f"[broncoscopio] passo {self.step_count}  t={t:.2f}s  " + "  |  ".join(lines))


class EndoscopeController(Sofa.Core.Controller):
    """Telecamera endoscopica sulla punta della sonda (vedi endoscope_camera.py).

    - capture(): scatta un'immagine dello stato attuale (la usano l'ambiente RL e view_agent.py).
    - Con l'interfaccia grafica, alla fine di ogni passo mostra in una finestra separata:
        [ anteprima ad alta risoluzione | cio' che vede l'agente RL | profondita' ]
      La finestra e' un altro processo (endoscope_viewer.py), perche' la finestra di OpenCV nello
      stesso processo di runSofa lo manda in crash (conflitto tra le librerie grafiche).
    - Se save_dir e' impostata, salva la stessa composizione in PNG ogni save_every passi
      (utile anche senza GUI). Se la finestra non si puo' aprire, si passa da soli al salvataggio.

    I tasti per muovere la sonda vanno premuti con la finestra di runSofa in primo piano, non con
    quella dell'endoscopio.

    Resa neurale (neural != None, vedi neural_render.py): l'immagine RGB non e' quella procedurale
    ma quella generata dal modello broncho-sim2real a partire dalla profondita'. La vede l'agente
    RL e compare nella finestra accanto a quella procedurale, per confronto. Con la GUI la rete gira
    in un thread a parte (la finestra mostra l'ultima immagine pronta, con qualche passo di
    ritardo); nel training invece ogni passo aspetta la sua immagine."""

    def __init__(self, camera, maze_dofs, probe_ctrl, resolution=64, show_window=False,
                 preview_resolution=320, save_dir=None, save_every=0, neural=None, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.camera = camera                # EndoscopeCamera
        self.neural = neural                # NeuralBronchoscopeRenderer, o None (solo procedurale)
        self.last_neural = None             # ultima immagine neurale (risoluzione della rete)
        self._neural_waiting_reported = False
        self.maze_dofs = maze_dofs          # MechanicalObject delle pareti (tessuto deformato o
                                            # superficie rigida, anche in movimento)
        self.probe_ctrl = probe_ctrl        # BronchoscopeController (posa della punta)
        self.resolution = resolution        # immagine dell'agente RL (quadrata)
        self.preview_resolution = preview_resolution
        self.show_window = show_window
        self.save_dir = save_dir
        self.save_every = save_every
        self.step_count = 0
        self.last_rgb = None                # ultima immagine dell'agente (uint8, H x W x 3)
        self.last_depth = None              # ultima profondita' (float32, H x W, metri)
        self._cv2 = None
        self._viewer = None                 # processo endoscope_viewer.py
        self._pending = None                # ultima immagine non ancora inviata alla finestra
        self._cond = threading.Condition()  # sveglia il thread che invia le immagini

    # --- API ---------------------------------------------------------------------
    def sync_mesh(self):
        """Copia nella telecamera le posizioni attuali dei nodi del labirinto.
        Con CUDA (CudaVec3f) .value copia i dati dalla GPU di SOFA alla CPU: ~0.4 MB con 30k nodi."""
        self.camera.update_mesh(np.asarray(self.maze_dofs.position.value))

    def capture(self, resolution=None, sync=True):
        """Scatta un'immagine dalla punta della sonda: (rgb uint8 (H, W, 3), depth float32 (H, W)).
        Con la resa neurale attiva, rgb e' l'immagine della rete (ridotta a `resolution`)."""
        if sync:
            self.sync_mesh()
        resolution = resolution or self.resolution
        pose = self.probe_ctrl.get_tip_pose()
        rgb, depth = self.camera.render(pose, resolution)   # procedurale + profondita'
        if self.neural is not None:
            neural_rgb = self._neural_image(pose, resolution)
            if neural_rgb is not None:
                rgb = neural_rgb
        if resolution == self.resolution:
            self.last_rgb, self.last_depth = rgb, depth
        return rgb, depth

    def _neural_image(self, pose, resolution):
        """Immagine della rete neurale alla risoluzione richiesta (None se non ancora pronta)."""
        from neural_render import apply_circle, fisheye_valid_mask
        size = self.neural.resolution
        # la rete vuole la profondita' alla sua risoluzione (256 px): la telecamera la ricalcola
        # a quella risoluzione (~1 ms), invece di ingrandire quella piccola dell'agente
        _, depth = self.camera.render(pose, size)
        valid = fisheye_valid_mask(depth, self.camera.optics.max_distance)
        if self.neural.asynchronous:
            self.neural.submit(depth, valid)     # non blocca: la rete lavora in un altro thread
            image = self.neural.latest()
            if image is None:
                if not self._neural_waiting_reported:
                    print("[endoscopio] rete neurale in preparazione: per ora immagine procedurale")
                    self._neural_waiting_reported = True
                return None
        else:
            image = self.neural.render(depth, valid)   # aspetta l'immagine (training)
        image = apply_circle(image)   # stessi angoli neri dell'ottica fisheye procedurale
        self.last_neural = image
        if resolution == size:
            return image
        cv2 = self._import_cv2()
        # INTER_AREA: riduzione che media i pixel (niente aliasing)
        return cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)

    # --- Evento SOFA: fine di ogni passo -----------------------------------------
    def onAnimateEndEvent(self, event):
        self.step_count += 1
        saving = bool(self.save_dir) and self.save_every > 0 and self.step_count % self.save_every == 0
        if not (self.show_window or saving):
            return   # in training (headless) le immagini le chiede l'ambiente con capture()

        rgb, depth = self.capture()
        # anteprima procedurale ad alta risoluzione (stessa geometria, piu' pixel)
        preview, _ = self.camera.render(self.probe_ctrl.get_tip_pose(), self.preview_resolution)
        panel = self._compose(preview, rgb, depth)

        if self.show_window:
            self._show(panel)
        if saving:
            self._save(panel)

    # --- Composizione dell'immagine ------------------------------------------------
    def _compose(self, preview, rgb, depth):
        """Affianca anteprima, vista dell'agente e profondita', tutte alla dimensione dell'anteprima.
        Restituisce un'immagine BGR (l'ordine dei canali che usa OpenCV)."""
        cv2 = self._import_cv2()
        size = preview.shape[0]
        tiles = []   # (immagine BGR size x size, etichetta)
        if self.neural is not None:
            # immagine neurale ingrandita con interpolazione (e' gia' a 256 px); nera finche' la rete
            # non ha prodotto la prima immagine (compilazione)
            if self.last_neural is not None:
                neural = cv2.resize(self.last_neural, (size, size), interpolation=cv2.INTER_LINEAR)
                ms = f", {self.neural.last_time * 1000:.0f} ms" if self.neural.last_time else ""
                label = f"neurale ({self.neural.steps} passi{ms})"
            else:
                neural, label = np.zeros((size, size, 3), np.uint8), "neurale (in preparazione)"
            tiles.append((cv2.cvtColor(neural, cv2.COLOR_RGB2BGR), label))
        tiles.append((cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
                      "procedurale" if self.neural is not None else "anteprima"))
        # vista dell'agente ingrandita senza interpolazione: si vedono i pixel veri della rete
        agent = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_NEAREST)
        tiles.append((cv2.cvtColor(agent, cv2.COLOR_RGB2BGR), f"agente {self.resolution}x{self.resolution}"))
        from endoscope_camera import depth_to_image
        d = depth_to_image(depth, self.camera.optics.max_distance)
        d = cv2.resize(cv2.applyColorMap(d, cv2.COLORMAP_INFERNO), (size, size),
                       interpolation=cv2.INTER_NEAREST)
        tiles.append((d, "profondita'"))

        panel = np.hstack([tile for tile, _ in tiles])
        for i, (_, text) in enumerate(tiles):
            cv2.putText(panel, text, (i * size + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        force, torque = self.probe_ctrl.get_coupling_load()
        cv2.putText(panel, f"forza {force:6.2f} N", ((len(tiles) - 1) * size + 8, size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return panel

    # --- Finestra (processo separato) ------------------------------------------------
    def _show(self, panel):
        if self._viewer is None:
            self._start_viewer()
            if self._viewer is None:
                return
        code = self._viewer.poll()
        if code is not None:   # il processo della finestra e' terminato
            self._stop_viewer()
            self.show_window = False
            if code == 0:
                print("[endoscopio] finestra chiusa")
            else:
                self._fallback_to_saving(f"il visualizzatore e' terminato con codice {code}")
            return
        # si lascia l'immagine al thread di invio; se quello e' ancora occupato con la precedente,
        # la precedente viene semplicemente scartata (la simulazione non aspetta mai la finestra)
        with self._cond:
            self._pending = panel
            self._cond.notify()

    def _start_viewer(self):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "endoscope_viewer.py")
        # Qt di OpenCV non ha il plugin per Wayland: si usa X11 (XWayland) per non avere avvisi
        env = dict(os.environ, QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "xcb"))
        try:
            self._viewer = subprocess.Popen([_python_executable(), script], stdin=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, env=env)
        except OSError as e:
            self.show_window = False
            self._fallback_to_saving(f"impossibile avviare il visualizzatore ({e})")
            return
        threading.Thread(target=self._send_loop, daemon=True).start()
        atexit.register(self._stop_viewer)   # alla fine di runSofa la finestra si chiude da sola

    def _send_loop(self):
        """Thread che invia le immagini al visualizzatore (scrivere nella pipe puo' bloccare)."""
        viewer = self._viewer
        while True:
            with self._cond:
                while self._pending is None:
                    self._cond.wait()
                panel, self._pending = self._pending, None
            if panel is _STOP:
                return
            header = struct.pack("<II", panel.shape[0], panel.shape[1])   # altezza, larghezza
            try:
                viewer.stdin.write(header + np.ascontiguousarray(panel).tobytes())
                viewer.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                return   # finestra chiusa: se ne accorge _show() con poll()

    def _stop_viewer(self):
        if self._viewer is None:
            return
        with self._cond:
            self._pending = _STOP
            self._cond.notify()
        try:
            self._viewer.stdin.close()   # il visualizzatore legge "fine" e si chiude
        except OSError:
            pass
        self._viewer = None

    # --- Salvataggio su disco ----------------------------------------------------------
    def _fallback_to_saving(self, reason):
        self.save_dir = self.save_dir or "endoscope_frames"
        self.save_every = self.save_every or 1
        print(f"[endoscopio] finestra non disponibile: {reason}. "
              f"Salvo le immagini in {os.path.abspath(self.save_dir)}")

    def _save(self, panel):
        cv2 = self._import_cv2()
        os.makedirs(self.save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.save_dir, f"endoscopio_{self.step_count:06d}.png"), panel)

    def _import_cv2(self):
        # import ritardato: OpenCV serve solo per visualizzare/salvare, non per il training.
        # In questo processo si usano solo le funzioni di elaborazione (resize, testo, PNG), mai
        # quelle delle finestre (imshow), che andrebbero in conflitto con la GUI di SOFA.
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        return self._cv2


_STOP = object()   # segnale per fermare il thread di invio


def _python_executable():
    """L'interprete Python dell'ambiente conda. Dentro runSofa sys.executable e' runSofa stesso,
    quindi lo si cerca in sys.prefix (la cartella dell'ambiente) o nel PATH."""
    if os.path.basename(sys.executable).lower().startswith("python"):
        return sys.executable
    for candidate in (os.path.join(sys.prefix, "bin", "python3"), os.path.join(sys.prefix, "python.exe"),
                      shutil.which("python3"), shutil.which("python")):
        if candidate and os.path.isfile(candidate):
            return candidate
    raise OSError("interprete Python non trovato")
