import Sofa
import Sofa.Core
import Sofa.Simulation
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

class ProbeController(Sofa.Core.Controller):
    """Muove la sonda. Da tastiera in runSofa, oppure da codice con set_command() (per l'RL).

    In modalita' "force" i comandi spostano il bersaglio (target_dofs) e la sonda lo segue tirata
    da una molla; in modalita' "kinematic" (target_dofs=None) spostano direttamente la sonda."""

    KEYMAP = {  # (avanzamento, yaw, pitch, sollevamento)
        "W": (+1, 0, 0, 0), "S": (-1, 0, 0, 0),
        "A": (0, +1, 0, 0), "D": (0, -1, 0, 0),
        "R": (0, 0, +1, 0), "F": (0, 0, -1, 0),
        "Q": (0, 0, 0, +1), "E": (0, 0, 0, -1),
    }

    def __init__(self, probe_dofs, up, trans_step, rot_step, target_dofs=None, coupling=None,
                 col_dofs=None, col_local=None, labels=None, print_every=0, keyboard=None,
                 *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.dofs = probe_dofs
        self.target = target_dofs       # bersaglio della molla (solo modalita' "force")
        self.coupling = coupling        # dict: stiffness, angular_stiffness, max_offset, max_angle
        self.col_dofs = col_dofs        # sfere di collisione (figlie della sonda via RigidMapping)
        self.col_local = np.asarray(col_local, dtype=float) if col_local is not None else None
        self.up = np.asarray(up, dtype=float)
        self.trans_step = trans_step
        self.rot_step = rot_step
        self.command = np.zeros(4)      # comando one-shot da codice (set_command)
        self.keyboard = keyboard        # WindowsKeyboard, o None (headless / RL)
        self.held_keys = set()          # tasti attualmente premuti in runSofa
        self.x_was_down = False
        self.start_pose = np.array(probe_dofs.position.value[0], dtype=float)
        self.reset_requested = False
        self.labels = labels or []      # OglLabel a schermo (solo con interfaccia grafica)
        self.print_every = print_every
        self.step_count = 0

        # assi di riferimento orizzontali: yaw = 0 quando la sonda punta verso +X
        self.ref_fwd = np.array([1.0, 0.0, 0.0])
        self.ref_left = np.cross(self.up, self.ref_fwd)

    # --- API per codice esterno / RL -----------------------------------------
    def set_command(self, advance=0.0, yaw=0.0, pitch=0.0, lift=0.0):
        """Comando per il PROSSIMO passo, ogni valore in [-1, 1] (scalato dagli step)."""
        self.command[:] = (advance, yaw, pitch, lift)

    def get_tip_pose(self):
        """Posa (reale) della punta della sonda: [x, y, z, qx, qy, qz, qw]."""
        return np.array(self.dofs.position.value[0], dtype=float)

    def get_target_pose(self):
        """Posa comandata: il bersaglio in modalita' "force", la sonda stessa in "kinematic"."""
        dofs = self.target if self.target is not None else self.dofs
        return np.array(dofs.position.value[0], dtype=float)

    def get_coupling_load(self):
        """(forza [N], coppia [N*m]) che la molla esercita sulla sonda (0, 0 in "kinematic").
        Quando la sonda e' ferma e' uguale alla forza/coppia con cui la sonda spinge le pareti."""
        if self.target is None:
            return 0.0, 0.0
        p, t = self.get_tip_pose(), self.get_target_pose()
        offset = np.linalg.norm(t[0:3] - p[0:3])
        angle = np.linalg.norm((R.from_quat(t[3:7]) * R.from_quat(p[3:7]).inv()).as_rotvec())
        return (self.coupling["stiffness"] * offset,
                self.coupling["angular_stiffness"] * angle)

    def get_tip_angles(self):
        """(yaw, pitch, roll) della sonda in gradi.
        yaw: rotazione attorno all'asse verticale (0 = verso +X); pitch: > 0 punta verso l'alto;
        roll: rotazione attorno all'asse della sonda."""
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
            roll = 0.0  # sonda verticale: roll e yaw non sono distinguibili
        return np.degrees([yaw, pitch, roll])

    def pose_text(self):
        p = self.get_tip_pose()
        yaw, pitch, roll = self.get_tip_angles()
        mm = p[0:3] * 1000.0
        lines = (f"pos [mm]: x={mm[0]:+8.2f}  y={mm[1]:+8.2f}  z={mm[2]:+8.2f}",
                 f"rot [deg]: yaw={yaw:+7.2f}  pitch={pitch:+7.2f}  roll={roll:+7.2f}",
                 f"quat (x,y,z,w): [{p[3]:+.4f}, {p[4]:+.4f}, {p[5]:+.4f}, {p[6]:+.4f}]")
        if self.target is not None:
            force, torque = self.get_coupling_load()
            lines += (f"molla: forza {force * 1000:6.1f} mN  coppia {torque * 1000:5.2f} mN*m",)
        return lines

    def reset_probe(self):
        if self.target is None:
            self._write_pose(self.start_pose)
            return
        # sonda dinamica: si riposizionano sonda e bersaglio e si azzera la velocita'
        for dofs in (self.dofs, self.target):
            for data in (dofs.position, dofs.free_position):
                with data.writeableArray() as pos:
                    pos[0] = self.start_pose
        for data in (self.dofs.velocity, self.dofs.free_velocity):
            with data.writeableArray() as vel:
                vel[0] = np.zeros(6)

    def _write_target(self, pose):
        for data in (self.target.position, self.target.free_position):
            with data.writeableArray() as pos:
                pos[0] = pose

    def _clamp_target(self):
        """Tiene il bersaglio entro max_offset / max_angle dalla sonda: cosi' la molla (e quindi
        la forza sulle pareti) non puo' superare rigidezza * scostamento massimo."""
        p, t = self.get_tip_pose(), self.get_target_pose()
        offset = t[0:3] - p[0:3]
        dist = np.linalg.norm(offset)
        if dist > self.coupling["max_offset"]:
            t[0:3] = p[0:3] + offset * (self.coupling["max_offset"] / dist)

        rot_p = R.from_quat(p[3:7])
        rel = (R.from_quat(t[3:7]) * rot_p.inv()).as_rotvec()
        angle = np.linalg.norm(rel)
        if angle > self.coupling["max_angle"]:
            t[3:7] = (R.from_rotvec(rel * (self.coupling["max_angle"] / angle)) * rot_p).as_quat()
        self._write_target(t)

    def _write_pose(self, pose):
        # si aggiorna anche free_position: FreeMotionAnimationLoop calcola i contatti sulle
        # posizioni "libere", altrimenti la sonda verrebbe vista un passo in ritardo
        with self.dofs.position.writeableArray() as pos:
            pos[0] = pose
        with self.dofs.free_position.writeableArray() as free:
            free[0] = pose
        # stessa cosa per le sfere di collisione (equivale a quanto fa la RigidMapping)
        if self.col_dofs is not None:
            spheres = pose[0:3] + R.from_quat(pose[3:7]).apply(self.col_local)
            with self.col_dofs.position.writeableArray() as pos:
                pos[:] = spheres
            with self.col_dofs.free_position.writeableArray() as free:
                free[:] = spheres

    # --- Tastiera -----------------------------------------------------------------
    def _poll_keyboard(self):
        """Aggiorna held_keys leggendo i tasti premuti in questo momento (solo con runSofa in primo piano)."""
        if self.keyboard is None:
            return
        focused = self.keyboard.has_focus()
        pressed = {k for k in self.KEYMAP if focused and self.keyboard.is_down(k)}
        for key in sorted(pressed - self.held_keys):
            print(f"[sonda] tasto {key} premuto")
        self.held_keys = pressed

        x_down = focused and self.keyboard.is_down("X")
        if x_down and not self.x_was_down:   # solo sul fronte di pressione, non a ogni passo
            self.reset_requested = True
            print("[sonda] reset alla posizione iniziale")
        self.x_was_down = x_down

    # --- Eventi SOFA ------------------------------------------------------------
    def onAnimateBeginEvent(self, event):
        self._poll_keyboard()
        if self.reset_requested:
            self.reset_probe()
            self.reset_requested = False
        command = self.command + sum((np.array(self.KEYMAP[k], dtype=float) for k in self.held_keys),
                                     np.zeros(4))
        self.command[:] = 0.0
        if np.any(command):
            advance, yaw, pitch, lift = np.clip(command, -1.0, 1.0)
            self._move(advance * self.trans_step,
                       yaw * self.rot_step,
                       pitch * self.rot_step,
                       lift * self.trans_step)
        if self.target is not None:
            self._clamp_target()

    def onAnimateEndEvent(self, event):
        self.step_count += 1
        lines = None
        if self.labels:
            lines = self.pose_text()
            for label, text in zip(self.labels, lines):
                label.label.value = text
        if self.print_every and self.step_count % self.print_every == 0:
            lines = lines or self.pose_text()
            t = self.getContext().getTime()
            print(f"[sonda] passo {self.step_count}  t={t:.2f}s  " + "  |  ".join(lines))

    def _move(self, advance, yaw, pitch, lift):
        p = self.get_target_pose()
        rot = R.from_quat(p[3:7])  # SOFA e scipy usano entrambi l'ordine (x, y, z, w)

        # yaw attorno all'asse verticale globale
        rot = R.from_rotvec(self.up * yaw) * rot

        # pitch attorno all'asse laterale della sonda
        fwd = rot.apply([1.0, 0.0, 0.0])
        side = np.cross(fwd, self.up)
        n = np.linalg.norm(side)
        if n > 1e-6:
            rot = R.from_rotvec(side / n * pitch) * rot

        fwd = rot.apply([1.0, 0.0, 0.0])
        p[0:3] += fwd * advance + self.up * lift
        p[3:7] = rot.as_quat()
        if self.target is not None:
            self._write_target(p)
        else:
            self._write_pose(p)
