"""
Movimenti predeterminati (cinematici) dei polmoni rigidi: respiro, battito cardiaco, ...

IDEA
  Con LUNG_RIGID = True (scene_maze.py) i polmoni non hanno un materiale: sono una superficie di
  triangoli che collide con il broncoscopio ma non viene spinta da lui. Questo non vuol dire che
  debbano stare fermi: la loro forma puo' cambiare nel tempo secondo una legge stabilita a priori
  (un movimento "prescritto"). Lo strumento ne subisce il movimento attraverso i contatti, il
  tessuto invece non risente dello strumento.

  Un movimento e' una sottoclasse di LungMotion che, dato il tempo, restituisce le posizioni di
  tutti i nodi della superficie. MotionController lo applica a ogni passo di simulazione al
  MechanicalObject della superficie (posizione, velocita' e posizione "libera" usata da
  FreeMotionAnimationLoop).

AGGIUNGERE UN MOVIMENTO
    class MioMovimento(LungMotion):
        def positions(self, t):
            return self.rest + ...          # (N, 3) posizioni al tempo t [m]

  e in scene_maze.py: LUNG_MOTION = lambda rest, info: MioMovimento(rest, ...).
  self.rest sono le posizioni a riposo (quelle della mesh dopo orientamento e posizionamento);
  info (SceneInfo) da' i riferimenti della scena: imboccatura, asse della trachea, verticale.
"""

from dataclasses import dataclass
import math

import numpy as np
import Sofa.Core


@dataclass
class SceneInfo:
    """Riferimenti geometrici della scena utili per costruire un movimento (tutto in metri)."""

    mouth: np.ndarray          # imboccatura delle vie aeree (punta del broncoscopio all'inizio)
    axis: np.ndarray           # direzione di inserimento (asse della trachea, verso la carena)
    up: np.ndarray             # verticale della scena (con i polmoni realistici: il davanti del paziente)


class LungMotion:
    """Movimento prescritto della superficie dei polmoni. Le sottoclassi implementano positions()."""

    def __init__(self, rest):
        self.rest = np.array(rest, dtype=float)    # (N, 3) posizioni a riposo [m]

    def positions(self, t):
        """(N, 3) posizioni dei nodi al tempo t [s]."""
        raise NotImplementedError


class Breathing(LungMotion):
    """Esempio di respiro: un allungamento periodico lungo l'asse della trachea, nullo
    all'imboccatura e crescente con la distanza da essa, come il diaframma che tira verso il
    basso i polmoni (i bronchi lontani si spostano di piu' della trachea).

    amplitude: spostamento [m] dei nodi a distanza 'reference' dall'imboccatura a fine
    inspirazione; period: durata di un atto respiratorio [s] (12-20 atti al minuto a riposo).
    Lo spostamento segue (1 - cos) / 2: parte da zero (fine espirazione) senza scatti."""

    def __init__(self, rest, info: SceneInfo, amplitude=0.01, period=4.0, reference=0.15):
        super().__init__(rest)
        self.axis = np.asarray(info.axis, dtype=float) / np.linalg.norm(info.axis)
        depth = np.clip((self.rest - info.mouth) @ self.axis, 0.0, None)
        self.weight = depth / reference          # 0 all'imboccatura, 1 a distanza 'reference'
        self.amplitude = amplitude
        self.period = period

    def positions(self, t):
        phase = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / self.period))   # 0 -> 1 -> 0
        return self.rest + np.outer(self.weight * self.amplitude * phase, self.axis)


class MotionController(Sofa.Core.Controller):
    """Applica un LungMotion al MechanicalObject della superficie a ogni passo.

    All'inizio del passo scrive le posizioni al tempo t + dt (quelle che la superficie deve avere
    alla fine del passo) e la velocita' corrispondente. La superficie non ha un solutore, quindi
    anche la posizione "libera" va scritta a mano, come per la mano dell'operatore
    (controller.py): e' quella che FreeMotionAnimationLoop usa per calcolare i contatti."""

    def __init__(self, motion: LungMotion, dofs, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.motion = motion
        self.dofs = dofs

    def onAnimateBeginEvent(self, event):
        """Inizio passo: porta la superficie dove deve essere alla fine del passo."""
        dt = self.getContext().dt.value
        t = self.getContext().getTime()
        new = self.motion.positions(t + dt)
        old = np.asarray(self.dofs.position.value, dtype=float)
        self._write(self.dofs.velocity, (new - old) / dt)
        self._write(self.dofs.position, new)
        self._write(self.dofs.free_position, new)

    def reset(self):
        """Dopo Sofa.Simulation.reset(): superficie nella posizione del tempo 0."""
        rest = self.motion.positions(0.0)
        self._write(self.dofs.position, rest)
        self._write(self.dofs.free_position, rest)
        self._write(self.dofs.velocity, np.zeros_like(rest))

    @staticmethod
    def _write(data, values):
        """Scrive un vettore di Vec3 in un Data di SOFA, anche se e' vuoto (free_position e
        velocity di un MechanicalObject senza solutore possono non essere allocati)."""
        if len(data.value) != len(values):
            data.value = values.tolist()
            return
        with data.writeableArray() as array:
            array[:] = values
