"""
Misura dei contatti tra il broncoscopio e le pareti delle vie aeree, e penalita' per il reward RL.

DA DOVE VENGONO LE FORZE
  Con FreeMotionAnimationLoop i contatti non sono molle: sono vincoli risolti dal solutore dei
  vincoli (BlockGaussSeidelConstraintSolver). Per ogni contatto il solutore crea tre righe
  (normale + due direzioni di attrito) e ne calcola i moltiplicatori di Lagrange lambda.
  Con computeConstraintForces=True il solutore espone il vettore lambda (constraintForces), e ogni
  MechanicalObject coinvolto espone la matrice J dei vincoli (dato "constraint", una matrice
  sparsa: riga = indice del vincolo, colonne = 3 componenti di ogni punto).
  La forza che i vincoli applicano ai punti e' allora J^T * lambda / dt: lambda e' un impulso
  (forza * passo temporale), per questo si divide per dt.

  Applicata ai centri delle sfere di collisione del broncoscopio si ottiene la forza di contatto
  su ogni sfera. Le sfere toccano solo le pareti (il tubo introduttore non e' un oggetto di
  collisione, vedi bronchoscope.py), quindi tutte le righe di J sono contatti con le pareti. Si
  usa la J dello strumento e non quella dei polmoni perche' funziona anche con i polmoni rigidi
  (LUNG_RIGID), che non sono simulati e non hanno righe di vincolo.

USO
    monitor = ContactMonitor(solver, scope.col_dofs, ...)              # in scene_maze.py
    info = monitor.last            # ContactInfo dell'ultimo passo (aggiornato a fine passo)
    penalty, terms = contact_penalty(info, **pesi)                     # in env.py
"""

from dataclasses import dataclass

import numpy as np
import Sofa
import Sofa.Core


# =============================================================================
# Risultato di una misura
# =============================================================================
@dataclass
class ContactInfo:
    """Contatti in un passo di simulazione. Forze in newton, lunghezze in metri."""

    forces: np.ndarray        # (S, 3) forza delle pareti su ogni sfera (normale + attrito)
    tip_zone: np.ndarray      # (S,) True per le sfere della parte distale (sezione di curvatura)
    spacing: float            # distanza tra le sfere (per convertire il numero di sfere in lunghezza)
    threshold: float = 1e-4   # [N] sotto questa forza una sfera non si considera "in contatto"

    @property
    def magnitudes(self):
        """(S,) modulo della forza su ogni sfera."""
        return np.linalg.norm(self.forces, axis=1)

    @property
    def in_contact(self):
        """(S,) True per le sfere che toccano le pareti."""
        return self.magnitudes > self.threshold

    @property
    def total_force(self):
        """Somma dei moduli delle forze su tutto lo strumento [N]."""
        return float(self.magnitudes.sum())

    @property
    def tip_force(self):
        """Somma dei moduli delle forze sulla parte distale [N]: e' la parte che puo' ferire."""
        return float(self.magnitudes[self.tip_zone].sum())

    @property
    def shaft_force(self):
        """Somma dei moduli delle forze sul resto dello strumento [N]."""
        return float(self.magnitudes[~self.tip_zone].sum())

    @property
    def max_force(self):
        """Forza massima su una singola sfera [N]: indica un appoggio concentrato (pressione alta)."""
        m = self.magnitudes
        return float(m.max()) if len(m) else 0.0

    @property
    def contact_length(self):
        """Lunghezza dello strumento a contatto con le pareti [m] (n. sfere in contatto * passo)."""
        return float(np.count_nonzero(self.in_contact) * self.spacing)

    @classmethod
    def empty(cls, n_spheres, tip_zone, spacing):
        """Nessun contatto."""
        return cls(np.zeros((n_spheres, 3)), tip_zone, spacing)


# =============================================================================
# Controller SOFA che misura i contatti a fine passo
# =============================================================================
class ContactMonitor(Sofa.Core.Controller):
    """A fine di ogni passo calcola le forze di contatto tra le sfere del broncoscopio e le pareti.

    solver        BlockGaussSeidelConstraintSolver con computeConstraintForces=True
    scope_col     MechanicalObject Vec3d dei centri delle sfere di collisione dello strumento
    sphere_abscissa  (S,) ascissa di ogni sfera dalla base [m]
    tip_length    lunghezza della parte distale che conta come "punta" [m]
    label         OglLabel su cui scrivere il riassunto (None = niente testo a schermo)"""

    def __init__(self, solver, scope_col, sphere_abscissa, tip_length,
                 label=None, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.solver = solver
        self.scope_col = scope_col
        abscissa = np.asarray(sphere_abscissa, dtype=float)
        self.n_spheres = len(abscissa)
        self.tip_zone = abscissa >= abscissa[-1] - tip_length
        self.spacing = float(abscissa[1] - abscissa[0]) if len(abscissa) > 1 else 0.0
        self.label = label
        self.last = ContactInfo.empty(self.n_spheres, self.tip_zone, self.spacing)

    def onAnimateEndEvent(self, event):
        """Fine passo: misura i contatti e aggiorna il testo a schermo."""
        self.last = self.measure()
        if self.label is not None:
            info = self.last
            self.label.label.value = (
                f"contatto pareti: totale {info.total_force:6.3f} N  punta {info.tip_force:6.3f} N  "
                f"max {info.max_force:6.3f} N  lunghezza {info.contact_length * 1000:5.1f} mm")

    def reset(self):
        """Dimentica l'ultima misura (da chiamare dopo un reset della simulazione)."""
        self.last = ContactInfo.empty(self.n_spheres, self.tip_zone, self.spacing)

    def measure(self):
        """Calcola il ContactInfo dello stato attuale (dopo la risoluzione dei vincoli)."""
        lam = np.asarray(self.solver.constraintForces.value, dtype=float)
        J = _constraint_matrix(self.scope_col)
        if lam.size == 0 or J is None or J.nnz == 0:
            return ContactInfo.empty(self.n_spheres, self.tip_zone, self.spacing)
        dt = self.getContext().dt.value
        rows = J.shape[0]
        lam = np.pad(lam, (0, max(0, rows - lam.size)))[:rows]

        # forza sui punti = J^T * lambda / dt. La matrice che arriva da SOFA ha solo le colonne
        # fino all'ultimo punto coinvolto in un vincolo: si completa con zeri fino al numero di sfere
        forces = np.zeros((self.n_spheres, 3))
        wall = (J.T @ lam).reshape(-1, 3) / dt
        forces[:len(wall)] = wall[:self.n_spheres]
        return ContactInfo(forces=forces, tip_zone=self.tip_zone, spacing=self.spacing)


def _constraint_matrix(dofs):
    """Matrice J dei vincoli di un MechanicalObject come scipy.sparse (None se vuota o non
    disponibile, ad esempio per un MechanicalObject CUDA)."""
    try:
        J = dofs.constraint.value
    except Exception:
        return None
    if not hasattr(J, "nnz"):
        return None
    return J.tocsr()


# =============================================================================
# Penalita' per il reward
# =============================================================================
def contact_penalty(info: ContactInfo, shaft_weight=0.5, tip_weight=2.0, length_weight=0.0,
                    shaft_tolerance=0.05, tip_tolerance=0.01, exponent=1.0):
    """Penalita' (>= 0, da sottrarre al reward) per il contatto dello strumento con le pareti.

    Tre termini, ognuno disattivabile con peso 0:
      shaft   shaft_weight * max(0, F_shaft - shaft_tolerance) ** exponent
              Contatto del tubo di inserimento. Nelle curve delle vie aeree lo shaft appoggia per
              forza sulle pareti, quindi si tollera una forza totale piccola (shaft_tolerance [N])
              e si penalizza solo quella in piu' (spinte inutili contro le pareti).
      tip     tip_weight * max(0, F_tip - tip_tolerance) ** exponent
              Contatto della parte distale (sezione di curvatura e punta): e' la parte rigida e
              stretta che puo' lesionare la mucosa, quindi pesa di piu' e si tollera meno.
      length  length_weight * (lunghezza a contatto in mm)
              Strisciare lungo le pareti (attrito, irritazione) anche con forze piccole.

    exponent = 1 penalizza in proporzione alla forza; 2 penalizza molto di piu' i picchi.
    Restituisce (penalita' totale, dizionario dei singoli termini per i log)."""
    terms = {
        "shaft": shaft_weight * max(0.0, info.shaft_force - shaft_tolerance) ** exponent,
        "tip": tip_weight * max(0.0, info.tip_force - tip_tolerance) ** exponent,
        "length": length_weight * info.contact_length * 1000.0,
    }
    return float(sum(terms.values())), terms


def unsafe_contact(info: ContactInfo, max_force=None):
    """True se su una sfera agisce piu' di max_force [N] (rischio di lesione/perforazione):
    l'ambiente puo' usarlo per terminare l'episodio. max_force=None -> mai."""
    return max_force is not None and info.max_force > max_force
