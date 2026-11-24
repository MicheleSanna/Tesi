"""
Tessuto deformabile (labirinto o polmoni) + broncoscopio flessibile a cavi controllabile da
tastiera (SOFA v26.06 + SofaPython3 + plugin BeamAdapter e SoftRobots).

USO
  runSofa -l SofaPython3 -a scene_maze.py
                                  -> interfaccia grafica (-a avvia subito la simulazione). Clicca sulla
                                     finestra di runSofa e tieni premuto un tasto (senza Ctrl; il comando
                                     agisce finche' lo tieni):
                                     W/S  inserisci / ritira lo strumento
                                     A/D  ruota lo strumento attorno al suo asse
                                     R/F  leva: piega la punta su / giu'
                                     Q/E  seconda leva: punta a sinistra / destra (solo CABLE_WAYS = 4)
                                     X    riporta lo strumento alla posizione iniziale
  python scene_maze.py            -> modalita' headless: misura quanti passi/secondo fa la simulazione

  Lo stato dello strumento (posa della punta, inserimento, rotazione, leve, spinta, tensione dei
  cavi) viene stampato nel terminale ogni PRINT_EVERY passi e, in runSofa, mostrato a schermo in
  alto a sinistra insieme alla forza di contatto con le pareti.

NOTE
  - Il broncoscopio (bronchoscope.py) e' una catena di travi: shaft + sezione di curvatura piegata
    da cavi tendinei. La MANO che lo inserisce e' collegata alla base da una molla: se le pareti lo
    bloccano la mano non puo' allontanarsi piu' di HAND_MAX_PUSH_OFFSET, quindi la spinta resta
    limitata (HAND_STIFFNESS * HAND_MAX_PUSH_OFFSET). Fuori dal paziente lo strumento scorre in un
    tubo introduttore.
  - Unita' SI (metri, kg, Pa). Se il labirinto e' in millimetri, imposta MESH_SCALE = 0.001.
"""

from scene_maze import createScene
import time
import Sofa
import Sofa.Core
import Sofa.Simulation

# =============================================================================
# Modalita' headless: benchmark di velocita'
# =============================================================================
def main(n_steps=300):
    """Inserisce lo strumento dritto per n_steps passi e misura la velocita' della simulazione."""
    root = Sofa.Core.Node("root")
    createScene(root, with_visual=False)
    Sofa.Simulation.init(root)
    ctrl = root.getObject("ScopeController")
    contacts = root.getObject("ContactMonitor")

    t0 = time.perf_counter()
    for _ in range(n_steps):
        ctrl.set_command(advance=1.0)   # lo strumento avanza dritto
        Sofa.Simulation.animate(root, root.dt.value)
    elapsed = time.perf_counter() - t0

    print(f"\n{n_steps} passi in {elapsed:.2f} s -> {n_steps / elapsed:.1f} passi/s")
    print(f"posa finale della punta: {ctrl.get_tip_pose()}")
    print(f"contatto con le pareti: {contacts.last.total_force:.3f} N")


if __name__ == "__main__":
    main()