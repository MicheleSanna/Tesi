"""
Labirinto deformabile + sonda rigida controllabile da tastiera (SOFA v26.06 + SofaPython3).

USO
  runSofa -l SofaPython3 -a scene_maze.py
                                  -> interfaccia grafica (-a avvia subito la simulazione). Clicca sulla
                                     finestra di runSofa e tieni premuto un tasto (senza Ctrl; la sonda
                                     si muove finche' lo tieni):
                                     W/S  avanti/indietro lungo l'asse della sonda
                                     A/D  ruota a sinistra/destra (yaw)
                                     R/F  punta su/giu (pitch)
                                     Q/E  trasla su/giu
                                     X    riporta la sonda alla posizione iniziale
  python scene_maze.py            -> modalita' headless: misura quanti passi/secondo fa la simulazione

  La posa della punta (posizione, yaw/pitch/roll, quaternione) viene stampata nel terminale ogni
  PRINT_EVERY passi e, in runSofa, mostrata a schermo in alto a sinistra.

NOTE
  - PROBE_MODE = "force" (default): la sonda e' un corpo rigido con massa, mosso da una molla che la
    collega a un BERSAGLIO invisibile comandato da tastiera/RL. Se le pareti del canale la bloccano,
    il bersaglio va avanti ma la sonda resta indietro: la forza che esercita e' quella della molla,
    limitata perche' il bersaglio non puo' allontanarsi piu' di COUPLING_MAX_OFFSET / COUPLING_MAX_ANGLE.
  - PROBE_MODE = "kinematic": la sonda segue esattamente i comandi e spinge le pareti con forza
    illimitata (le puo' attraversare o deformare a piacere).
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
    root = Sofa.Core.Node("root")
    createScene(root, with_visual=False)
    Sofa.Simulation.init(root)
    ctrl = root.getObject("ProbeController")

    t0 = time.perf_counter()
    for _ in range(n_steps):
        ctrl.set_command(advance=1.0)   # la sonda avanza dritta
        Sofa.Simulation.animate(root, root.dt.value)
    elapsed = time.perf_counter() - t0

    print(f"\n{n_steps} passi in {elapsed:.2f} s -> {n_steps / elapsed:.1f} passi/s")
    print(f"posa finale della punta: {ctrl.get_tip_pose()}")


if __name__ == "__main__":
    main()