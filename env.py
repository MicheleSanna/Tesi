"""
Ambiente di reinforcement learning per la scena del labirinto (stessa interfaccia di gymnasium:
reset() -> (obs, info), step(action) -> (obs, reward, terminated, truncated, info), close()).

AZIONE (4 valori in [-1, 1], come ProbeController.set_command)
  [avanzamento, yaw, pitch, sollevamento]

OSSERVAZIONE (12 valori)
  [0:7]   posa della punta: x, y, z [m], qx, qy, qz, qw
  [7:9]   forza [N] e coppia [N*m] della molla bersaglio -> sonda (contatto con le pareti)
  [9:12]  vettore punta -> GOAL [m] nel sistema di riferimento della sonda

RICOMPENSA
  spostamento della punta lungo +Y in questo passo, in millimetri

USO
  python env.py                   -> un episodio con azioni casuali
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

import Sofa
import Sofa.Core
import Sofa.Simulation

import scene_maze

# fondo del canale di maze4.1 (il canale e' cieco: l'unica apertura e' l'ingresso su -Y)
GOAL = np.array([0.05, 0.51, 0.0]) * scene_maze.MESH_SCALE   # [m]
MAX_STEPS = 500


class Box:
    """Versione minima di gymnasium.spaces.Box."""

    def __init__(self, low, high, shape):
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)
        self.shape = shape
        self.rng = np.random.default_rng()

    def sample(self):
        return self.rng.uniform(self.low, self.high).astype(np.float32)


class MazeEnv:
    def __init__(self, goal=GOAL, max_steps=MAX_STEPS):
        self.goal = np.asarray(goal, dtype=float)
        self.max_steps = max_steps
        self.action_space = Box(-1.0, 1.0, (4,))
        self.observation_space = Box(-np.inf, np.inf, (12,))

        self.root = Sofa.Core.Node("root")
        scene_maze.createScene(self.root, with_visual=False)
        Sofa.Simulation.init(self.root)
        self.ctrl = self.root.getObject("ProbeController")
        self.ctrl.print_every = 0     # niente stampe a terminale durante il training

        self.steps = 0
        self.prev_y = 0.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.action_space.rng = np.random.default_rng(seed)
        Sofa.Simulation.reset(self.root)   # riporta labirinto, sonda e bersaglio allo stato iniziale
        self.steps = 0
        self.prev_y = self.ctrl.get_tip_pose()[1]
        return self._obs(), {}

    def step(self, action):
        self.ctrl.set_command(*np.clip(np.asarray(action, dtype=float), -1.0, 1.0))
        Sofa.Simulation.animate(self.root, self.root.dt.value)
        self.steps += 1

        obs = self._obs()
        y = obs[1]
        reward = (y - self.prev_y) * 1000.0
        self.prev_y = y

        terminated = not np.all(np.isfinite(obs))   # simulazione divergente
        truncated = self.steps >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def close(self):
        Sofa.Simulation.unload(self.root)

    def _obs(self):
        pose = self.ctrl.get_tip_pose()
        force, torque = self.ctrl.get_coupling_load()
        goal_rel = R.from_quat(pose[3:7]).inv().apply(self.goal - pose[0:3])
        return np.concatenate([pose, [force, torque], goal_rel]).astype(np.float32)


if __name__ == "__main__":
    env = MazeEnv(max_steps=100)
    obs, info = env.reset(seed=0)
    total = 0.0
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        done = terminated or truncated
    print(f"passi: {env.steps}  ricompensa totale: {total:.3f}  osservazione finale: {obs}")
    env.close()
