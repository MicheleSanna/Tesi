"""
Addestramento DQN della sonda nel labirinto (MazeEnv di env.py), in PyTorch.

DQN richiede azioni discrete: il wrapper DiscreteActions trasforma le 4 azioni continue
dell'ambiente in 8 azioni discrete, le stesse dei tasti in runSofa (W/S, A/D, R/F, Q/E).

USO
  python training_loop.py         -> addestra e salva i pesi della rete in dqn_maze.pt
"""

import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import MazeEnv

# =============================================================================
# Iperparametri
# =============================================================================
TOTAL_STEPS = 500_000          # passi di simulazione totali
MAX_EPISODE_STEPS = 300
BUFFER_SIZE = 50_000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-3
LEARNING_STARTS = 1_000       # passi casuali prima di iniziare ad aggiornare la rete
TRAIN_EVERY = 40               # un aggiornamento ogni N passi
TARGET_UPDATE_EVERY = 1_000   # copia la rete nella rete target ogni N passi
EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY_STEPS = 20_000      # epsilon scende linearmente da EPS_START a EPS_END
SEED = 0
SAVE_PATH = "dqn_maze"

# scala le osservazioni a ordini di grandezza ~1: posizioni [m] -> [cm], forza / 100 N, coppia / 1 N*m
OBS_SCALE = np.array([100, 100, 100, 1, 1, 1, 1, 0.01, 1, 100, 100, 100], dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE: ", device)

# =============================================================================
# Wrapper
# =============================================================================
class DiscreteActions(gym.ActionWrapper):
    """Azione intera -> comando continuo [avanzamento, yaw, pitch, sollevamento]."""

    COMMANDS = np.array([
        [+1, 0, 0, 0], [-1, 0, 0, 0],   # W / S: avanti / indietro
        [0, +1, 0, 0], [0, -1, 0, 0],   # A / D: yaw sinistra / destra
        [0, 0, +1, 0], [0, 0, -1, 0],   # R / F: pitch su / giu
        [0, 0, 0, +1], [0, 0, 0, -1],   # Q / E: trasla su / giu
    ], dtype=np.float32)

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(self.COMMANDS))

    def action(self, action):
        return self.COMMANDS[action]


class ScaleObservation(gym.ObservationWrapper):
    def observation(self, obs):
        return obs * OBS_SCALE


# =============================================================================
# Rete Q e replay buffer
# =============================================================================
class QNetwork(nn.Module):
    def __init__(self, n_obs, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, size):
        self.data = deque(maxlen=size)

    def add(self, obs, action, reward, next_obs, done):
        self.data.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        obs, action, reward, next_obs, done = zip(*random.sample(self.data, batch_size))
        to = lambda x, dtype: torch.as_tensor(np.array(x), dtype=dtype, device=device)
        return (to(obs, torch.float32), to(action, torch.int64), to(reward, torch.float32),
                to(next_obs, torch.float32), to(done, torch.float32))

    def __len__(self):
        return len(self.data)


def epsilon(step):
    frac = min(step / EPS_DECAY_STEPS, 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)


# =============================================================================
# Training loop
# =============================================================================
def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = ScaleObservation(DiscreteActions(MazeEnv(max_steps=MAX_EPISODE_STEPS)))
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = QNetwork(n_obs, n_actions).to(device)
    target_net = QNetwork(n_obs, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(BUFFER_SIZE)
    print(f"dispositivo: {device}")

    obs, _ = env.reset(seed=SEED)
    episode, episode_return, episode_len = 0, 0.0, 0

    for step in range(TOTAL_STEPS):
        # --- scelta dell'azione (epsilon-greedy) ---
        if step < LEARNING_STARTS or random.random() < epsilon(step):
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q = q_net(torch.as_tensor(obs, device=device).unsqueeze(0))
            action = int(q.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward
        episode_len += 1

        # una simulazione divergente non va nel buffer: si ricomincia l'episodio
        if np.all(np.isfinite(next_obs)):
            # truncated non e' uno stato terminale: il valore futuro va comunque stimato
            buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs

        # --- aggiornamento della rete Q ---
        if step >= LEARNING_STARTS and step % TRAIN_EVERY == 0:
            b_obs, b_act, b_rew, b_next, b_done = buffer.sample(BATCH_SIZE)
            with torch.no_grad():
                next_q = target_net(b_next).max(dim=1).values
                target = b_rew + GAMMA * (1.0 - b_done) * next_q
            q_pred = q_net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q_pred, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

        if step % TARGET_UPDATE_EVERY == 0:
            target_net.load_state_dict(q_net.state_dict())

        # --- fine episodio ---
        if terminated or truncated:
            episode += 1
            print(f"episodio {episode:4d}  passo {step + 1:6d}  passi {episode_len:3d}  "
                  f"ritorno {episode_return:8.2f}  y finale {obs[1] * 10:+7.2f} mm  "
                  f"eps {epsilon(step):.2f}")
            obs, _ = env.reset()
            episode_return, episode_len = 0.0, 0

        if step % 5000 == 0:
            torch.save(q_net.state_dict(), SAVE_PATH + str(step) + ".pt")
            print(f"pesi salvati in {SAVE_PATH + str(step) + ".pt"}")
    env.close()


if __name__ == "__main__":
    train()
