"""
Addestramento DQN del broncoscopio (MazeEnv di env.py), in PyTorch.

DQN richiede azioni discrete: il wrapper DiscreteActions trasforma le 4 azioni continue
dell'ambiente in azioni discrete, le stesse dei tasti in runSofa: W/S (inserisci/ritira),
A/D (ruota), R/F (leva su/giu') e, solo con CABLE_WAYS = 4, Q/E (leva sinistra/destra).
Il reward include la penalita' per il contatto con le pareti (CONTACT_PENALTY in env.py).

USO
  python training_loop.py         -> addestra e salva i pesi della rete in dqn_maze<passo>.pt

  Osservazione: OBS_MODE in env.py. "state" usa un MLP sui 12 valori; "camera" usa una rete
  convoluzionale sulle immagini dell'endoscopio (CameraQNetwork).
"""

import random
from collections import deque
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import scene_maze
from env import MazeEnv

# =============================================================================
# Iperparametri
# =============================================================================
TOTAL_STEPS = 6_000          # passi di simulazione totali
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

# scala le osservazioni "state" (18 valori, vedi env.py) a ordini di grandezza ~1
OBS_SCALE = np.array([100, 100, 100,          # posizione della punta [m] -> [cm]
                      1, 1, 1, 1,             # quaternione
                      10, 1, 1, 1,            # inserimento [m] -> [dm], rotazione [rad], leve
                      1, 30,                  # spinta [N] (max ~2), coppia [N*m] (max ~0.03)
                      1, 1,                   # forza di contatto punta / shaft [N]
                      100, 100, 100],         # punta -> goal [m] -> [cm]
                     dtype=np.float32)
# modo "camera" (env.OBS_MODE): scala di [spinta, coppia, leva su/giu', leva sx/dx];
# le immagini si dividono per 255 nella rete
VECTOR_SCALE = np.array([1, 30, 1, 1], dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE: ", device)

# =============================================================================
# Wrapper
# =============================================================================
class DiscreteActions(gym.ActionWrapper):
    """Azione intera -> comando continuo [avanzamento, rotazione, leva su/giu', leva sx/dx]."""

    ALL_COMMANDS = np.array([
        [+1, 0, 0, 0], [-1, 0, 0, 0],   # W / S: inserisci / ritira
        [0, +1, 0, 0], [0, -1, 0, 0],   # A / D: ruota lo strumento
        [0, 0, +1, 0], [0, 0, -1, 0],   # R / F: punta su / giu'
        [0, 0, 0, +1], [0, 0, 0, -1],   # Q / E: punta a sinistra / destra (solo 4 cavi)
    ], dtype=np.float32)
    # con 2 cavi la seconda leva non esiste: le ultime due azioni non farebbero niente
    COMMANDS = ALL_COMMANDS if scene_maze.CABLE_WAYS == 4 else ALL_COMMANDS[:6]

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(self.COMMANDS))

    def action(self, action):
        """Indice dell'azione -> comando continuo per MazeEnv.step()."""
        return self.COMMANDS[action]


def scale_observation(obs):
    """Porta l'osservazione a valori ~1 (usata anche da view_agent.py).
    Modo "camera" (dizionario): si scala solo il vettore spinta/coppia/leve; le immagini restano uint8
    (4 volte meno memoria nel replay buffer) e si normalizzano dentro la rete."""
    if isinstance(obs, dict):
        return {"image": obs["image"], "vector": obs["vector"] * VECTOR_SCALE}
    return obs * OBS_SCALE


class ScaleObservation(gym.ObservationWrapper):
    def observation(self, obs):
        return scale_observation(obs)


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


class CameraQNetwork(nn.Module):
    """Rete Q per il modo "camera": immagini dell'endoscopio + [spinta, coppia, leve].

    - encoder convoluzionale (lo stesso di DQN per i giochi Atari): 3 strati che riducono
      l'immagine 64x64 a 64 mappe 4x4, cioe' 1024 "caratteristiche" visive;
    - le caratteristiche si concatenano con il vettore (spinta, coppia, leve) e passano in un MLP
      che da' un valore Q per ognuna delle azioni discrete."""

    def __init__(self, image_shape, n_vector, n_actions, hidden=256):
        super().__init__()
        channels, height, width = image_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4), nn.ReLU(),   # 64x64 -> 15x15
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),         # 15x15 -> 6x6
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),         # 6x6   -> 4x4
            nn.Flatten(),
        )
        with torch.no_grad():   # dimensione dell'uscita dell'encoder (dipende dalla risoluzione)
            n_features = self.encoder(torch.zeros(1, channels, height, width)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(n_features + n_vector, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        image = obs["image"].float() / 255.0   # uint8 [0, 255] -> float [0, 1]
        return self.head(torch.cat([self.encoder(image), obs["vector"]], dim=1))


def make_q_network(observation_space, n_actions):
    """Rete adatta all'osservazione: MLP per "state", CNN + MLP per "camera"."""
    if isinstance(observation_space, gym.spaces.Dict):
        return CameraQNetwork(observation_space["image"].shape, observation_space["vector"].shape[0],
                              n_actions)
    return QNetwork(observation_space.shape[0], n_actions)


def to_tensor(observations):
    """Lista di osservazioni (array o dizionari) -> tensore (o dizionario di tensori) sul device.
    Le immagini restano uint8 fino alla rete; tutto il resto diventa float32."""
    if isinstance(observations[0], dict):
        return {key: torch.as_tensor(np.stack([o[key] for o in observations]), device=device,
                                     dtype=torch.uint8 if key == "image" else torch.float32)
                for key in observations[0]}
    return torch.as_tensor(np.array(observations), dtype=torch.float32, device=device)


def is_finite(obs):
    values = obs["vector"] if isinstance(obs, dict) else obs
    return bool(np.all(np.isfinite(values)))


class ReplayBuffer:
    """Memoria delle transizioni (obs, azione, ricompensa, obs successiva, fine).
    In modo "camera" ogni osservazione contiene le immagini: con 64x64, 1 canale e FRAME_STACK = 2
    sono 8 kB, quindi ~0.8 GB per 50000 transizioni (obs + obs successiva). Riduci BUFFER_SIZE se
    la RAM non basta."""

    def __init__(self, size):
        self.data = deque(maxlen=size)

    def add(self, obs, action, reward, next_obs, done):
        self.data.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        obs, action, reward, next_obs, done = zip(*random.sample(self.data, batch_size))
        to = lambda x, dtype: torch.as_tensor(np.array(x), dtype=dtype, device=device)
        return (to_tensor(obs), to(action, torch.int64), to(reward, torch.float32),
                to_tensor(next_obs), to(done, torch.float32))

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

    # il tipo di osservazione ("state" o "camera") si sceglie con OBS_MODE in env.py
    env = ScaleObservation(DiscreteActions(MazeEnv(max_steps=MAX_EPISODE_STEPS)))
    n_actions = env.action_space.n

    q_net = make_q_network(env.observation_space, n_actions).to(device)
    target_net = make_q_network(env.observation_space, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(BUFFER_SIZE)
    print(f"dispositivo: {device}")

    obs, info = env.reset(seed=SEED)
    episode, episode_return, episode_len, episode_penalty = 0, 0.0, 0, 0.0

    for step in range(TOTAL_STEPS):
        # --- scelta dell'azione (epsilon-greedy) ---
        if step < LEARNING_STARTS or random.random() < epsilon(step):
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q = q_net(to_tensor([obs]))   # batch di una sola osservazione
            action = int(q.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        episode_len += 1
        episode_penalty += info.get("contact_penalty", 0.0)

        # una simulazione divergente non va nel buffer: si ricomincia l'episodio
        if is_finite(next_obs):
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
                  f"ritorno {episode_return:8.2f}  y finale {info['tip_pose'][1] * 1000:+7.2f} mm  "
                  f"penalita' contatto {episode_penalty:6.2f}"
                  f"{'  (forza eccessiva)' if info.get('unsafe_contact') else ''}  "
                  f"eps {epsilon(step):.2f}")
            obs, info = env.reset()
            episode_return, episode_len, episode_penalty = 0.0, 0, 0.0

        if step % 5000 == 0:
            torch.save(q_net.state_dict(), SAVE_PATH + str(step) + ".pt")
            print(f"pesi salvati in {SAVE_PATH + str(step) + ".pt"}")
    env.close()


if __name__ == "__main__":
    start_time = time.time()
    train()
    print(f"tempo totale: {time.time() - start_time}")
