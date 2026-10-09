"""
Mostra in runSofa la sonda guidata dalla rete DQN addestrata con training_loop.py.

USO
  runSofa -l SofaPython3 -a view_agent.py

  A ogni passo la rete legge l'osservazione (la stessa di MazeEnv) e sceglie una delle 8 azioni
  discrete. Restano attivi i tasti della scena: X riporta la sonda alla posizione iniziale.
"""

import os

import numpy as np
import torch
import Sofa.Core

import scene_maze
from env import GOAL, observation
from training_loop import OBS_SCALE, DiscreteActions, QNetwork

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dqn_maze495000.pt")
GOAL_RADIUS = scene_maze.PROBE_RADIUS   # [m] raggio della sfera che mostra il goal


class AgentController(Sofa.Core.Controller):
    def __init__(self, probe_ctrl, q_net, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.probe_ctrl = probe_ctrl
        self.q_net = q_net

    def onAnimateEndEvent(self, event):
        # come in MazeEnv: osservazione dopo il passo -> azione applicata nel passo successivo
        obs = observation(self.probe_ctrl, GOAL) * OBS_SCALE
        with torch.no_grad():
            action = int(self.q_net(torch.as_tensor(obs).unsqueeze(0)).argmax(dim=1).item())
        self.probe_ctrl.set_command(*DiscreteActions.COMMANDS[action])


def createScene(root):
    scene_maze.createScene(root, with_visual=True)
    probe_ctrl = root.getObject("ProbeController")

    q_net = QNetwork(len(OBS_SCALE), len(DiscreteActions.COMMANDS))
    q_net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    q_net.eval()

    root.addObject(AgentController(probe_ctrl, q_net, name="AgentController"))

    # goal: piccola sfera rossa, solo grafica (nessuna collisione con la sonda)
    root.addObject("RequiredPlugin", name="goalPlugins", pluginName=["Sofa.Component.IO.Mesh"])
    goal = root.addChild("Goal")
    goal.addObject("MeshOBJLoader", name="loader", filename="mesh/sphere.obj")   # sfera di raggio 1
    goal.addObject("OglModel", src="@loader", color=[1.0, 0.0, 0.0, 1.0],
                   scale3d=[GOAL_RADIUS] * 3, translation=list(GOAL))
    return root
