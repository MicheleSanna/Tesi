"""
Mostra in runSofa il broncoscopio guidato dalla rete DQN addestrata con training_loop.py.

USO
  runSofa -l SofaPython3 -a view_agent.py

  A ogni passo la rete legge l'osservazione (la stessa di MazeEnv) e sceglie una delle azioni
  discrete (DiscreteActions). Restano attivi i tasti della scena: X riporta lo strumento alla
  posizione iniziale.

  L'osservazione segue OBS_MODE di env.py: i pesi devono essere stati addestrati con lo stesso
  modo ("state" o "camera") e con la stessa scena (CABLE_WAYS decide il numero di azioni).
  In modo "camera" la finestra dell'endoscopio mostra anche cio' che vede la rete.
"""

import os

import torch
import Sofa.Core

import scene_maze
from env import GOAL, ObservationBuilder
from training_loop import DiscreteActions, device, make_q_network, scale_observation, to_tensor

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dqn_maze495000.pt")
GOAL_RADIUS = scene_maze.PROBE_RADIUS   # [m] raggio della sfera che mostra il goal


class AgentController(Sofa.Core.Controller):
    """A fine passo calcola l'osservazione e da' al broncoscopio l'azione scelta dalla rete."""

    def __init__(self, probe_ctrl, camera_ctrl, contacts, obs_builder, q_net, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.probe_ctrl = probe_ctrl        # BronchoscopeController
        self.camera_ctrl = camera_ctrl      # EndoscopeController (None in modo "state")
        self.contacts = contacts            # ContactMonitor
        self.obs_builder = obs_builder
        self.q_net = q_net

    def onAnimateEndEvent(self, event):
        # come in MazeEnv: osservazione dopo il passo -> azione applicata nel passo successivo.
        # NB: questo controller e' aggiunto dopo ContactMonitor, quindi vede gia' i contatti
        # misurati in questo passo
        obs = scale_observation(self.obs_builder(self.probe_ctrl, self.camera_ctrl, self.contacts))
        with torch.no_grad():
            action = int(self.q_net(to_tensor([obs])).argmax(dim=1).item())
        self.probe_ctrl.set_command(*DiscreteActions.COMMANDS[action])


def createScene(root):
    """Scena di scene_maze.py + rete addestrata che comanda lo strumento + sfera del goal."""
    obs_builder = ObservationBuilder()
    camera_mode = obs_builder.mode == "camera"
    # in modo "camera" la telecamera serve sempre; altrimenti la scena segue CAMERA_ENABLED
    scene_maze.createScene(root, with_visual=True, with_camera=True if camera_mode else None)
    probe_ctrl = root.getObject("ScopeController")
    contacts = root.getObject("ContactMonitor")
    camera_ctrl = root.getObject("EndoscopeController") if camera_mode else None

    q_net = make_q_network(obs_builder.space, len(DiscreteActions.COMMANDS))
    # i tensori dell'osservazione vanno sullo stesso device della rete
    q_net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    q_net.to(device).eval()

    root.addObject(AgentController(probe_ctrl, camera_ctrl, contacts, obs_builder, q_net,
                                   name="AgentController"))

    # goal: piccola sfera rossa, solo grafica (nessuna collisione con la sonda)
    root.addObject("RequiredPlugin", name="goalPlugins", pluginName=["Sofa.Component.IO.Mesh"])
    goal = root.addChild("Goal")
    goal.addObject("MeshOBJLoader", name="loader", filename="mesh/sphere.obj")   # sfera di raggio 1
    goal.addObject("OglModel", src="@loader", color=[1.0, 0.0, 0.0, 1.0],
                   scale3d=[GOAL_RADIUS] * 3, translation=list(GOAL))
    return root
