import os
import re

from gym import spaces
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
import torch
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform
from torchrl.modules import MLP, QValueModule

from players.abstract import Player


ALL_ACTIONS = {i: [-1 + 2 * (i // 2), -1 + 2 * (i % 2)] for i in range(4)}
ACTION_SPACE = spaces.Discrete(4)
ACTION_SPEC = _gym_to_torchrl_spec_transform(ACTION_SPACE)


def models_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def last_model_version():
    all_models = os.listdir(models_dir())
    if not all_models:
        return None
    all_ids = [int(re.findall(r'\d+', model)[0]) for model in all_models]
    return max(all_ids)


def get_path_to_model(version):
    return os.path.join(models_dir(), f"player_{version}.torch")


def get_initial_model():
    return MLP(in_features=8, out_features=ACTION_SPEC.shape, num_cells=[], bias_last_layer=False)


def wrap_to_policy(model):
    value_net = Mod(model, in_keys=["observation"], out_keys=["action_value"])
    return Seq(value_net, QValueModule(None, spec=ACTION_SPEC))


def load_last_model():
    model = get_initial_model()
    model.load_state_dict(torch.load(get_path_to_model(last_model_version())))
    model.eval()
    return model


def state_to_observation(player: Player, state):
    return [
        (state.ball_x - player.pos()[0]) / 100,
        (state.ball_y - player.pos()[1]) / 100,
        (state.phantom_ball_x - state.ball_x) / 100,
        (state.phantom_ball_y - state.ball_y) / 100,
        player.speed[0],
        player.speed[1],
        player.acc[0],
        player.acc[1],
    ]


class NNPlayer(Player):
    def __init__(self, field, color=None, label=None):
        super().__init__(field, color, label)
        self.model = wrap_to_policy(load_last_model())

    def react_to_state(self, state):
        obs = state_to_observation(self, state)
        source = TensorDict({"observation": torch.FloatTensor([obs])}, batch_size=(1,))
        out = self.model(source)
        action = ACTION_SPEC.to_numpy(out["action"], safe=False)[0]
        self.pressed_acc = ALL_ACTIONS[action]
