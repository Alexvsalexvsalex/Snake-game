import logging
import math
import random
import time

import gym
import torch
from gym import spaces
from tensordict.nn import TensorDictSequential
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import StepCounter, TransformedEnv, GymWrapper
from torchrl.modules import EGreedyModule
from torchrl.objectives import DQNLoss, SoftUpdate

from players.abstract import Player
from players.nn import get_initial_model, wrap_to_policy, ACTION_SPACE, ALL_ACTIONS, state_to_observation, \
    last_model_version, load_last_model, get_path_to_model
from game.collector import CollectorGame
from game.ui import GameUI


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def random_2d_point(seed=None):
    if seed:
        random.seed(seed)
    return [random.uniform(-1, 1), random.uniform(-1, 1)]


class ActionSpace(gym.Space):
    @property
    def is_np_flattenable(self):
        return True

    def __init__(self):
        super().__init__(shape=(2,))

    def sample(self, seed=None):
        return random_2d_point(seed)

    def contains(self, x2):
        return abs(x2[0]) <= 1 and abs(x2[1]) <= 1


class ObservationSpace(gym.Space):
    @property
    def is_np_flattenable(self):
        return True

    def __init__(self):
        super().__init__(shape=(2,))

    def sample(self, seed=None):
        pass

    def contains(self, x):
        pass


class GameEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.ui = GameUI()
            self.game = self.ui.game
        else:
            self.ui = None
            self.game = CollectorGame()
        self.nn_player = Player(self.game.field)
        self.game.add_player(self.nn_player)
        self.action_space = ACTION_SPACE
        self.observation_space = spaces.Box(-15.0, 15.0, shape=(8,))
        self.last_observation = None
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        self.game.reset()
        self.last_observation = state_to_observation(self.nn_player, self.game.state())
        self.render()
        return self.last_observation, {}

    def step(self, action):
        reward = 0

        self.nn_player.pressed_acc = ALL_ACTIONS[action]

        old_score = self.nn_player.score
        old_obs = self.last_observation
        for _ in range(10):
            self.game.step()
        new_obs = state_to_observation(self.nn_player, self.game.state())

        reward -= 0.05
        score_profit = self.nn_player.score - old_score
        if score_profit == 0:
            reward += (math.sqrt(old_obs[0] ** 2 + old_obs[1] ** 2) - math.sqrt(new_obs[0] ** 2 + new_obs[1] ** 2)) / 100
            # reward += self.nn_player.dist_to_wall() / 4000
        else:
            reward += 20 * score_profit

        self.last_observation = new_obs

        self.render()

        terminated = self.game.steps > 300 * (self.nn_player.score + 1)

        if terminated:
            reward = -10

        terminated = terminated or self.game.steps > 100_000

        return new_obs, reward, terminated, False, {}

    def render(self):
        if self.ui is not None:
            self.ui.pre_step_action()
            self.ui.draw()
            # self.ui.time_clock.tick(1)


def main():
    torch.manual_seed(0)

    env = TransformedEnv(GymWrapper(GameEnv()), StepCounter())
    env.set_seed(0)

    last_model_id = last_model_version()

    if last_model_id is None:
        last_model_id = -1
        logger.info("Create new model")
        value_mlp = get_initial_model()
    else:
        logger.info(f"Recover from model {last_model_id}")
        value_mlp = load_last_model()
    policy = wrap_to_policy(value_mlp)
    exploration_module = EGreedyModule(env.action_spec, annealing_num_steps=10_000, eps_end=0.01)
    policy_explore = TensorDictSequential(policy, exploration_module)

    init_rand_steps = 5000
    frames_per_batch = 200
    optim_steps = 10
    collector = SyncDataCollector(
        env,
        policy_explore,
        frames_per_batch=frames_per_batch,
        total_frames=-1,
        init_random_frames=init_rand_steps,
    )
    rb = ReplayBuffer(storage=LazyTensorStorage(20_000))

    loss = DQNLoss(value_network=policy_explore, action_space=env.action_spec, delay_value=True)
    optim = Adam(loss.parameters(), lr=0.01)
    updater = SoftUpdate(loss, eps=0.99)

    total_count = 0
    total_episodes = 0
    t0 = time.time()
    for i, data in enumerate(collector):
        rb.extend(data)
        max_length = rb[:]["next", "step_count"].max()
        if i % 10 == 0:
            logger.info(f"Batch {i}. Max num steps: {max_length}")
            logger.info(f"Model's state: {value_mlp.state_dict()}")
        if len(rb) > init_rand_steps:
            for _ in range(optim_steps):
                sample = rb.sample(1000)
                loss_vals = loss(sample)
                loss_vals["loss"].backward()
                optim.step()
                optim.zero_grad()
                exploration_module.step(data.numel())
                updater.step()
                total_count += data.numel()
                total_episodes += data["next", "done"].sum()
        if i % 500 == 499:
            logger.info(f"Saved new version of model with state: {value_mlp.state_dict()}")
            torch.save(value_mlp.state_dict(), get_path_to_model(last_model_id + 1))
        if max_length > 100_000:
            break

    torch.save(value_mlp.state_dict(), get_path_to_model(last_model_id + 1))

    t1 = time.time()

    logger.info(f"solved after {total_count} steps, {total_episodes} episodes and in {t1 - t0}s.")

    env.close()


if __name__ == '__main__':
    main()
