import gym
from gym.spaces import Box, Discrete
from tianshou.env import DummyVectorEnv
from .utility import CompUtility
import numpy as np

class AIGCEnv(gym.Env):

    def __init__(self):

        self._flag = 0
        # Define observation space based on the shape of the state
        self._observation_space = Box(shape=self.state.shape, low=0, high=1)
        # Define action space - discrete space with 3 possible actions
        self._action_space = Discrete(2*5)
        # self._action_space = Box(low=-1, high=1, shape=(1,))

        self._num_steps = 0
        self._terminated = False
        self._laststate = None
        self.last_expert_action = None
        # Define the number of steps per episode
        self._steps_per_episode = 1

    @property
    def observation_space(self):
        # Return the observation space
        return self._observation_space

    @property
    def action_space(self):
        # Return the action space
        return self._action_space


    @property
    def state(self):
        # Provide the current state to the agent
        # rng = np.random.default_rng(seed=0)
        # states1 = rng.uniform(1, 2, 5)
        # states2 = rng.uniform(0, 1, 5)

        states1 = np.random.uniform(1, 2, 5)
        states2 = np.random.uniform(0, 1, 5)

        reward_in = []
        reward_in.append(0)
        states = np.concatenate([states1, states2, reward_in])

        self.channel_gains = np.concatenate([states1, states2])
        self._laststate = states
        return states


    def step(self, action):
        # Check if episode has ended
        assert not self._terminated, "One episodic has terminated"
        # Calculate reward based on last state and action taken
        reward, expert_action, sub_expert_action, real_action, data_rate = CompUtility(self.channel_gains, action)

        self._laststate[-1] = reward
        self._laststate[0:-1] = self.channel_gains * real_action
        # self._laststate[0:-1] = self.channel_gains * real_action
        self._num_steps += 1
        # Check if episode should end based on number of steps taken
        if self._num_steps >= self._steps_per_episode:
            self._terminated = True
        # Information about number of steps taken
        info = {'num_steps': self._num_steps, 'expert_action': expert_action, 'sub_expert_action': sub_expert_action, 'data_rate': data_rate}
        return self._laststate, reward, self._terminated, info

    def reset(self):
        # Reset the environment to its initial state
        self._num_steps = 0
        self._terminated = False
        state = self.state
        return state, {'num_steps': self._num_steps}

    def seed(self, seed=None):
        # Set seed for random number generation
        np.random.seed(seed)


def make_aigc_env(training_num=0, test_num=0):
    """Wrapper function for AIGC env.
    :return: a tuple of (single env, training envs, test envs).
    """
    env = AIGCEnv()
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        # Create multiple instances of the environment for training
        train_envs = DummyVectorEnv(
            [lambda: AIGCEnv() for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        # Create multiple instances of the environment for testing
        test_envs = DummyVectorEnv(
            [lambda: AIGCEnv() for _ in range(test_num)])
        test_envs.seed(0)
    return env, train_envs, test_envs

# import gym
# import numpy as np
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.monitor import Monitor

# def make_aigc_env_sb3(n_envs=1):
#     """Wrapper function for SB3-compatible AIGC environment."""
#     def _make_env():
#         env = AIGCEnv()
#         return Monitor(env)  # Để SB3 có thể log reward

#     return DummyVecEnv([_make_env for _ in range(n_envs)])
