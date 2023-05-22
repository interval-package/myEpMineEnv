from typing import Any, Dict, Optional, Type, Union
import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import torch as th
from envs.SingleAgent.mine_toy import EpMineEnv
from envs.SingleAgent.TransEpMineEnv import TransEpMineEnv

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    env_id = "EpMineEnv-v0"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    # env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv)
    # env = gym.make("EpMineEnv-v0")
    env = TransEpMineEnv(port=3000, file_name="envs/SingleAgent/MineField_Windows-0505-random/drl.exe")
    model = PPO("CnnPolicy", env, verbose=1)
    # model = PPO.load("./model.zip", env, verbose=1)
    model.learn(total_timesteps=1e6)

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

    model.save("./model.zip")
    
