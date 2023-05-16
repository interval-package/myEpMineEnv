import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from envs.SingleAgent.mine_toy import EpMineEnv
from envs.SingleAgent.TransEpMineEnv import TransEpMineEnv

import cv2

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
    num_cpu = 1  # Number of processes to use
    # Create the vectorized environment
    # env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv)

    env = TransEpMineEnv(port=3000)
    # env = gym.make("EpMineEnv-v0")
    model = PPO("CnnPolicy", env, verbose=1)
    # model = PPO.load("./models/model_nav", env, verbose=1)
    # model.learn(total_timesteps=1e6)

    obs = env.reset()

    img_array = []
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        img_array.append(obs)
        # cv2.imwrite(f"D:\\coding\\PythonProjects\\data\\img_epMine\\trainning\\pic_{_}_step.png", obs)

    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (128,128))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
