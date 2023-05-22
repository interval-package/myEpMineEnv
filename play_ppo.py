import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from envs.SingleAgent.mine_toy import EpMineEnv
from envs.SingleAgent.TransEpMineEnv import TransEpMineEnv, blur_img
from server.numpy_tcp import depth_img

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

def save_video(img_array, name):
    shape = (img_array[0].shape[1], img_array[0].shape[0])
    out = cv2.VideoWriter(f'{name}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, shape)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    pass

if __name__ == "__main__":
    env_id = "EpMineEnv-v0"
    num_cpu = 1  # Number of processes to use
    # Create the vectorized environment
    # env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv)

    # env = TransEpMineEnv(port=3000, file_name="envs/SingleAgent/MineField_Windows-0505-random/drl.exe")
    env = EpMineEnv(port=3000, file_name="envs/SingleAgent/MineField_Windows-0505-random/drl.exe")
    # env = gym.make("EpMineEnv-v0")
    # model = PPO("CnnPolicy", env, verbose=1)
    model = PPO.load("./model.zip", verbose=1)
    # model.learn(total_timesteps=1e6)

    obs = env.reset()
    arr = []
    for _ in range(300):
        action, _states = model.predict(obs)
        # action = env.action_space.sample()
        obs, rewards, dones, info = env.step(action)
        blur = blur_img(obs).astype('uint8')
        true_depth = depth_img(obs).astype('uint8')
        fake_depth = depth_img(blur).astype('uint8')
        disp = cv2.hconcat([obs, true_depth, blur, fake_depth])
        arr.append(disp)
        cv2.imshow("disp", disp)
        # cv2.imwrite("cur.png", obs)
        print(f"step {_}, action {action}")
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    save_video(arr, "compare")