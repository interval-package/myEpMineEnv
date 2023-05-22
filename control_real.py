import gym
import numpy as np

from stable_baselines3 import PPO
from envs.SingleAgent.TransEpMineEnv import TransEpMineEnv, blur_img
from server.numpy_tcp import depth_img, real_img

from RealEpMIneEnv import *

import cv2

def save_video(img_array, name):
    shape = (img_array[0].shape[1], img_array[0].shape[0])
    out = cv2.VideoWriter(f'{name}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, shape)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    pass

if __name__ == "__main__":
    # env = gym.make("EpMineEnv-v0")
    # model = PPO("CnnPolicy", env, verbose=1)
    model = PPO.load("./model.zip", verbose=1)
    # model.learn(total_timesteps=1e6)
    env = RealEpMineWrapper()

    action = [1,0,0,0,0]
    obs = env.send_action_and_observe(action=action)
    arr = []
    for _ in range(300):
        action, _states = model.predict(obs)
        action = action.tolist()
        action = [i*1 for i in action]
        # action = env.action_space.sample()
        obs = env.send_action_and_observe(action)
        obs_real = real_img(obs).astype('uint8')
        obs_depth = depth_img(obs).astype('uint8')
        disp = cv2.hconcat([obs, obs_depth, obs_real])
        cv2.imshow("temp", disp)
        # cv2.imwrite("cur.png", obs)
        print(f"step {_}, action {action}")
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    save_video(arr, "compare")