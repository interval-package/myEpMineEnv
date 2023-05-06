from envs.SingleAgent import *
import time
import cv2 as cv

def main():
    env = EpMineEnv(port=3000, max_episode_steps=2000, file_name="envs\SingleAgent\MineField_x86-0417", obs_shape=(640, 320))
    # result = env.step()  # if not, the env won't update
    obs = env.reset()
    done = False
    step = 0
    while not done:
        print(time.time())
        action = env.action_space.sample()
        # action = [0.0, 5.0, 0.0]
        # action = [hori, vert, spin, arm_ang, catching]
        obs, reward, done, info = env.step(action)
        position = info["robot_position"]
        # print(reward)
        # print(np.array(obs["image"]).shape)
        cv.imwrite("data/images/1/{}-({}, {}).png".format(step, position[0], position[2]), obs)
        print('----------------------------------------')
        step += 1


if __name__ == '__main__':
    main()