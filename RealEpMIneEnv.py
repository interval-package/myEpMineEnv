from envs.SingleAgent.mine_toy import *
from server.numpy_tcp import *
import requests, json
import cv2

class RealEpMineEnv(EpMineEnv):

    def __init__(self,
            file_name: str = "envs/SingleAgent/MineField_Linux-0421/drl.x86_64",
            port: Optional[int] = 2000,
            seed: int = 0,
            work_id: int = 0,
            time_scale: float = 20.0,
            max_episode_steps: int = 1000,
            only_image: bool = True,
            only_state: bool = False,
            no_graph: bool = False):
        # init super
        super().__init__(file_name, port, seed, work_id, time_scale, 
                                        max_episode_steps, only_image, only_state, no_graph)

        pass
    pass

class RealEpMineWrapper:
    def __init__(self) -> None:
        pass

    host = "localhost"

    step_url = "http://192.168.1.55:1123/step"



    def restruct_img(self, img):
        height, width, channels = img.shape
        min_len = min(height, width)
        img = img[
            int(height - min_len):int(height), 
            int((width-min_len)/2):int((width+min_len)/2)
            ]
        img = cv2.resize(img, (128,128), cv2.INTER_AREA)
        return img

    def send_action_and_observe(self, action):
        obj = dict()
        obj["action"] = action
        obj = json.dumps(obj)
        r = requests.post(self.step_url, json=obj)
        res = r.text
        img = json2np(res).astype('uint8')
        return self.restruct_img(img)

    step_num = 0
    def step(self, action):
        # action: [vy, vx, vw, arm_ang, catching]
        action = [action[0], action[1], action[2], 10.0, 0]
        action = ActionTuple(np.array([action], dtype=np.float32))
        action_dict = warp_action(action=action)
        toal_reward = 0.0
        for _ in range(1):
            obs, reward, done, info = self._step(action_dict=action_dict)
            toal_reward += reward
        self.step_num += 1
        return obs, toal_reward, done, info
    pass

if __name__ == "__main__":

    pass