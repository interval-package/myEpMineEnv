from envs.SingleAgent.mine_toy import *

# import the depthGAN for the 
import sys
sys.path.append("../pytorch-CycleGAN-and-pix2pix")

# import dillusion
from server.numpy_tcp import blur_img

class TransEpMineEnv(EpMineEnv):
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

    decoder_iter = 0
    def decoder_results(self, results):
        org_obs = results[TEAM_NAME].obs
        img = cv.cvtColor(np.array(org_obs[0][AGENT_ID] * 255, dtype=np.uint8), cv.COLOR_RGB2BGR)
        rotation = org_obs[1][AGENT_ID][0:4]
        position = org_obs[1][AGENT_ID][4:7]
        arm_angle = org_obs[1][AGENT_ID][7]
        catching = org_obs[1][AGENT_ID][8]
        is_catched = org_obs[1][AGENT_ID][9]
        mineral_pose = org_obs[1][AGENT_ID][10:13]
        state = org_obs[1][AGENT_ID]

        # enable dillusion
        self.decoder_iter += 1
        if self.decoder_iter == 3:
            img = blur_img(img).astype('uint8')
            self.decoder_iter = 0

        obs = {"image": img, "state": state}
        # print(position, mineral_pose)
        self.catch_state = catching
        if self.only_image:
            return img
        elif self.only_state:
            return np.array(org_obs[1][AGENT_ID][:7])
        return obs

    def get_dense_reward(self, results):
        final_reward = results[TEAM_NAME].reward[AGENT_ID]
        # print(final_reward)
        current_dist = self.get_dist_to_mine(reuslts=results)
        # print(self.last_dist - current_dist)
        delta_r = (self.last_dist - current_dist) 
        final_reward += delta_r
        # if current_dist < 0.5:
        #     final_reward += 1.0
        self.last_dist = current_dist
        return final_reward

    pass