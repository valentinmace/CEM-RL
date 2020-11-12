import numpy as np
import os

from gym import utils
from gym.spaces import Dict, Box
from gym.utils import EzPickle
from gym.envs.mujoco import mujoco_env
from gym.envs.registration import registry, register, make, spec


class AntMaze(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal = np.array([35, -25])
        local_path = os.path.dirname(__file__)
        xml_file = local_path + "/mujoco_assets/ant_maze.xml"
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        reward = - np.sqrt(np.sum(np.square(self.data.qpos[:2] - self.goal)))
        done = False
        ob = self._get_obs()

        return ob, reward, done, dict(bc=self.data.qpos[:2],
                                      x_pos=self.data.qpos[0],
                                      x_position=self.data.qpos[0],
                                      y_position=self.data.qpos[1])

    def _get_obs(self):
        qpos = self.data.qpos.flatten()
        qpos[:2] = (qpos[:2] - 5) / 70
        return np.concatenate([
            qpos,
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.8
        self.viewer.cam.elevation = -45
        self.viewer.cam.lookat[0] = 4.2
        self.viewer.cam.lookat[1] = 0
        self.viewer.opengl_context.set_buffer_size(4024, 4024)