import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os

class AntTrapEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        local_path = os.path.dirname(__file__)
        xml_file = local_path + "/mujoco_assets/ant_trap.xml"
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
















if __name__ == "__main__":
    import time

    gym.envs.register(
        id='AntTrap-v0',
        max_episode_steps=1000,
        entry_point='environments.continuous_environments.ant_trap:AntTrapEnv',
    )

    env = gym.make('AntTrap-v0')
    print(env.action_space, env.observation_space, flush=True)

    def play_episode(env, rendering=False):
        t0 = time.time()
        timesteps = 0
        obs = env.reset()
        if rendering:
            env.render()
            # time.sleep(50)
        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            if rendering:
                env.render()
            total_reward += reward
            timesteps += 1
        tf = time.time()
        print("episode reward: {}, time in s : {}, num timesteps: {}".format(total_reward, tf - t0, timesteps), flush=True)

    num_episodes = 100
    for _ in range(num_episodes):
        play_episode(env, rendering=True)