import gym


def register_envs():
    print("Envs registration")
    gym.envs.register(
        id='AntTrap-v0',
        max_episode_steps=1000,
        entry_point='environments.ant_trap.ant_trap:AntTrapEnv',
    )

    gym.envs.register(
        id='AntMaze-v0',
        max_episode_steps=3000,
        entry_point='environments.ant_maze.ant_maze:AntMaze',
    )

    gym.envs.registration.register(
        id="PointMazeInertiaMujoco-v0",
        entry_point="environments.point_maze_mujoco.point_maze_inertia_mujoco:PointMazeInertiaMujoco"
    )