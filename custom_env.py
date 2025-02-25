import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

from robot_world import RobotWorld

import os
import numpy as np

# Register the env in Gym
register(
    id="CustomEnv-v0",
    entry_point="custom_env:CustomEnv",  # module_name:class_name
)


# Msut inherit from gym.Env
# https://gymnasium.faram.org/api/env/
class CustomEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Init
        self.robot_world = RobotWorld(render_mode=render_mode)

        # Init action space
        self.action_space = spaces.Box(
            low=self.robot_world.action_space_limits[0].cpu().numpy(),
            high=self.robot_world.action_space_limits[1].cpu().numpy(),
        )

        # Init obs space
        self.observation_space = spaces.Box(
            low=np.concatenate(
                [
                    self.robot_world.action_space_limits[0].cpu().numpy(),
                    -2*np.ones(len(self.robot_world.target_pos)),
                ]
            ),
            high=np.concatenate(
                [
                    self.robot_world.action_space_limits[1].cpu().numpy(),
                    2*np.ones(len(self.robot_world.target_pos)),
                ]
            ),
        )
        print()
        print(self.observation_space)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return self.robot_world.reset(seed=seed)

    def step(self, action):
        return self.robot_world.step(action)

    def render(self):
        # no need since genesis take care of it
        pass


if __name__ == "__main__":
    env_name = "CustomEnv-v0"

    # print("Checking custom env...")
    # check_env(env.unwrapped)
    # print("Checking complete.\n")
