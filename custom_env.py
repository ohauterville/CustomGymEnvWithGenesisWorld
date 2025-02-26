import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

from genesis_world import GenesisWorldEnv

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
        self.sim = GenesisWorldEnv(render_mode=render_mode)

        # Init action space
        self.action_space = spaces.Box(
            low=self.sim.action_space_limits[0].cpu().numpy(),
            high=self.sim.action_space_limits[1].cpu().numpy(),
        )

        # Init obs space
        self.observation_space = spaces.Box(
            low=np.concatenate(
                [
                    self.sim.action_space_limits[0].cpu().numpy(),
                    -2 * np.ones(len(self.sim.target_pos)),
                ]
            ),
            high=np.concatenate(
                [
                    self.sim.action_space_limits[1].cpu().numpy(),
                    2 * np.ones(len(self.sim.target_pos)),
                ]
            ),
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return self.sim.reset(seed=seed)

    def step(self, action):
        return self.sim.step(action)

    def render(self):
        # no need since genesis take care of it
        pass


if __name__ == "__main__":
    env_name = "CustomEnv-v0"

    # print("Checking custom env...")
    # check_env(env.unwrapped)
    # print("Checking complete.\n")
