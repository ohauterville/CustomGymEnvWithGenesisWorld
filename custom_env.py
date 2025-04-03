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
        self.dofs_velocity_limit = 1.0  # arbitrary
        self.action_space = spaces.Box(
            low=-self.dofs_velocity_limit * np.ones(9),
            high=self.dofs_velocity_limit * np.ones(9),
        )

        print("Action space boundaries:")
        print(self.action_space.low)
        print(self.action_space.high)

        # --- Define Observation Space ---
        # Get accurate joint position limits
        joint_pos_low = self.sim.action_space_limits[0].cpu().numpy()
        joint_pos_high = self.sim.action_space_limits[1].cpu().numpy()

        # Define reasonable velocity limits (adjust if needed)
        # Example: Allow velocity up to 2x the action limit
        joint_vel_limit = self.dofs_velocity_limit * 2.0
        joint_vel_low = -joint_vel_limit * np.ones(9, dtype=np.float32)
        joint_vel_high = joint_vel_limit * np.ones(9, dtype=np.float32)

        # Define EE position bounds (relative to env_size, adjust if needed)
        ee_pos_bound = self.sim.env_size * 1.5  # A bit larger than env_size
        ee_pos_low = -ee_pos_bound * np.ones(3, dtype=np.float32)
        ee_pos_high = ee_pos_bound * np.ones(3, dtype=np.float32)
        # More precise Z bound if needed (e.g., slightly below 0 to slightly above max reach)
        # ee_pos_low[2] = -0.05
        # ee_pos_high[2] = 1.5 # Estimate max reach height

        # Define target position bounds (based on generation logic)
        target_pos_bound = self.sim.env_size
        target_pos_low = -target_pos_bound * np.ones(3, dtype=np.float32)
        target_pos_high = target_pos_bound * np.ones(3, dtype=np.float32)
        # Give Z a tiny range around 0 to satisfy Gymnasium check
        target_pos_low[2] = -1e-6
        target_pos_high[2] = 1e-6

        # Define EE-Target distance bounds
        dist_low = np.array([0.0], dtype=np.float32)
        # Max possible distance is roughly diagonal across a 2*env_size square
        max_dist = np.sqrt(2 * (2 * self.sim.env_size) ** 2) + 1.0  # Add some buffer
        dist_high = np.array([max_dist], dtype=np.float32)

        # Concatenate all bounds with consistent dtype
        obs_low = np.concatenate(
            [joint_pos_low, joint_vel_low, ee_pos_low, target_pos_low, dist_low]
        ).astype(np.float32)
        obs_high = np.concatenate(
            [joint_pos_high, joint_vel_high, ee_pos_high, target_pos_high, dist_high]
        ).astype(np.float32)

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        print("Observation space boundaries defined.")
        # print("Low:", self.observation_space.low)
        # print("High:", self.observation_space.high)

    def reset(self, seed=None, options=None):
        # Call the superclass reset for proper seeding handling by Gymnasium
        super().reset(seed=seed)
        # Note: self.sim.reset now returns (obs, info)
        observation, info = self.sim.reset(seed=seed)
        return observation, info

    def step(self, action):
        # self.sim.step already returns the correct (obs, reward, terminated, truncated, info) tuple
        return self.sim.step(action)

    def render(self):
        # no need since genesis take care of it
        pass


if __name__ == "__main__":
    env_name = "CustomEnv-v0"

    # print("Checking custom env...")
    # check_env(env.unwrapped)
    # print("Checking complete.\n")
