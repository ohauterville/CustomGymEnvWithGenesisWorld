import gymnasium as gym
from gymnasium.envs.registration import register # Keep for potential future use, though not used now
from stable_baselines3.common.vec_env import VecEnv # Import VecEnv
from genesis_world import GenesisWorldEnv  # Import your environment class
import numpy as np
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Type
import time # Keep time import for potential future use

class CustomEnv(VecEnv): # Inherit from VecEnv
    # Add n_envs parameter to init
    def __init__(self, n_envs=16, render_mode=None):
        # Store render_mode for VecEnv compatibility
        self.render_mode = render_mode
        # Pass n_envs and render_mode to GenesisWorldEnv
        self.sim = GenesisWorldEnv(n_envs=n_envs, render_mode=render_mode)
        # VecEnv requires num_envs, observation_space, action_space at init
        # Important: SB3 VecEnv expects spaces *without* the n_envs dimension
        # Use hardcoded shapes as in the previous working version
        obs_shape = (30,) # Shape for a single environment
        action_shape = (9,) # Shape for a single environment

        single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        single_action_space = gym.spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)

        # Use the passed n_envs for super().__init__
        # Ensure self.sim has n_envs attribute available after init
        actual_n_envs = getattr(self.sim, 'n_envs', n_envs) # Use sim's n_envs if available, else the param
        super().__init__(actual_n_envs, single_observation_space, single_action_space)

        # Store sim attributes after super().__init__
        # Ensure these attributes exist in GenesisWorldEnv
        self.max_steps = getattr(self.sim, 'max_steps', 1000) # Default if not present
        self.min_dist_task_completion = getattr(self.sim, 'min_dist_task_completion', 0.1)
        self.energy_penalty_weight = getattr(self.sim, 'energy_penalty_weight', 0.0)
        self.time_penalty = getattr(self.sim, 'time_penalty', 0.0)
        self.task_completion_bonus = getattr(self.sim, 'task_completion_bonus', 0.0)
        self.end_ep_penalty = getattr(self.sim, 'end_ep_penalty', 0.0)
        self.collision_penalty = getattr(self.sim, 'collision_penalty', 0.0)
        self.max_collisions = getattr(self.sim, 'max_collisions', 5)
        self._actions: np.ndarray # To store actions for step_wait
        # Add trackers for episode stats again
        self.episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_start_times = np.array([time.time()] * self.num_envs, dtype=np.float64)


    # Implement VecEnv abstract methods (Re-adding episode tracking)
    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        # Assuming sim.step returns arrays for rewards, terminated, truncated
        next_obs, rewards, terminated, truncated, info = self.sim.step(self._actions)

        # Update episode trackers
        self.episode_rewards += rewards
        self.episode_lengths += 1

        # Convert info dict if necessary (assuming single dict with arrays)
        infos = [{} for _ in range(self.num_envs)]
        for key, value in info.items():
            if isinstance(value, (list, np.ndarray)) and len(value) == self.num_envs:
                for i in range(self.num_envs):
                    infos[i][key] = value[i]
            else:
                 for i in range(self.num_envs):
                    infos[i][key] = value

        terminated = np.array(terminated, dtype=bool)
        truncated = np.array(truncated, dtype=bool)
        dones = terminated | truncated

        # Add final observation/info for SB3 compatibility when env terminates/truncates
        for i in range(self.num_envs):
             if dones[i]:
                 # Store final observation if not already present in info
                 if "final_observation" not in infos[i]:
                     infos[i]["final_observation"] = next_obs[i].copy()
                 # Store final info if not already present
                 if "final_info" not in infos[i]:
                     original_env_info = {k: v[i] for k, v in info.items() if isinstance(v, (list, np.ndarray)) and len(v) == self.num_envs}
                     infos[i]["final_info"] = original_env_info
                 # Add the episode statistics dictionary for SB3 logging
                 # DEBUG: Print length before recording
                 print(f"DEBUG: Env {i} done. Length recorded: {self.episode_lengths[i]}")
                 episode_info = {
                     "r": self.episode_rewards[i],
                     "l": self.episode_lengths[i],
                     "t": round(time.time() - self.episode_start_times[i], 6),
                 }
                 infos[i]["episode"] = episode_info

                 # Reset trackers for this environment
                 self.episode_rewards[i] = 0
                 self.episode_lengths[i] = 0
                 self.episode_start_times[i] = time.time()

        # Return rewards array (plural) as tracked
        return next_obs, rewards, dones, infos

    def reset(self) -> np.ndarray:
        obs, _info = self.sim.reset()
        # Reset episode trackers
        self.episode_rewards.fill(0.0)
        self.episode_lengths.fill(0)
        self.episode_start_times = np.array([time.time()] * self.num_envs, dtype=np.float64)
        return obs

    def close(self) -> None:
        self.sim.close()

    # --- Other VecEnv methods (simplified implementations) ---
    def get_attr(self, attr_name: str, indices: Optional[Union[int, Sequence[int]]] = None) -> List[Any]:
        # Handle render_mode specifically for VecEnv compatibility
        if attr_name == "render_mode":
            # Return self.render_mode for all requested indices
            if indices is None:
                indices = list(range(self.num_envs))
            elif isinstance(indices, int):
                indices = [indices]
            return [self.render_mode for _ in indices]

        # Otherwise, try to get the attribute from self.sim
        target_obj = self.sim
        if indices is None:
            indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            indices = [indices]
        try:
            attr = getattr(target_obj, attr_name)
            if isinstance(attr, (list, np.ndarray)) and len(attr) == self.num_envs:
                 return [attr[i] for i in indices]
            else:
                return [attr for _ in indices]
        except AttributeError:
            # Fallback to VecEnv default or raise error
            return super().get_attr(attr_name, indices)


    def set_attr(self, attr_name: str, value: Any, indices: Optional[Union[int, Sequence[int]]] = None) -> None:
        target_obj = self.sim
        # Simplified: Set on the main sim object. May need adjustment based on GenesisWorldEnv.
        if hasattr(target_obj, attr_name):
             setattr(target_obj, attr_name, value)
        else:
             # Fallback to VecEnv default or raise error
             super().set_attr(attr_name, value, indices)


    def env_method(self, method_name: str, *method_args, indices: Optional[Union[int, Sequence[int]]] = None, **method_kwargs) -> List[Any]:
        target_obj = self.sim
        if indices is None:
            indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            indices = [indices]
        try:
            method = getattr(target_obj, method_name)
            result = method(*method_args, **method_kwargs) # Assumes method handles vectorization or is shared
            return [result for _ in indices]
        except AttributeError:
            # Fallback to VecEnv default or raise error
            return super().env_method(method_name, *method_args, indices=indices, **method_kwargs)


    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: Optional[Union[int, Sequence[int]]] = None) -> List[bool]:
        if indices is None:
            indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            indices = [indices]
        # Check if self.sim is an instance of the wrapper class
        is_wrapped = isinstance(self.sim, wrapper_class)
        # Check if self is an instance (for wrappers applied to CustomEnv itself)
        self_is_wrapped = isinstance(self, wrapper_class)
        # Combine checks (adjust logic if needed)
        return [(is_wrapped or self_is_wrapped) for _ in indices]


    def get_images(self) -> Sequence[np.ndarray]:
        if hasattr(self.sim, 'render') and callable(self.sim.render):
             # Placeholder: Assumes render returns a list of images or handles num_envs
             try:
                 # Try rendering all envs if possible
                 return self.sim.render(mode='rgb_array')
             except:
                 # Fallback to rendering one by one (less efficient)
                 return [self.sim.render(mode='rgb_array', env_index=i) for i in range(self.num_envs)]
        else:
             return super().get_images()

# Removed gymnasium registration
