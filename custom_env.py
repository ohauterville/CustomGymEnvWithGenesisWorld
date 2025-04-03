import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import VecEnv  # Import VecEnv
from genesis_world import GenesisWorldEnv  # Import your environment class
import numpy as np

# Removed Dict from import as we'll use built-in dict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Type

# import time # Remove time


class CustomEnv(VecEnv):  # Inherit from VecEnv
    def __init__(self, render_mode=None, n_envs=1):
        self.sim = GenesisWorldEnv(render_mode=render_mode, n_envs=n_envs)
        # VecEnv requires num_envs, observation_space, action_space at init
        # Important: SB3 VecEnv expects spaces *without* the n_envs dimension
        obs_shape = (30,)  # Shape for a single environment
        action_shape = (9,)  # Shape for a single environment
        single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        single_action_space = gym.spaces.Box(
            low=-1, high=1, shape=action_shape, dtype=np.float32
        )

        super().__init__(self.sim.n_envs, single_observation_space, single_action_space)

        # Store sim attributes after super().__init__
        self.max_steps = self.sim.max_steps
        self.min_dist_task_completion = self.sim.min_dist_task_completion
        self.energy_penalty_weight = self.sim.energy_penalty_weight
        self.time_penalty = self.sim.time_penalty
        self.task_completion_bonus = self.sim.task_completion_bonus
        self.end_ep_penalty = self.sim.end_ep_penalty
        self.collision_penalty = self.sim.collision_penalty
        self.max_collisions = self.sim.max_collisions
        self._actions: np.ndarray  # To store actions for step_wait

    # Implement VecEnv abstract methods
    def step_async(self, actions: np.ndarray) -> None:
        # Store actions, actual step happens in step_wait
        self._actions = actions

    def step_wait(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:  # Changed Dict to dict
        # Use the stored actions to step the underlying sim
        # Note: GenesisWorldEnv.step likely returns arrays for reward, terminated, truncated
        next_obs, rewards, terminated, truncated, info = self.sim.step(
            self._actions
        )  # Assuming reward is plural now

        # SB3 VecEnv expects 'infos' to be a list of dicts, one per env.
        # If self.sim.step returns a single dict with array values, convert it.
        # Assuming info is a dict like {'key1': [val1_env1, val1_env2], 'key2': [val2_env1, val2_env2]}
        # Or potentially already a list of dicts if GenesisWorldEnv is well-behaved.
        # Let's assume we need to convert it for robustness.
        # If info is already a list of dicts, this might need adjustment based on GenesisWorldEnv's actual return.
        infos = [{} for _ in range(self.num_envs)]
        for key, value in info.items():
            if isinstance(value, (list, np.ndarray)) and len(value) == self.num_envs:
                for i in range(self.num_envs):
                    infos[i][key] = value[i]
            else:
                # If a value isn't per-env, broadcast it (e.g., shared info)
                for i in range(self.num_envs):
                    infos[i][key] = value

        # Ensure dones are boolean arrays
        terminated = np.array(terminated, dtype=bool)
        truncated = np.array(truncated, dtype=bool)
        dones = terminated | truncated  # Combine for easier iteration

        # Handle automatic reset and logging episode stats
        for i in range(self.num_envs):
            if dones[i]:
                # SB3 expects 'final_observation' and 'final_info' in the info dict
                # when an episode ends. Capture them *before* potential reset overwrites next_obs.
                # Check if the original 'info' dict from self.sim.step contained them.
                # If not, use the current next_obs[i] as final_observation.
                if "final_observation" not in infos[i]:
                    infos[i]["final_observation"] = next_obs[i].copy()
                if "final_info" not in infos[i]:
                    # Include any other relevant info from the original dict if needed
                    original_env_info = {
                        k: v[i]
                        for k, v in info.items()
                        if isinstance(v, (list, np.ndarray)) and len(v) == self.num_envs
                    }
                    infos[i]["final_info"] = original_env_info

                # Reset specific environment 'i' if GenesisWorldEnv supports it,
                # otherwise, reset all and update the observation for this env.
                # Assuming self.sim.reset() resets all:
                # Note: This might be inefficient if only one env needs reset.
                # A better GenesisWorldEnv would allow resetting individual envs.
                # We only reset if *any* env is done, to avoid unnecessary resets.
                # This reset logic might need refinement based on self.sim behavior.
                # Let's assume reset happens implicitly or is handled by SB3's buffer logic
                # based on the 'dones' signal and 'final_observation'/'final_info'.
                # SB3's OnPolicyAlgorithm handles the reset based on dones.
                pass  # Rely on SB3's handling via dones signal

        # SB3 expects a single 'dones' array
        return next_obs, rewards, dones, infos

    def reset(self) -> np.ndarray:  # VecEnv reset returns only observations
        # Assuming self.sim.reset() returns obs, info
        obs, _info = self.sim.reset()
        return obs

    def close(self) -> None:
        self.sim.close()

    # Implement other VecEnv methods (can often be simple pass-throughs or basic implementations)
    def get_attr(
        self, attr_name: str, indices: Optional[Union[int, Sequence[int]]] = None
    ) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        # Simplified: assumes attribute exists on self.sim and is same across envs or handles indices if needed
        target_obj = self.sim
        if indices is None:
            indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            indices = [indices]

        try:
            attr = getattr(target_obj, attr_name)
            # If the attribute itself is array-like and per-env, return the indexed values
            if isinstance(attr, (list, np.ndarray)) and len(attr) == self.num_envs:
                return [attr[i] for i in indices]
            else:
                # Otherwise, assume it's a shared attribute and return it for each index
                return [attr for _ in indices]
        except AttributeError:
            raise AttributeError(f"Attribute {attr_name} not found in {target_obj}")

    def set_attr(
        self,
        attr_name: str,
        value: Any,
        indices: Optional[Union[int, Sequence[int]]] = None,
    ) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        # Simplified: assumes attribute exists on self.sim
        target_obj = self.sim
        if indices is None:
            indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            indices = [indices]

        # Simple approach: set the attribute on the main sim object.
        # This might not work correctly if the attribute needs to be per-environment.
        # Requires knowledge of how self.sim handles attributes.
        if hasattr(target_obj, attr_name):
            setattr(target_obj, attr_name, value)
        else:
            raise AttributeError(f"Attribute {attr_name} not found in {target_obj}")

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Optional[Union[int, Sequence[int]]] = None,
        **method_kwargs,
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        # Simplified: calls method on self.sim
        target_obj = self.sim
        if indices is None:
            indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            indices = [indices]

        try:
            method = getattr(target_obj, method_name)
            # Simple approach: call method once. Assumes method handles vectorization or is shared.
            result = method(*method_args, **method_kwargs)
            # Assume result applies to all envs, return list
            return [result for _ in indices]
        except AttributeError:
            raise AttributeError(f"Method {method_name} not found in {target_obj}")

    def env_is_wrapped(
        self,
        wrapper_class: Type[gym.Wrapper],
        indices: Optional[Union[int, Sequence[int]]] = None,
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        # Since we wrap self.sim directly, check if self.sim is wrapped.
        # This is a simplification. A real VecEnv might have individual wrappers.
        if indices is None:
            indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            indices = [indices]

        # Check if self.sim itself is an instance of wrapper_class
        return [isinstance(self.sim, wrapper_class) for _ in indices]


register(
    id="CustomEnv-v0",
    entry_point="custom_env:CustomEnv",  # Corrected entry point
    max_episode_steps=1000,  # Or your desired max steps
)
