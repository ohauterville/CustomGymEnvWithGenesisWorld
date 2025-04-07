from typing import Optional, Tuple, List, Dict, Any
import genesis as gs
import numpy as np
import random
import torch
import time

class GenesisWorldEnv:
    """
    A custom environment for a Franka Emika Panda robot learning to reach a target
    using the Genesis physics simulator.

    Args:
        render_mode (Optional[str]): The rendering mode. Can be 'human' to show the
                                     viewer or None for headless operation. Defaults to None.
        max_steps (int): The maximum number of steps allowed per episode. Defaults to 1000.
    """

    def __init__(
        self, n_envs: int = 1, render_mode: Optional[str] = None, max_steps: int = 1000
    ):
        ########################## config #######################
        # Note: Consider moving these hardcoded values to a configuration dictionary
        # or object for better flexibility.

        ########################## init ##########################
        if not gs._initialized:  # Prevent re-initialization
            gs.init(backend=gs.gpu)

        ########################## create a scene ##########################
        self.mode = "train"
        if render_mode == None:
            self.scene = gs.Scene(
                show_viewer=False,
                show_FPS=False,
                rigid_options=gs.options.RigidOptions(
                    dt=0.01,
                ),
            )

        elif render_mode == "human":
            self.mode = "test"
            self.scene = gs.Scene(
                show_viewer=True,
                show_FPS=False,
                viewer_options=gs.options.ViewerOptions(
                    camera_pos=(3.5, -1.0, 2.5),
                    camera_lookat=(0.0, 0.0, 0.5),
                    camera_fov=40,
                    max_FPS=60,
                ),
                rigid_options=gs.options.RigidOptions(
                    dt=0.01,
                ),
            )
        else:
            print("render_mode ill defined.")

        self.env_size: float = 0.6  # the space around the robot
        ########################## entities ##########################
        self.plane: gs.Entity = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        # The target
        self.n_envs: int = n_envs  # Number of parallel environments
        self.target_pos: np.ndarray = self.generate_target_pos(
            self.env_size, self.n_envs
        )
        self.target: gs.Entity = self.scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05))
        )

        self.robot_entity: gs.Entity = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
            ),
        )

        jnt_names: List[str] = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ]
        # Indices for the controllable degrees of freedom (DOFs)
        self.dofs_idx: List[int] = [
            self.robot_entity.get_joint(name).dof_idx_local for name in jnt_names
        ]

        self.current_step: np.ndarray = np.zeros(self.n_envs, dtype=int)
        self.collision_counts: np.ndarray = np.zeros(self.n_envs, dtype=int)

        #### learning params
        self.max_steps: int = max_steps  # max_steps per episodes
        self.min_dist_task_completion: float = 0.1
        # --- Reward Coefficients (Consider moving to a config dict) ---
        # self.distance_weight: float = 1.0 # Example if using linear distance reward
        self.energy_penalty_weight: float = 0.001  # Weight for action penalty
        self.time_penalty: float = -0.1  # Constant penalty per step
        self.task_completion_bonus: float = 100.0  # Large bonus upon success
        self.end_ep_penalty: float = -10.0  # Penalty for episode truncation (timeout)
        self.collision_penalty: float = -1.0  # Penalty for each collision detected
        self.max_collisions: int = (
            5  # Max allowed collisions before truncation (-1 disables), Example: set to 5
        )

        ########################## build ##########################
        self.scene.build(n_envs=self.n_envs)

        # # *** ADD THIS SECTION FOR EXPLICIT WAIT ***
        # if render_mode == "human":
        #     print("Waiting for Pyglet initialization...")
        #     while not self.scene.visualizer._viewer._initialized_event.is_set():
        #         time.sleep(0.1)  # Check every 0.1 seconds
        #     print("Pyglet initialization complete.")
        # # *** END OF WAIT SECTION ***

        self.target.set_pos(
            torch.tensor(self.target_pos, dtype=torch.float32, device=gs.device)
        )

        ########################## get info #########################
        self.episode_count: int = 0
        # Get joint limits for potential use in defining action space
        self.action_space_limits: torch.Tensor = self.robot_entity.get_dofs_limit()
        print(
            f"\nThe max time duration of an episode: {self.max_steps*self.scene.rigid_options.dt:.2f} seconds.\n"
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Advances the environment by one timestep.

        Args:
            action (np.ndarray): The action to apply (likely joint velocities).

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]: A tuple containing:
                - next_obs (np.ndarray): The observation after the step.
                - reward (float): The reward received for the step.
                - terminated (bool): Whether the episode ended due to task completion.
                - truncated (bool): Whether the episode ended due to timeout or other limits.
                - info (Dict): Additional information (currently empty).
        """
        # Apply action (control joint velocities)
        # Note: Genesis expects torch tensors for control, consider converting action if needed
        # or ensure the agent outputs tensors directly. For now, assuming action is compatible.
        action_tensor = torch.tensor(
            action, dtype=torch.float32, device=gs.device
        )  # Removed parentheses from gs.device
        self.robot_entity.control_dofs_velocity(action_tensor)

        # Step the physics simulation
        self.scene.step()

        # Get the resulting observation
        next_obs = self.get_observation()

        self.current_step += 1

        # Calculate reward and termination conditions
        reward, terminated, truncated = self.compute_reward_function(action=action)

        # info dictionary (can be populated with debug info if needed)
        info = {}

        return next_obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Resets the environment to the start of a new episode.

        Args:
            seed (Optional[int]): An optional seed for reproducibility. Defaults to None.

        Returns:
            Tuple[np.ndarray, Dict]: A tuple containing:
                - obs (np.ndarray): The initial observation after reset.
                - info (Dict): Additional information (currently empty).
        """
        # Add episode counter
        if hasattr(self, "episode_count"):
            self.episode_count += 1
        else:
            self.episode_count = 0  # Initialize if it's the first reset

        # Seed the random number generator if a seed is provided
        # Note: For full reproducibility with external libraries (like RL frameworks),
        # integrating with their seeding mechanism (e.g., Gymnasium's) is recommended.
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            # Consider seeding torch as well if stochastic operations are used elsewhere:
            # torch.manual_seed(seed)
            # if torch.cuda.is_available():
            #     torch.cuda.manual_seed_all(seed)
            # --- Gymnasium Integration Note ---
            # If inheriting from gymnasium.Env, call super().reset(seed=seed) here.
            # It handles seeding self.np_random. Then use self.np_random for
            # all stochastic operations below (e.g., self.np_random.uniform(...)).
            # super().reset(seed=seed)
            pass  # Placeholder if not using Gymnasium's seeding

        # Reset the simulation scene (resets physics state, potentially robot pose/velocity)
        self.scene.reset()

        # --- Optional: Randomize Initial Robot Pose ---
        initial_qpos = (
            self.robot_entity.get_dofs_position()
        )  # Get default pose after scene reset
        noise_scale = 0.02  # Adjust scale as needed
        noise = np.random.uniform(
            low=-noise_scale, high=noise_scale, size=initial_qpos.shape
        )
        # Ensure noise respects joint limits if necessary
        randomized_qpos = initial_qpos.cpu().numpy() + noise
        # Clamp to limits if needed: np.clip(randomized_qpos, self.action_space_limits[0].cpu().numpy(), self.action_space_limits[1].cpu().numpy())
        self.robot_entity.set_dofs_position(
            torch.tensor(randomized_qpos, dtype=torch.float32, device=gs.device)
        )
        self.robot_entity.set_dofs_velocity(
            torch.zeros_like(self.robot_entity.get_dofs_velocity())
        )  # Ensure zero initial velocity

        # Explicitly reset target position after scene reset
        self.target_pos = self.generate_target_pos(
            self.env_size, self.n_envs
        )  # Generate new target position
        self.target.set_pos(
            torch.tensor(self.target_pos, dtype=torch.float32, device=gs.device)
        )  # Set in simulation (Removed parentheses from gs.device)

        # Reset step counter and collision counter for the new episode
        self.current_step = np.zeros(self.n_envs, dtype=int)
        self.collision_counts = np.zeros(self.n_envs, dtype=int)
        # Reset any other episode-specific state variables here
        # e.g., if using potential-based rewards:
        # self.last_distance_to_target = self.compute_ee_target_distance()

        # Calculate initial distance AFTER setting state
        initial_distance = self.compute_ee_target_distance()
        print(f"DEBUG: Env Reset - Initial Distances: {initial_distance}") # Print for all envs

        # Get the initial observation AFTER setting initial state and target
        obs = self.get_observation()

        # Populate info dictionary with useful reset information
        info = {
            "target_position": self.target_pos,
            # Add other relevant reset info if needed
        }

        return obs, info

    def get_observation(self) -> np.ndarray:
        """
        Retrieves the current observation state of the environment.

        The observation includes:
        - Robot joint positions
        - Robot joint velocities
        - End-effector position (approximated)
        - Target position
        - Distance between end-effector and target

        Returns:
            np.ndarray: The concatenated observation vector.
        """
        ee_pos = self.get_ee_pos()  # Calculate once
        target_pos = self.target.get_pos().cpu().numpy()  # Get once
        ee_target_dist = np.linalg.norm(ee_pos - target_pos, axis=1, keepdims=True)

        # Add zeros to have 30 values
        zeros = np.zeros((self.n_envs, 5))

        return np.concatenate(
            [
                self.robot_entity.get_dofs_position().cpu().numpy(),  # Joint positions
                self.robot_entity.get_dofs_velocity().cpu().numpy(),  # Joint velocities
                ee_pos,  # End-effector position
                target_pos,  # Target position
                ee_target_dist,  # Distance to target
                zeros,
            ],
            axis=1,
        ).astype(np.float32)

    def compute_reward_function(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the reward for the current state and action, and determines
        termination conditions.

        Args:
            action (np.ndarray): The action taken in the current step (used for energy penalty).

        Returns:
            Tuple[float, bool, bool]: A tuple containing:
                - reward (float): The calculated reward for the step.
                - terminated (bool): Whether the episode should terminate (task success).
                - truncated (bool): Whether the episode should truncate (timeout, limits).
        """
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)

        # Calculate distance to target
        distance_to_target = self.compute_ee_target_distance()

        # --- Reward Components ---
        # Note: The relative scaling of these components is crucial and often requires tuning.

        # 1. Shaped distance reward: Higher reward for being closer.
        #    Current: 0.5 / (1 + dist)^2. Encourages getting very close.
        r_distance = 0.5 / (
            1 + distance_to_target + 1e-6
        )  # Add epsilon for stability if dist can be 0
        r_distance *= r_distance
        # --- Alternative Distance Rewards (for experimentation) ---
        # a) Negative Euclidean distance:
        # r_distance = -self.distance_weight * distance_to_target
        # b) Exponential reward:
        # k = 10 # Example scaling factor
        # r_distance = np.exp(-k * distance_to_target)

        # 2. Energy efficiency penalty: Penalize large actions (squared velocity commands).
        r_energy = -self.energy_penalty_weight * np.sum(action**2, axis=1)

        # 3. Time penalty: Small constant penalty per step to encourage speed.
        r_time = np.full(self.n_envs, self.time_penalty)

        # 4. Task Completion Bonus & Termination:
        r_completion = np.zeros(self.n_envs)
        terminated = distance_to_target < self.min_dist_task_completion
        r_completion[terminated] = self.task_completion_bonus  # Use defined bonus

        # 5. Collision Penalty & Potential Truncation:
        r_collision = np.zeros(self.n_envs)
        if self.max_collisions >= 0:  # Only check if collision limit is active
            # Check contacts between robot links and the plane
            # The condition `any(x > 4 ...)` needs clarification based on the MJCF model.
            # Assuming indices > 4 correspond to relevant robot links colliding with the plane (index 0?).
            contacts = self.robot_entity.get_contacts(with_entity=self.plane)
            # Ensure 'link_b' exists and is not empty before checking its contents
            if contacts and "link_b" in contacts and len(contacts["link_b"]) > 0:
                # Check if any link index in 'link_b' (colliding robot part) is greater than 4
                # Note: Ensure contacts['link_b'] contains numerical indices
                try:
                    for i in range(self.n_envs):
                        if any(int(link_idx) > 4 for link_idx in contacts["link_b"][i]):
                            self.collision_counts[i] += 1
                            r_collision[i] = (
                                self.collision_penalty
                            )  # Use defined penalty
                            # Optional: Truncate if max collisions exceeded
                            if self.collision_counts[i] > self.max_collisions:
                                truncated[i] = True  # Truncate based on collision count
                                # Optionally add an extra penalty for truncation due to collision
                                # r_collision += self.end_ep_penalty / 2 # Example
                except (ValueError, TypeError) as e:
                    # Handle cases where link_idx might not be convertible to int
                    print(
                        f"Warning: Could not process collision link index: {e}. Contact data: {contacts['link_b']}"
                    )
                    pass  # Or apply a default penalty/action
                    r_collision = np.full(
                        self.n_envs, self.collision_penalty
                    )  # Use defined penalty
                    # Optional: Truncate if max collisions exceeded
                    truncated = self.collision_counts > self.max_collisions
                    # Optionally add an extra penalty for truncation due to collision
                    # r_collision += self.end_ep_penalty / 2 # Example

        # 6. End of Episode Penalty & Truncation (Timeout):
        r_end_episode = np.zeros(self.n_envs)
        # Check for timeout ONLY if not already terminated or truncated by collision
        timeout = np.logical_and(np.logical_not(terminated), np.logical_not(truncated))
        timeout = np.logical_and(timeout, self.current_step >= self.max_steps)
        truncated[timeout] = True  # End episode due to time limit
        r_end_episode[timeout] = self.end_ep_penalty  # Use defined penalty

        # --- Total Reward ---
        reward = (
            r_distance + r_energy + r_time + r_completion + r_collision + r_end_episode
        )

        # Debug info during testing mode
        if self.mode == "test":
            # print(f"Step: {self.current_step}, Dist: {distance_to_target:.3f}, Reward: {reward:.3f}")
            for i in range(self.n_envs):
                if truncated[i]:
                    print(f"\nEpisode {i} truncated at step {self.current_step[i]}.")
                if terminated[i]:
                    print(
                        f"\nEpisode {i} terminated successfully at step {self.current_step[i]}!"
                    )

        return reward, terminated, truncated

    def compute_ee_target_distance(self) -> np.ndarray:
        """Calculates the Euclidean distance between the end-effector and the target."""
        ee_pos = self.get_ee_pos()
        target_pos = self.target.get_pos().cpu().numpy()
        return np.linalg.norm(ee_pos - target_pos, axis=1)

    def get_ee_pos(self) -> np.ndarray:
        """
        Calculates the position of the end-effector.

        Currently defined as the midpoint between the two finger joints.
        Consider verifying if the MJCF model provides a more direct 'site' or link
        for the end-effector.

        Returns:
            np.ndarray: The (x, y, z) coordinates of the end-effector.
        """
        # Get positions of the two finger joints
        finger1_pos = (
            self.robot_entity.get_joint("finger_joint1").get_pos().cpu().numpy()
        )
        finger2_pos = (
            self.robot_entity.get_joint("finger_joint2").get_pos().cpu().numpy()
        )
        # Calculate the midpoint
        ee_pos = (finger1_pos + finger2_pos) / 2.0
        return ee_pos
        # Alternative using last two links (commented out in original):
        # link_positions = self.robot_entity.get_links_pos().cpu().numpy()
        # ee_pos = (link_positions[-1, :] + link_positions[-2, :]) / 2.0
        # return ee_pos

    def generate_target_pos(self, env_size: float, n_envs: int) -> np.ndarray:
        """
        Generates a random target position within the specified environment size,
        respecting minimum and maximum distance constraints.

        In 'train' mode, the distance constraints might gradually increase based
        on the episode count, potentially implementing curriculum learning.

        Args:
            env_size (float): The radius defining the boundary for target generation.
            n_envs (int): The number of environments.

        Returns:
            np.ndarray: The (x, y, z) coordinates for the new target position (z is always 0).
        """
        target_positions = np.zeros((n_envs, 3), dtype=np.float32)

        if self.mode == "train":
            # Curriculum Learning: Gradually increase difficulty during training
            episode_num = getattr(self, "episode_count", 0)
            # Increase min/max distance based on episode number, capping the increase
            min_dist = 0.2 + min(0.2, (episode_num / 1000) * 0.2)
            max_dist = 0.3 + min(0.3, (episode_num / 1000) * 0.3)
        else:  # 'test' mode or other modes
            # Fixed, potentially harder range for testing
            min_dist = 0.4
            max_dist = env_size

        for i in range(n_envs):
            # Generate random (x, y) coordinates until they fall within the desired distance range
            while True:
                # Sample x, y uniformly within [-env_size, env_size]
                xy = np.random.uniform(low=-env_size, high=env_size, size=2)
                # Calculate distance from origin (0, 0)
                dist = np.sqrt(np.power(xy[0], 2) + np.power(xy[1], 2))
                # Check if the distance is within the allowed range
                if min_dist < dist < max_dist:
                    break  # Valid position found

            # Return the [x, y, z] position (z is 0 for the plane)
            target_positions[i] = np.array([xy[0], xy[1], 0.0], dtype=np.float32)

        return target_positions
