import genesis as gs
import numpy as np
import random
import torch


class GenesisWorldEnv:
    def __init__(self, render_mode=None, max_steps=1000):
        ########################## config #######################

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

        self.env_size = 0.6  # the space around the robot
        ########################## entities ##########################
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        # The target
        self.target_pos = self.generate_target_pos(self.env_size)
        self.target = self.scene.add_entity(
            gs.morphs.Box(pos=self.target_pos, size=(0.05, 0.05, 0.05))
        )

        self.robot_entity = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
            ),
        )

        jnt_names = [
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
        self.dofs_idx = [
            self.robot_entity.get_joint(name).dof_idx_local for name in jnt_names
        ]

        self.current_step = 0
        self.collision_counts = 0

        #### learning params
        self.max_steps = max_steps  # max_steps per episodes
        self.min_dist_task_completion = 0.2
        self.distance_weight = 1.0
        self.task_completion_reward = 5
        self.end_ep_reward = -1
        self.collision_reward = -1
        self.max_collisions = 999  # if negative, the collision detection is not active

        ########################## build ##########################
        self.n_envs = 1  # stick to 1 for now
        self.scene.build()

        ########################## get info #########################
        self.episode_count = 0
        self.action_space_limits = self.robot_entity.get_dofs_limit()
        print(
            f"\nThe max time duration of an episode: {self.max_steps*self.scene.rigid_options.dt} seconds.\n"
        )

    def step(self, action):
        self.robot_entity.control_dofs_velocity(action)
        self.scene.step()
        next_obs = self.get_observation()

        self.current_step += 1

        reward, terminated, truncated = self.compute_reward_function(action=action)

        # info
        info = {}

        return next_obs, reward, terminated, truncated, info

    def reset(self, seed=None):
        # Add episode counter
        if hasattr(self, "episode_count"):
            self.episode_count += 1
        else:
            self.episode_count = 0

        random.seed(seed)

        self.scene.reset()
        obs = self.get_observation()
        self.current_step = 0
        info = {}
        return obs, info

    def get_observation(self):
        return np.concatenate(
            [
                self.robot_entity.get_dofs_position().cpu().numpy(),
                self.robot_entity.get_dofs_velocity().cpu().numpy(),
                self.robot_entity.get_links_pos()[-1, :].cpu().numpy(),
                self.target.get_pos().cpu().numpy(),
                np.array([self.compute_ee_target_distance()]),
            ]
        )

    def compute_reward_function(self, action=None):
        terminated = False
        truncated = False

        # Calculate distance to target
        distance_to_target = self.compute_ee_target_distance()

        # # Shaped distance reward (exponential penalty for distance)
        # r_distance = -self.distance_weight * (distance_to_target**2)
        r_distance = 1.0 / (1 + distance_to_target)
        r_distance *= r_distance
        # # Energy efficiency reward
        # velocity = self.robot_entity.get_dofs_velocity().cpu().numpy()
        # r_energy = -0.01 * np.sum(np.square(velocity))
        r_energy = -0.01 * np.sum(action**2)

        # Time penalty
        r_time = 0.0  # Small penalty for each timestep to encourage efficiency

        if distance_to_target < self.min_dist_task_completion:
            r_distance *= 2
            terminated = True

        # Collision penalty
        r_collision = 0
        if self.max_collisions >= 0:
            if any(
                x > 4
                for x in self.robot_entity.get_contacts(with_entity=self.plane)[
                    "link_b"
                ]
            ):
                self.collision_counts += 1
                r_collision = self.collision_reward
                if self.collision_counts > self.max_collisions:
                    # truncated = True
                    pass

        # End of episode handling
        r_end_episode = 0
        if self.current_step > self.max_steps:
            truncated = True
            r_end_episode = self.end_ep_reward

        # Combine all rewards
        reward = r_distance + r_energy + r_collision + r_end_episode + r_time 

        # Debug info for testing
        if self.mode == "test":
            if truncated:
                print("\nEpisode truncated.")
            if terminated:
                print("\nEpisode terminated.")

        return reward, terminated, truncated

    def compute_ee_target_distance(self):
        return np.linalg.norm(self.get_ee_pos() - self.target.get_pos().cpu().numpy())

    def get_ee_pos(self):
        return (
            self.robot_entity.get_joint("finger_joint1").get_pos().cpu().numpy()
            + self.robot_entity.get_joint("finger_joint2").get_pos().cpu().numpy()
        ) / 2
        # return (
        #     self.robot_entity.get_links_pos()[-1, :].cpu().numpy()
        #     + self.robot_entity.get_links_pos()[-2, :].cpu().numpy()
        # ) / 2

    def generate_target_pos(self, env_size):
        if self.mode == "train":
            # During training, gradually increase difficulty
            episode_num = getattr(self, "episode_count", 0)
            min_dist = 0.2 + min(
                0.2, (episode_num / 1000) * 0.2
            )  # Gradually increase min distance
            max_dist = 0.3 + min(
                0.3, (episode_num / 1000) * 0.3
            )  # Gradually increase max distance
        else:
            min_dist = 0.4
            max_dist = env_size

        while True:
            xy = np.random.uniform(low=-env_size, high=env_size, size=2)
            dist = np.sqrt(np.power(xy[0], 2) + np.power(xy[1], 2))
            if min_dist < dist < max_dist:
                break

        return np.array([*xy, 0])


### Unit testing
# if __name__ == "__main__":
#      pass
