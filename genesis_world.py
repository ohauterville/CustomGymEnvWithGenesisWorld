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
                    max_FPS=20,
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
        self.distance_weight = 0.1
        self.task_completion_reward = 10
        self.end_ep_reward = -10
        self.collision_reward = -10
        self.max_collisions = 0

        ########################## build ##########################
        self.n_envs = 1  # stick to 1 for now
        self.scene.build()

        ########################## get info #########################
        self.action_space_limits = self.robot_entity.get_dofs_limit()
        print(f"\nThe max time duration of an episode: {self.max_steps*self.scene.rigid_options.dt}\n")

    def step(self, action):
        self.robot_entity.control_dofs_velocity(action)
        self.scene.step()
        next_obs = self.get_observation()

        self.current_step += 1

        reward, terminated, truncated = self.compute_reward_function()

        # info
        info = {}

        return next_obs, reward, terminated, truncated, info

    def reset(self, seed=None):
        self.robot_entity.set_dofs_velocity(np.zeros(len(self.dofs_idx)))

        random.seed(seed)

        self.scene.reset()
        obs = self.get_observation()
        self.current_step = 0
        info = {}
        return obs, info

    def get_observation(self):
        return np.concatenate(
            [
                self.robot_entity.get_links_pos()[-1, :].cpu().numpy(),
                self.robot_entity.get_dofs_velocity().cpu().numpy(),
                self.target.get_pos().cpu().numpy(),
            ]
        )

    def compute_reward_function(self):
        terminated = False
        truncated = False
        r_success = 0
        r_distance = 0
        r_collision = 0
        r_end_episode = 0

        ##### r_distance #####
        distance_to_target = self.compute_ee_target_distance()
        r_distance = -self.distance_weight * distance_to_target

        ##### r_time ####
        if self.current_step > self.max_steps:
            truncated = True
            r_end_episode = self.end_ep_reward

        ##### r_collision #####
        if any(
            x > 4
            for x in self.robot_entity.get_contacts(with_entity=self.plane)["link_b"]
        ):
            self.collision_counts += 1
            r_collision = self.collision_reward

            if self.collision_counts > self.max_collisions:
                truncated = True

        ##### r_success #####
        if distance_to_target < self.min_dist_task_completion and not truncated:
            r_success = self.task_completion_reward
            terminated = True

        ##### rewards #######
        reward = r_success + r_distance + r_collision + r_end_episode

        #################### for testing ################
        if self.mode == "test":
            if truncated:
                print("\nEpisode truncated.")
            if terminated:
                print("\nEpisode terminated.")

        return reward, terminated, truncated

    def compute_ee_target_distance(self):
        last_link_pos = self.robot_entity.get_links_pos()[-1, :].cpu().numpy()
        return np.linalg.norm(last_link_pos - self.target.get_pos().cpu().numpy())

    def generate_target_pos(self, env_size):
        z = 0
        dist = 0

        while dist < 0.2:
            xy = np.random.uniform(low=-env_size, high=env_size, size=2)
            dist = np.sqrt(np.power(xy[0], 2) + np.power(xy[1], 2))

        return np.array([*xy, z])


### Unit testing
# if __name__ == "__main__":
#      pass
