import genesis as gs
import numpy as np
import random
import torch


class GenesisWorld:
    def __init__(self, render_mode=None, max_steps=150):
        ########################## config #######################

        ########################## init ##########################
        if not gs._initialized:  # Prevent re-initialization
            gs.init(backend=gs.gpu)

        ########################## create a scene ##########################
        if render_mode == None:
            self.scene = gs.Scene(
                show_viewer=False,
                show_FPS=False,
                rigid_options=gs.options.RigidOptions(
                    # dt=0.01,
                ),
            )

        elif render_mode == "human":
            self.scene = gs.Scene(
                show_viewer=True,
                show_FPS=False,
                viewer_options=gs.options.ViewerOptions(
                    camera_pos=(3.5, -1.0, 2.5),
                    camera_lookat=(0.0, 0.0, 0.5),
                    camera_fov=40,
                    max_FPS=30,
                ),
                rigid_options=gs.options.RigidOptions(
                    dt=0.01,
                ),
            )
        else:
            print("render_mode ill defined.")

        ########################## entities ##########################
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        # The target
        self.target_pos = self.generate_target_pos()

        self.target = self.scene.add_entity(
            gs.morphs.Box(pos=self.target_pos, size=(0.1, 0.1, 0.1))
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
        self.max_steps = max_steps  # max_steps per episodes

        #### learning params
        self.tau = 0.5

        ########################## build ##########################
        self.n_envs = 1  # stick to 1 for now
        self.scene.build()

        ########################## get info #########################
        self.action_space_limits = self.robot_entity.get_dofs_limit()
        # self.observations_dims = 9

    def step(self, action):
        self.robot_entity.control_dofs_position(action, self.dofs_idx)
        self.scene.step()
        next_obs = self.get_observation()

        self.current_step += 1

        reward, terminated, truncated = self.compute_reward_function()

        # info
        info = {}

        return next_obs, reward, terminated, truncated, info

    def reset(self, seed=None):
        self.robot_entity.set_dofs_position(np.zeros(len(self.dofs_idx)), self.dofs_idx)

        random.seed(seed)

        self.scene.reset()
        obs = self.get_observation()
        self.current_step = 0
        info = {}
        return obs, info

    def get_observation(self, dofs_idx_local=None):
        if dofs_idx_local is not None:
            return (
                self.robot_entity.get_dofs_position(dofs_idx_local=dofs_idx_local)
                .cpu()
                .numpy()
            )
        else:
            return np.concatenate(
                [self.robot_entity.get_dofs_position().cpu().numpy(), self.target_pos]
            )

    def compute_reward_function(self, threshold=0.05, max_reward=500.0, c=0.1, d=2):
        # if n_envs > 1, change,  to do
        last_link_pos = self.robot_entity.get_links_pos()[-1, :].cpu().numpy()

        distance_to_target = np.linalg.norm(last_link_pos - self.target_pos)

        terminated = False
        truncated = False

        if distance_to_target < threshold:
            success_reward = max_reward  # * (1 - self.current_step / self.max_steps)
            reward = success_reward
            terminated = True
            return reward, terminated, truncated

        else:
            r_distance = -d * distance_to_target
            r_time = -c

            reward = r_distance + r_time
            if self.current_step > self.max_steps:
                truncated = True

            return reward, terminated, truncated

    def generate_target_pos(self):
        z = 0
        dist = 0

        while dist < 0.3:
            xy = np.random.uniform(low=-1.0, high=1.0, size=2)
            dist = np.sqrt(np.power(xy[0], 2) + np.power(xy[1], 2))

        return np.array([*xy, z])


### Unit testing
# if __name__ == "__main__":
#      pass