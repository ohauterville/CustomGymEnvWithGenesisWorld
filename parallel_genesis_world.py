import genesis as gs
import numpy as np
import random
import torch


class GenesisWorldEnv:
    def __init__(self, render_mode=None, max_steps=300, n_envs=2):
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

        self.env_size = 0.6  # the space around the robot
        ########################## entities ##########################
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        # The target
        self.target_pos = self.generate_target_pos(self.env_size)
        # self.target_pos = [0.5, 0.4, 0.05]
        self.target = self.scene.add_entity(
            gs.morphs.Box(pos=self.target_pos.cpu().numpy(), size=(0.05, 0.05, 0.05))
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

        ########################## build ##########################
        self.n_envs = n_envs  # stick to 1 for now
        self.scene.build(
            n_envs=n_envs, env_spacing=(2 * self.env_size, 2 * self.env_size)
        )

        ########################## get info #########################
        self.action_space_limits = self.robot_entity.get_dofs_limit()
        # self.observations_dims = 9

    def step(self, action):
        action = torch.tensor(
            action, device=gs.device, dtype=torch.float32
        )  # Convert NumPy → PyTorch
        self.robot_entity.control_dofs_velocity(action)
        self.scene.step()
        next_obs = self.get_observation()

        self.current_step += 1

        reward, terminated, truncated = self.compute_reward_function()

        # info
        info = {}

        return next_obs.cpu().numpy(), reward, terminated, truncated, info

    def reset(self, seed=None):
        self.robot_entity.set_dofs_velocity(
            torch.zeros(len(self.dofs_idx), device=gs.device)
        )

        random.seed(seed)

        self.scene.reset()
        obs = self.get_observation()
        self.current_step = 0
        info = {}

        return obs[0, :].cpu().numpy(), info

    def get_observation(self):
        obs = torch.cat(
            [
                self.robot_entity.get_links_pos()[
                    :, -1, :
                ],  # dims = [n_envs, n_links, 3]
                self.robot_entity.get_dofs_velocity(),
                self.target.get_pos(),
            ],
            dim=1,
        ).to(gs.device)

        return obs

    def compute_reward_function(self, threshold=0.1, max_reward=300, c=0.01, d=0.1):
        # if n_envs > 1, change,  to do
        distance_to_target = self.compute_ee_target_distance()

        terminated = False
        truncated = False

        if distance_to_target < threshold:
            reward = max_reward  # * (1 - self.current_step / self.max_steps)
            terminated = True
            return reward, terminated, truncated

        else:
            r_distance = -d * torch.exp(distance_to_target)
            r_time = -c

            reward = r_distance + r_time
            if self.current_step > self.max_steps:
                truncated = True
                reward -= 100

            contacts = torch.tensor(
                self.robot_entity.get_contacts(with_entity=self.plane)["link_b"],
                dtype=torch.float32,
            )
            if torch.any(contacts > 4):
                truncated = True
                reward -= 100

            return reward, terminated, truncated

    def compute_ee_target_distance(self):
        last_link_pos = self.robot_entity.get_links_pos()[:, -1, :]
        return torch.norm(last_link_pos - self.target.get_pos())

    def generate_target_pos(self, env_size):
        z = 0
        dist = 0

        while dist < 0.2:
            xy = (2 * env_size) * torch.rand(
                2, device=gs.device
            ) - env_size  # Torch equivalent of np.uniform
            dist = torch.norm(xy)  # Torch equivalent of np.sqrt(x^2 + y^2)

        return torch.cat((xy, torch.tensor([z], device=gs.device)))


### Unit testing
# if __name__ == "__main__":
#      pass
