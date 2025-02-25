import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

from torch.utils.tensorboard import SummaryWriter

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
    n_episodes = 100
    env_name = "CustomEnv-v0"

    model_dir = os.path.join("models", env_name)
    log_dir = os.path.join("logs", env_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env = gym.make(env_name, render_mode="human")
    # env = gym.make(env_name, render_mode=None)

    # print("Checking custom env...")
    # check_env(env.unwrapped)
    # print("Checking complete.\n")

    writer = SummaryWriter(log_dir)
    best_score = 0
    best_ep = 0

    episode_identifier = f"{env_name}"  # =actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} layer1_size={layer1_size} layer2_size={layer2_size}"

    for i in range(n_episodes):
        obs = env.reset()[0]
        score = 0
        done = False

        # Take some random actions
        while not done:
            rand_action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(rand_action)
            score += reward

            if terminated or truncated:
                done = True

        writer.add_scalar(f"Run - {episode_identifier}", score, global_step=i)

        print(f"Episode: {i} score {score}")
        if i % 25 == 0 or i == n_episodes - 1:
            pass
            # agent.save_models()

        if score > best_score:
            # agent.save_models(best_models=True)
            best_score = score
            best_ep = i

    print("\nTraining complete.\n")
    print(f"Best score: {best_score} at episode: {best_ep}\n")
