import gymnasium as gym
from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import custom_env  # Even though we don't use this class here, we should include it here so that it registers the WarehouseRobot environment.
from torch.utils.tensorboard import SummaryWriter

# Where to store trained model and logs
model_dir = "models"
log_dir = "logs"
# os.makedirs(model_dir, exist_ok=True)
# os.makedirs(log_dir, exist_ok=True)


# Test using StableBaseline3. Lots of hardcoding for simplicity.
def test_sb3(
    env_name="CustomEnv-v0",
    run_name="run_0",
    episode_to_load=1000,
    nb_tests=1,
    render=True,
):

    env = gym.make(env_name, render_mode="human" if render else None)

    # Load model
    model = PPO.load(os.path.join(model_dir, f"{run_name}_{episode_to_load}"), env=env)

    for _ in range(nb_tests):
        # Run a test
        obs = env.reset()[0]
        terminated = False
        truncated = False
        score = 0

        while True:
            action, _ = model.predict(
                observation=obs, deterministic=True
            )  # Turn on deterministic, so predict always returns the same behavior
            obs, reward, terminated, truncated, _ = env.step(action)
            score += reward

            if terminated or truncated:
                print(f"Score: {score}")
                break


if __name__ == "__main__":
    env_name = "CustomEnv-v0"

    test_sb3(env_name=env_name, run_name="PPO_run_1", episode_to_load=425000)
