import gymnasium as gym
from stable_baselines3 import A2C, PPO, TD3
import os
import custom_env # Even though we don't use this class here, we should include it here so that it registers the WarehouseRobot environment.
from torch.utils.tensorboard import SummaryWriter

# Where to store trained model and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Train using StableBaseline3. Lots of hardcoding for simplicity i.e. use of the A2C (Advantage Actor Critic) algorithm.
def train_sb3(env_name="CustomEnv-v0", run_name="run_0", lr=0.001, timesteps=25000):
    env = gym.make(env_name)

    # Use Advantage Actor Critic (A2C) algorithm.
    # Use MlpPolicy for observation space 1D vector.
    model = PPO('MlpPolicy', env, learning_rate=lr, verbose=1, device='cuda', tensorboard_log=log_dir)

    TIMESTEPS = timesteps
    i = 0
    while True:
        i += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False) # train
        model.save(os.path.join(model_dir, f"{run_name}_{TIMESTEPS*i}")) # Save a trained model every TIMESTEPS

# Test using StableBaseline3. Lots of hardcoding for simplicity.
def test_sb3(env_name="CustomEnv-v0",  run_name="run_0", episode=1000, render=True):

    env = gym.make('CustomEnv-v0', render_mode='human' if render else None)

    # Load model
    model = PPO.load(os.path.join(model_dir, f'{run_name}_{episode}'), env=env)

    # Run a test
    obs = env.reset()[0]
    terminated = False
    truncated = False
    score = 0

    while True:
        action, _ = model.predict(observation=obs, deterministic=True) # Turn on deterministic, so predict always returns the same behavior
        obs, reward, terminated, truncated, _ = env.step(action)
        score += reward

        if terminated or truncated:
            print(f"Score: {score}")
            break


if __name__ == "__main__":
    env_name = "CustomEnv-v0"

    # train_sb3(env_name=env_name, run_name="PPO_run_1")

    test_sb3(env_name=env_name, run_name="PPO_run_1", episode=180000)