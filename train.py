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
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


# Train using StableBaseline3. Lots of hardcoding for simplicity i.e. use of the A2C (Advantage Actor Critic) algorithm.
def train_sb3(env_name="CustomEnv-v0", run_name="run_0", lr=0.001, timesteps=25000):
    # env = gym.make(env_name)
    env = DummyVecEnv([lambda: Monitor(gym.make(env_name))])

    # Check environment properties
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    # Use Advantage Actor Critic (A2C) algorithm.
    # Use MlpPolicy for observation space 1D vector.
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        verbose=1,
        device="cuda",
        tensorboard_log=log_dir,
    )

    TIMESTEPS = timesteps
    i = 0
    while True:
        i += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)  # train
        model.save(
            os.path.join(model_dir, f"{run_name}_{TIMESTEPS*i}")
        )  # Save a trained model every TIMESTEPS


if __name__ == "__main__":
    env_name = "CustomEnv-v0"

    train_sb3(env_name=env_name, run_name="PPO_run_1")
