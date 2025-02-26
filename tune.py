import gymnasium as gym
from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os

from ray.tune import run
from ray import tune

env_name = "CustomEnv-v0"


def train_rl(config):
    import custom_env  # it is weird but we need it here

    env = gym.make(env_name)

    # Initialize PPO with hyperparameters from Ray Tune
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=config["lr"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        verbose=0,
        device="cuda",
    )

    model.learn(total_timesteps=100)  # Train for 50k steps

    # Evaluate performance (reward over episodes)
    mean_reward = evaluate_model(model, env)

    # Log the results for Ray Tune
    tune.report(mean_reward=mean_reward)


def evaluate_model(model, env, num_episodes=10):
    """Run the trained model and compute the mean reward."""
    total_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()[0]
        done, episode_reward = False, 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)

    return sum(total_rewards) / num_episodes


if __name__ == "__main__":

    search_space = {
        "lr": tune.loguniform(1e-5, 1e-3),  # Learning rate
        "batch_size": tune.choice([32, 64, 128, 256]),  # Batch sizes
        "gamma": tune.uniform(0.95, 0.999),  # Discount factor
    }

    analysis = tune.run(
        train_rl,  # Training function
        config=search_space,
        num_samples=20,  # Number of trials
        metric="mean_reward",  # Optimize for highest reward
        mode="max",  # Maximize reward
        resources_per_trial={"cpu": 2, "gpu": 0.5},  # Limit GPU usage
    )

    print("Best Hyperparameters:", analysis.best_config)
