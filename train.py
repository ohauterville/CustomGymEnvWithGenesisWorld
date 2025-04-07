import gymnasium as gym
from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
)  # Corrected import path

# from stable_baselines3.common.monitor import Monitor #remove Monitor
from torch.utils.tensorboard import SummaryWriter
import json
import sys
import argparse


# from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

# Import CustomEnv directly instead of relying on registration side-effect
from custom_env import CustomEnv


def train_sb3(
    # env_name removed, n_envs added
    n_envs=16, # Default value
    run_name="run_0",
    model_name="PPO",
    model_learning_rate=0.001,  # default was 0.001
    timesteps=100000,
    learning_sessions=1,
    # model_learning_starts=1000,
):
    # Where to store trained model and logs
    model_dir = os.path.join("models", run_name)
    log_dir = os.path.join("logs", run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nThe run name is:\n\n{run_name}\n")

    # Instantiate CustomEnv directly, passing n_envs
    env = CustomEnv(n_envs=n_envs)

    # Check environment properties
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    if model_name == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            device="cpu",  # Use "cuda" if you want to try the GPU, but it might be slower
            tensorboard_log=log_dir,
            learning_rate=model_learning_rate,
            ent_coef=0.01,
        )
    elif model_name == "TD3":
        model = TD3(
            "MlpPolicy",
            env,
            verbose=0,
            device="cuda",
            tensorboard_log=log_dir,
            learning_rate=model_learning_rate,
        )
    else:
        model = None
        print(f"Error. Model {model_name} not implemented.\nExiting.")
        sys.exit

    writer = SummaryWriter(log_dir=log_dir)
    # get_base_env needs the VecEnv instance
    base_env = get_base_env(env)
    # Your fixed configuration dictionary
    run_specs = {
        "env_class": "CustomEnv", # Use class name instead of env_name ID
        "num_envs": n_envs, # Log the number of envs used
        "run_name": run_name,
        "algorithm": model_name,
        "hyperparameters": {
            "learning_rate": model.learning_rate,
            "gamma": model.gamma,
            # Add other relevant model hyperparameters if needed
        },
        "reward_config": {
            "max_steps": base_env.max_steps,
            "min_dist_task_completion": base_env.min_dist_task_completion,
            "energy_penalty_weight": base_env.energy_penalty_weight,
            "time_penalty": base_env.time_penalty,
            "task_completion_bonus": base_env.task_completion_bonus,
            "end_ep_penalty": base_env.end_ep_penalty,
            "collision_penalty": base_env.collision_penalty,
            "max_collisions": base_env.max_collisions,
        },
        "notes": "Reach target task with updated reward structure.",  # Updated notes
    }

    config_json = json.dumps(run_specs, indent=2)
    tensorboard_text = f"```json\n{config_json}\n```"
    writer.add_text("Run_specs", tensorboard_text)
    writer.close()

    i = 0
    while learning_sessions > i:
        i += 1

        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)  # train
        model.save(os.path.join(model_dir, f"{run_name}_{timesteps*i}"))

    env.close()  # Close the direct env instance
    print(f"\nThe run name is:\n\n{run_name}\n")


def get_base_env(vec_env: VecEnv):  # Expect a VecEnv
    # Access the underlying sim environment from CustomEnv
    # Assuming CustomEnv is the direct VecEnv instance passed
    if isinstance(vec_env, CustomEnv):
        # Access the .sim attribute we defined in CustomEnv
        # Need to make sure CustomEnv actually has .sim accessible
        # It does, from __init__
        return vec_env.sim
    else:
        # This case should ideally not happen now, but keep as fallback
        # Attempt to get the attribute from the first environment if it's a standard VecEnv wrapper
        try:
            return vec_env.get_attr("sim")[0]
        except (AttributeError, IndexError):
            # If 'sim' doesn't exist or get_attr fails, return the base env
            current_env = vec_env
            while hasattr(current_env, "env"):
                current_env = current_env.env
            return current_env

    # The lines below were incorrectly indented remnants and are removed.
    # The logic is handled by the 'else' block above.


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--model", type=str, default="PPO", help="TD3 or PPO")
    parser.add_argument("--n_envs", type=int, default=16, help="Number of parallel environments")
    args = parser.parse_args()

    model_name = args.model
    n_envs = args.n_envs

    now = datetime.now()
    formatted_time = now.strftime("%m%d%H%M")

    run_name = formatted_time + "_" + model_name
    learning_sessions = 10

    train_sb3(
        n_envs=n_envs, # Pass n_envs
        run_name=run_name,
        learning_sessions=learning_sessions,
        model_name=model_name,
    )
