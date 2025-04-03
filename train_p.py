import gymnasium as gym
from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor, # <--- Import VecMonitor
)
from torch.utils.tensorboard import SummaryWriter
import json
import sys
import argparse
import os
from datetime import datetime

# Import CustomEnv directly instead of relying on registration side-effect
from custom_env import CustomEnv


def train_sb3(
    # env_name is no longer needed here as we instantiate CustomEnv directly
    run_name="run_0",
    model_name="PPO",
    model_learning_rate=0.001,
    timesteps=100000,
    learning_sessions=1,
    n_envs=256, # <-- Add n_envs parameter
):
    # Where to store trained model and logs
    model_dir = os.path.join("models", run_name)
    log_dir = os.path.join("logs", run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nThe run name is:\n\n{run_name}\n")

    # 1. Instantiate CustomEnv directly (this is your base VecEnv)
    print(f"Creating CustomEnv with {n_envs} environments...")
    # Use a temporary name like base_vec_env or unwrapped_env
    base_vec_env = CustomEnv(n_envs=n_envs, render_mode=None) # Ensure render_mode is suitable for training

    # Check environment properties (can use base_vec_env)
    print("Observation Space:", base_vec_env.observation_space)
    print("Action Space:", base_vec_env.action_space)

    # 2. Access the underlying sim object to get configuration parameters BEFORE wrapping
    # This assumes CustomEnv stores the GenesisWorldEnv instance as self.sim
    print("Reading simulation configuration...")
    try:
        sim_instance = base_vec_env.sim # Access the sim object within CustomEnv
        reward_config = {
            "max_steps": sim_instance.max_steps,
            "min_dist_task_completion": sim_instance.min_dist_task_completion,
            "energy_penalty_weight": sim_instance.energy_penalty_weight,
            "time_penalty": sim_instance.time_penalty,
            "task_completion_bonus": sim_instance.task_completion_bonus,
            "end_ep_penalty": sim_instance.end_ep_penalty,
            "collision_penalty": sim_instance.collision_penalty,
            "max_collisions": sim_instance.max_collisions,
        }
    except AttributeError as e:
        print(f"Error accessing sim configuration from base_vec_env.sim: {e}")
        print("Ensure CustomEnv initializes self.sim correctly.")
        reward_config = {"error": "Could not read sim config"}


    # 3. Wrap the base VecEnv with VecMonitor
    print("Wrapping environment with VecMonitor...")
    # filename=None prevents VecMonitor from saving its own stats CSV file,
    # SB3 logger will handle TensorBoard logging.
    env = VecMonitor(base_vec_env, filename=None)

    # 4. Instantiate the SB3 model, passing the *monitored* env
    print(f"Creating {model_name} agent...")
    if model_name == "PPO":
        model = PPO(
            "MlpPolicy",
            env, # <-- Pass the VecMonitor-wrapped env
            verbose=1, # Set to 1 to see SB3 training outputs including ep_len_mean etc.
            device="cuda" if torch.cuda.is_available() else "cpu", # Better device check
            tensorboard_log=log_dir,
            learning_rate=model_learning_rate,
            ent_coef=0.01,
            n_steps=2048, # Example PPO hyperparam, adjust as needed
            batch_size=64 * n_envs, # Example PPO hyperparam
            n_epochs=10, # Example PPO hyperparam
        )
    elif model_name == "TD3":
        # Note: TD3 is off-policy and might require different env setup/hyperparams
        # Ensure CustomEnv works well with off-policy (e.g., proper done signals)
        model = TD3(
            "MlpPolicy",
            env, # <-- Pass the VecMonitor-wrapped env
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log=log_dir,
            learning_rate=model_learning_rate,
            # TD3 specific params might be needed: buffer_size, learning_starts, etc.
            buffer_size=int(1e6),
            learning_starts=10000, # Let buffer fill before learning
            batch_size=256, # Typical TD3 batch size
        )
    else:
        print(f"Error. Model {model_name} not implemented.\nExiting.")
        env.close() # Close env before exiting
        sys.exit()

    # --- Log run configuration to TensorBoard ---
    writer = SummaryWriter(log_dir=log_dir)
    run_specs = {
        "environment_class": "CustomEnv", # Use class name
        "num_envs": n_envs,
        "run_name": run_name,
        "algorithm": model_name,
        "hyperparameters": {
            "learning_rate": model.learning_rate,
            "gamma": model.gamma if hasattr(model, 'gamma') else 'N/A',
            # Add other relevant model hyperparameters if needed (access via model.<param>)
            "n_steps": model.n_steps if hasattr(model, 'n_steps') else 'N/A',
            "batch_size": model.batch_size if hasattr(model, 'batch_size') else 'N/A',
            "n_epochs": model.n_epochs if hasattr(model, 'n_epochs') else 'N/A',
            "ent_coef": model.ent_coef if hasattr(model, 'ent_coef') else 'N/A',
        },
        "reward_config": reward_config, # Use the config read earlier
        "notes": "Reach target task with updated reward structure.",
    }

    config_json = json.dumps(run_specs, indent=2)
    tensorboard_text = f"```json\n{config_json}\n```"
    writer.add_text("Run_specs", tensorboard_text, 0) # Add at step 0
    writer.close()
    # --- End TensorBoard config logging ---

    # --- Training Loop ---
    print(f"Starting training for {learning_sessions} sessions of {timesteps} timesteps each...")
    total_timesteps_elapsed = 0
    for i in range(learning_sessions):
        session_num = i + 1
        print(f"\n--- Learning Session {session_num}/{learning_sessions} ---")
        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=False, # Continue timestep count across sessions
            log_interval=1 # Log Tensorboard scalars every PPO/TD3 update (ep stats appear when episodes end)
        )
        total_timesteps_elapsed += timesteps # model.num_timesteps should reflect this
        save_path = os.path.join(model_dir, f"{run_name}_{total_timesteps_elapsed}")
        print(f"Saving model checkpoint to {save_path}")
        model.save(save_path)

    # --- Cleanup ---
    print("\nTraining finished. Closing environment.")
    env.close()  # Close the monitored env (which should close the base_vec_env)
    print(f"\nThe run name was:\n\n{run_name}\n")
    print(f"Models saved in: {model_dir}")
    print(f"Logs saved in: {log_dir}")
    print(f"\nTo view logs, run:\ntensorboard --logdir {log_dir}")


# Removed the get_base_env function as it's no longer needed for config reading

# Import torch for device check if not already imported
import torch

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Train RL agent in Genesis environment.")

    # Add arguments
    parser.add_argument("--model", type=str, default="PPO", choices=["PPO", "TD3"], help="RL Algorithm (PPO or TD3)")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate (default: 0.0003 for PPO)") # Adjusted default LR often better for PPO
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Timesteps per learning session")
    parser.add_argument("--sessions", type=int, default=10, help="Number of learning sessions")
    parser.add_argument("--envs", type=int, default=16, help="Number of parallel environments") # Reduced default envs, 256 might be too high depending on CPU
    parser.add_argument("--name", type=str, default=None, help="Optional base name for the run")

    args = parser.parse_args()

    now = datetime.now()
    formatted_time = now.strftime("%m%d%H%M")

    # Construct run name
    base_name = args.name if args.name else f"{args.model}" #_{args.envs}env"
    run_name = f"{formatted_time}_{base_name}_lr{args.lr}"

    train_sb3(
        run_name=run_name,
        model_name=args.model,
        model_learning_rate=args.lr,
        timesteps=args.timesteps,
        learning_sessions=args.sessions,
        n_envs=args.envs, # Pass n_envs
    )