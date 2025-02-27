import gymnasium as gym
from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
import json

# from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

import custom_env  # Even though we don't use this class here, we should include it here so that it registers the environment.


def train_sb3(
    env_name="CustomEnv-v0",
    run_name="run_0",
    model_learning_rate=0.001,
    timesteps=50000,
    learning_sessions=1,
    model_learning_starts=1000,
):

    # Where to store trained model and logs
    model_dir = os.path.join("models", run_name)
    log_dir = os.path.join("logs", run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    vec_env = DummyVecEnv([lambda: Monitor(gym.make(env_name))])
    # env = gym.make(env_name)
    # env = Monitor(env)

    # num_envs = 1  # Number of parallel environments
    # env = DummyVecEnv([lambda: Monitor(gym.make(env_name)) for _ in range(num_envs)])  # Parallelized envs

    # Check environment properties
    print("Observation Space:", vec_env.observation_space)
    print("Action Space:", vec_env.action_space)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        device="cuda",
        tensorboard_log=log_dir,
        # learning_starts=model_learning_starts,
        learning_rate=model_learning_rate,
    )

    writer = SummaryWriter(log_dir=log_dir)
    base_env = get_base_env(vec_env)
    # Your fixed configuration dictionary
    run_specs = {
        "env": env_name,
        "run_name": run_name,
        "algorithm": "PPO",
        "hyperparameters": {
            "learning_rate": model.learning_rate,
            "gamma": model.gamma,
            "ent_coef": model.ent_coef,
        },
        "reward_config": {
            "max_steps": base_env.sim.max_steps,
            "min_dist_task_completion": base_env.sim.min_dist_task_completion,
            "distance_weight": base_env.sim.distance_weight,
            "task_completion_reward": base_env.sim.task_completion_reward,
            "time_reward": base_env.sim.end_ep_reward,
            "collision_reward": base_env.sim.collision_reward,
            "max_collisions": base_env.sim.max_collisions,
        },
        "notes": "Target near floor with collision avoidance",
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

    vec_env.close()


def tune(env_name, run_name, parameter_list, learning_sessions=1):
    for _param_ in parameter_list:
        run_name_param = run_name + str(_param_)
        train_sb3(
            env_name=env_name,
            run_name=run_name_param,
            model_learning_starts=_param_,
            learning_sessions=learning_sessions,
        )


def get_base_env(vec_env, env_idx=0):
    current_env = vec_env.envs[env_idx]
    while hasattr(current_env, "env"):  # Unwrap all layers
        current_env = current_env.env
    return current_env


if __name__ == "__main__":
    now = datetime.now()
    formatted_time = now.strftime("%m%d%H%M")

    env_name = "CustomEnv-v0"
    run_name = formatted_time + "_PPO"
    learning_sessions = 6

    tuning = False

    if tuning:
        parameter_list = [50, 100]
        run_name = run_name + "_learning_starts_"

        tune(
            env_name=env_name,
            run_name=run_name,
            parameter_list=parameter_list,
            learning_sessions=learning_sessions,
        )
    else:
        train_sb3(
            env_name=env_name,
            run_name=run_name,
            learning_sessions=learning_sessions,
        )
