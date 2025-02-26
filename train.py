import gymnasium as gym
from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# from torch.utils.tensorboard import SummaryWriter
import os
import custom_env  # Even though we don't use this class here, we should include it here so that it registers the environment.


def train_sb3(
    env_name="CustomEnv-v0",
    run_name="run_0",
    model_learning_rate=0.001,
    timesteps=50000,
    learning_sessions=1,
    model_learning_starts=100,
):

    # Where to store trained model and logs
    model_dir = os.path.join("models", run_name)
    log_dir = os.path.join("logs", run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    env = DummyVecEnv([lambda: Monitor(gym.make(env_name))])

    # Check environment properties
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    model = TD3(
        "MlpPolicy",
        env,
        verbose=0,
        device="cuda",
        tensorboard_log=log_dir,
        learning_starts=model_learning_starts,
        learning_rate=model_learning_rate,
    )

    i = 0
    while learning_sessions > i:
        i += 1

        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)  # train
        model.save(os.path.join(model_dir, f"{run_name}_{timesteps*i}"))

    env.close()


def tune(env_name, run_name, parameter_list, learning_sessions=1):
    for _param_ in parameter_list:
        run_name_param = run_name + str(_param_)
        train_sb3(
            env_name=env_name,
            run_name=run_name_param,
            model_learning_starts=_param_,
            learning_sessions=learning_sessions,
        )


if __name__ == "__main__":
    env_name = "CustomEnv-v0"
    run_name = "TD3_run0"
    learning_sessions = 10

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
