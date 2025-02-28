import gymnasium as gym
from stable_baselines3 import A2C, PPO, TD3

import os
import argparse
import sys

import custom_env  # Even though we don't use this class here, we should include it here so that it registers the WarehouseRobot environment.


# Test using StableBaseline3. Lots of hardcoding for simplicity.
def test_sb3(
    env_name="CustomEnv-v0",
    model_to_load="run_0",
    model_name="TD3",
    episode_to_load=50000,
    nb_tests=1,
    render=True,
    seed=None
):
    # Where to load the trained model and logs
    model_dir = os.path.join("models", model_to_load)

    env = gym.make(env_name, render_mode="human" if render else None)

    # Load model
    if model_name == "PPO":
        model = PPO.load(
            os.path.join(model_dir, f"{model_to_load}_{episode_to_load}"),
            env=env,
        )
    elif model_name == "TD3":
        model = TD3.load(
            os.path.join(model_dir, f"{model_to_load}_{episode_to_load}"),
            env=env,
        )
    else:
        print(f"Unknown model_name {model_name}.\nExiting.")
        sys.exit()

    for _ in range(nb_tests):
        # Run a test
        obs = env.reset(seed=seed)[0]
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
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add an argument for the string
    parser.add_argument("--run_id", type=str, required=True, help="run_id MMDDHHMM")
    parser.add_argument("--model", type=str, default="TD3", help="TD3 or PPO")
    parser.add_argument(
        "--episode",
        type=int,
        default=300000,
        required=False,
        help="episode to test, every i*100000",
    )
    parser.add_argument("--seed", type=int, default=None, required=False)

    # Parse the arguments
    args = parser.parse_args()

    env_name = "CustomEnv-v0"
    model_name = args.model
    run_name = args.run_id + "_" + model_name
    episode = args.episode
    seed = args.seed

    test_sb3(
        env_name=env_name,
        model_to_load=run_name,
        model_name = model_name,
        episode_to_load=episode,
        seed=seed,
    )
