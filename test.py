import gymnasium as gym
from stable_baselines3 import A2C, PPO, TD3

import os
import argparse

import custom_env  # Even though we don't use this class here, we should include it here so that it registers the WarehouseRobot environment.


# Test using StableBaseline3. Lots of hardcoding for simplicity.
def test_sb3(
    env_name="CustomEnv-v0",
    model_to_load="run_0",
    episode_to_load=50000,
    nb_tests=1,
    render=True,
):
    # Where to load the trained model and logs
    model_dir = os.path.join("models", model_to_load)

    env = gym.make(env_name, render_mode="human" if render else None)

    # Load model
    model = PPO.load(
        os.path.join(model_dir, f"{model_to_load}_{episode_to_load}"),
        env=env,
    )

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
    # Create the parser
    parser = argparse.ArgumentParser(description="Process a string input.")

    # Add an argument for the string
    parser.add_argument("--run_id", type=str, required=True, help="run_id MMDDHHMM")
    parser.add_argument("--episode", type=int, default=300000, required=False, help="episode to test x*50000")

    # Parse the arguments
    args = parser.parse_args()

    env_name = "CustomEnv-v0"
    run_name = args.run_id + "_PPO"
    episode = args.episode

    test_sb3(
        env_name=env_name,
        model_to_load=run_name,
        episode_to_load=episode,
    )
