import gymnasium as gym
from stable_baselines3 import A2C, PPO, TD3
import os
import custom_env  # Even though we don't use this class here, we should include it here so that it registers the WarehouseRobot environment.

# Test using StableBaseline3. Lots of hardcoding for simplicity.
def test_sb3(
    env_name="CustomEnv-v0",
    run_name="run_0",
    episode_to_load=1000,
    nb_tests=1,
    render=True,
):
    # Where to load the trained model and logs
    model_dir = os.path.join("models", run_name)
    log_dir = os.path.join("logs", run_name)

    env = gym.make(env_name, render_mode="human" if render else None)

    # Load model
    model = TD3.load(
        os.path.join(model_dir, f"{run_name}_{episode_to_load}"),
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
    env_name = "CustomEnv-v0"

    test_sb3(env_name=env_name, run_name="TD3_run0", episode_to_load=1200000)
