from pathlib import Path

from stable_baselines3 import PPO

try:
    from .realenv import CartPoleEnv
except ImportError:
    from realenv import CartPoleEnv

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    env = CartPoleEnv()
    model = PPO.load(str(_ROOT / "asdasdasd4.zip"))

    obs, info = env.reset()

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        if terminated or truncated:
            print("Resetting environment...")
            obs, info = env.reset()


if __name__ == "__main__":
    main()
