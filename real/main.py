import argparse
import sys
from pathlib import Path

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cartpole_constants import CHECKPOINT_SIM2REAL, MAX_EPISODE_STEPS
from real.realenv import CartPoleEnv


def _model_path(stem: str) -> Path:
    for candidate in (_ROOT / f"{stem}.zip", _ROOT / stem):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing: {_ROOT / (stem + '.zip')}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run a PPO policy on the real CartPole")
    p.add_argument(
        "--model",
        default=CHECKPOINT_SIM2REAL,
        help=f"Same default as sim.test (default: {CHECKPOINT_SIM2REAL}.zip)",
    )
    p.add_argument("--port", default=None, help="Serial port e.g. COM9")
    args = p.parse_args()

    print(f"Loading: {_model_path(args.model)}")

    kw = {"verbose": True}
    if args.port:
        kw["port"] = args.port
    env = TimeLimit(CartPoleEnv(**kw), max_episode_steps=MAX_EPISODE_STEPS)

    try:
        model = PPO.load(
            str(_model_path(args.model)),
            env=env,
            custom_objects={"observation_space": env.observation_space},
        )

        obs, info = env.reset()
        prev_raw = (env.unwrapped.prev_angle, env.unwrapped.prev_belt)
        stale_steps = 0

        while True:
            action, _states = model.predict(obs, deterministic=True)
            a = int(np.asarray(action).reshape(-1)[0])

            obs, reward, terminated, truncated, info = env.step(action)
            raw = (env.unwrapped.prev_angle, env.unwrapped.prev_belt)
            stale_steps = stale_steps + 1 if raw == prev_raw else 0
            prev_raw = raw

            o = np.asarray(obs, dtype=np.float32)
            print(
                f"action={a} ({'LEFT' if a == 0 else 'RIGHT'}) "
                f"raw_angle={raw[0]} raw_belt={raw[1]} "
                f"obs=[{o[0]:.3f},{o[1]:.3f},{o[2]:.3f},{o[3]:.3f}] "
                f"reward={reward:.3f} term={terminated} trunc={truncated}"
            )
            if stale_steps >= 8:
                print(
                    "WARNING: encoder counts unchanged for 8+ steps — "
                    "policy will stick to one action. Check motor moves cart in Serial Monitor."
                )
                stale_steps = 0

            if terminated or truncated:
                print("Resetting environment...")
                obs, info = env.reset()
                prev_raw = (env.unwrapped.prev_angle, env.unwrapped.prev_belt)
                stale_steps = 0
    finally:
        env.close()


if __name__ == "__main__":
    main()
