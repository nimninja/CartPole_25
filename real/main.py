from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

try:
    from .realenv import CartPoleEnv
except ImportError:
    from realenv import CartPoleEnv

_ROOT = Path(__file__).resolve().parent.parent
_MODEL = _ROOT / "cartpole_ppo_speed.zip"


def main() -> None:
    env = CartPoleEnv()
    model = PPO.load(str(_MODEL), env=env)

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        force = float(np.asarray(action).reshape(-1)[0])
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"force={force:+.3f} term={terminated} trunc={truncated}")
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
