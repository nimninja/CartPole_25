from pathlib import Path

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from .fullvirenv import CartPoleEnv
except ImportError:
    from fullvirenv import CartPoleEnv

_ROOT = Path(__file__).resolve().parent.parent
_MODEL_STEM = "cartpole_ppo_speed"


def _model_path() -> Path:
    for candidate in (_ROOT / f"{_MODEL_STEM}.zip", _ROOT / _MODEL_STEM):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing {_MODEL_STEM}.zip — train with: python -m sim.train"
    )


def _make_env():
    return TimeLimit(CartPoleEnv(render_mode="human"), max_episode_steps=500)


def main() -> None:
    env = DummyVecEnv([_make_env])
    model = PPO.load(str(_model_path()), env=env)

    obs = env.reset()
    step = 0
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            force = float(np.asarray(action).reshape(-1)[0])
            if step % 20 == 0:
                print(f"step={step} force={force:.3f} obs={np.asarray(obs[0])}")

            obs, _, dones, _ = env.step(action)
            env.render()
            step += 1
            if dones[0]:
                obs = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
