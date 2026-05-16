from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from .fullvirenv import CartPoleEnv
except ImportError:
    from fullvirenv import CartPoleEnv

_ROOT = Path(__file__).resolve().parent.parent
_MODEL_STEM = "asdasdasd4.zip"


def _model_path() -> Path:
    for candidate in (_ROOT / f"{_MODEL_STEM}.zip", _ROOT / _MODEL_STEM):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing policy file: expected {_ROOT / (_MODEL_STEM + '.zip')} "
        f"(train with sim/train.py or copy your checkpoint to the repo root)."
    )


def _make_env():
    return CartPoleEnv(render_mode="human")


def main() -> None:
    env = DummyVecEnv([_make_env])
    model = PPO.load(str(_model_path()), env=env)

    obs = env.reset()
    step = 0
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            a = int(np.asarray(action).reshape(-1)[0])
            if step % 15 == 0:
                print(f"step={step} action={a} obs={np.asarray(obs[0])}")

            obs, rewards, dones, infos = env.step(action)
            env.render()
            step += 1

            if dones[0]:
                obs = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
