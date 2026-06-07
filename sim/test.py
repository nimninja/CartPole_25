import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cartpole_constants import CHECKPOINT_SIM2REAL
from sim.fullvirenv import CartPoleEnv


def _model_path(stem: str) -> Path:
    for candidate in (_ROOT / f"{stem}.zip", _ROOT / stem):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing: {_ROOT / (stem + '.zip')}")


def _make_env():
    return CartPoleEnv(render_mode="human", sim2real_reward=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=CHECKPOINT_SIM2REAL)
    args = p.parse_args()

    print(f"Loading: {_model_path(args.model)}")
    env = DummyVecEnv([_make_env])
    model = PPO.load(
        str(_model_path(args.model)),
        env=env,
        custom_objects={"observation_space": env.observation_space},
    )

    obs = env.reset()
    step = 0
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            a = int(np.asarray(action).reshape(-1)[0])
            if step % 15 == 0:
                o = np.asarray(obs[0])
                print(f"step={step} action={a} x={o[0]:.3f} x_dot={o[1]:.3f} theta={o[2]:.3f}")

            obs, rewards, dones, infos = env.step(action)
            env.render()
            step += 1
            if dones[0]:
                obs = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
