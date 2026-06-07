"""Train PPO for sim2real (inverted CartPole, 60 ms steps, light cart, low force)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cartpole_constants import (
    CHECKPOINT_SIM2REAL,
    LEGACY_CHECKPOINT,
    MAX_EPISODE_STEPS,
    SIM_FORCE_MAG,
    SIM_MASSCART,
    SIM_TRAIN_TIMESTEPS,
    STEP_DT,
)

try:
    from .fullvirenv import CartPoleEnv
except ImportError:
    from fullvirenv import CartPoleEnv

_TB_LOG = _ROOT / "cartpole_tensorboard"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train sim PPO (sim2real profile)")
    p.add_argument(
        "--save",
        default=CHECKPOINT_SIM2REAL,
        help=f"Output checkpoint stem (default: {CHECKPOINT_SIM2REAL})",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=SIM_TRAIN_TIMESTEPS,
        help="Training timesteps (from-scratch run)",
    )
    p.add_argument(
        "--continue-from",
        default=None,
        metavar="STEM",
        help="Load this checkpoint and train more (e.g. cartpole_sim2real)",
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        help=f"Save to {LEGACY_CHECKPOINT} instead (old naming)",
    )
    return p.parse_args()


def _make_env():
    return TimeLimit(
        CartPoleEnv(render_mode=None, sim2real_reward=True),
        max_episode_steps=MAX_EPISODE_STEPS,
    )


def _resolve_ckpt(stem: str) -> Path | None:
    for candidate in (_ROOT / f"{stem}.zip", _ROOT / stem):
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    args = _parse_args()
    save_stem = LEGACY_CHECKPOINT if args.legacy else args.save

    env = DummyVecEnv([_make_env])
    ckpt = _resolve_ckpt(args.continue_from) if args.continue_from else None

    if ckpt is not None:
        print(f"Continuing from: {ckpt}")
        model = PPO.load(
            str(ckpt),
            env=env,
            tensorboard_log=str(_TB_LOG),
            custom_objects={"observation_space": env.observation_space},
        )
        model.learn(
            total_timesteps=int(args.timesteps),
            reset_num_timesteps=False,
            progress_bar=True,
        )
    else:
        print(
            f"Training from scratch → {save_stem}.zip  "
            f"(masscart={SIM_MASSCART} kg, force={SIM_FORCE_MAG} N, step={STEP_DT} s, sim2real reward)"
        )
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(_TB_LOG),
            learning_rate=3e-4,
            ent_coef=0.02,
            n_steps=2048,
            batch_size=64,
            clip_range=0.2,
            gamma=0.99,
        )
        model.learn(total_timesteps=int(args.timesteps), progress_bar=True)

    out = _ROOT / save_stem
    model.save(str(out))
    print(f"Saved: {out}.zip")
    env.close()


if __name__ == "__main__":
    main()
