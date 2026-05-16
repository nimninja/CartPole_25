"""
Fine-tune a sim-trained PPO policy on the real CartPole (serial env).

Run from the repository root (CartPole_25), either:

    python -m real_training.train_irl --model YOUR_CHECKPOINT

or:

    python real_training/train_irl.py --model YOUR_CHECKPOINT

Requires:
  - Mapped observations in CartPoleEnv so obs shape/meaning matches the loaded policy.
  - CARTPOLE_SERIAL_PORT / CARTPOLE_SERIAL_BAUD set if defaults in real/realenv.py are wrong.

Safety: use a short --timesteps first, e-stop ready, and monitor the first rollouts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IRL PPO fine-tuning on real CartPoleEnv")
    p.add_argument(
        "--model",
        type=str,
        default="cartpole_ppo_speed",
        help="Sim checkpoint stem or path under repo root (adds .zip if missing)",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=2_000,
        help="Fine-tune steps (keep small on hardware; repeat runs as needed)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (lower than sim training)",
    )
    p.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
        help="Entropy coef (0 = deterministic-ish updates; tiny e.g. 0.005 if you want exploration)",
    )
    p.add_argument(
        "--max-episode-steps",
        type=int,
        default=400,
        help="TimeLimit on real episodes",
    )
    p.add_argument(
        "--save",
        type=str,
        default="cartpole_ppo_irl_finetune",
        help="Output stem for final save under repo root",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="irl_checkpoints",
        help="Subfolder under repo root for periodic checkpoints during learn",
    )
    p.add_argument(
        "--checkpoint-freq",
        type=int,
        default=500,
        help="Save checkpoint every N timesteps (0 = disable)",
    )
    p.add_argument(
        "--tensorboard",
        type=str,
        default="irl_tensorboard",
        help="Tensorboard log folder under repo root",
    )
    p.add_argument("--port", type=str, default=None, help="Serial port (overrides env default)")
    p.add_argument("--baud", type=int, default=None, help="Serial baud rate")
    return p.parse_args()


def _resolve_model_path(stem_or_path: str) -> Path:
    raw = Path(stem_or_path)
    if raw.is_file():
        return raw
    cand = _ROOT / stem_or_path
    if cand.is_file():
        return cand
    z = _ROOT / f"{stem_or_path}.zip"
    if z.is_file():
        return z
    raise FileNotFoundError(
        f"Could not find model checkpoint: tried {raw}, {cand}, {z}"
    )


def _make_real_env(max_episode_steps: int, port: str | None, baud: int | None):
    from real.realenv import CartPoleEnv

    kw: dict = {}
    if port is not None:
        kw["port"] = port
    if baud is not None:
        kw["baudrate"] = baud
    return TimeLimit(CartPoleEnv(**kw), max_episode_steps=max_episode_steps)


def main() -> None:
    args = _parse_args()
    model_path = _resolve_model_path(args.model)

    env = DummyVecEnv(
        [
            lambda: _make_real_env(
                args.max_episode_steps,
                port=args.port,
                baud=args.baud,
            )
        ]
    )

    print(f"Loading policy from: {model_path}")
    model = PPO.load(
        str(model_path),
        env=env,
        tensorboard_log=str(_ROOT / args.tensorboard),
    )

    # Override LR schedule for fine-tuning (constant LR)
    lr = float(args.lr)
    model.lr_schedule = lambda _progress_remaining: lr
    model.ent_coef = float(args.ent_coef)

    callbacks = []
    if args.checkpoint_freq > 0:
        ckpt_dir = _ROOT / args.checkpoint_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(ckpt_dir),
                name_prefix="irl_ppo",
            )
        )

    print(
        f"Starting IRL learn: timesteps={args.timesteps}, lr={lr}, "
        f"ent_coef={model.ent_coef}, n_steps={getattr(model, 'n_steps', '?')} (from checkpoint)"
    )
    try:
        model.learn(
            total_timesteps=int(args.timesteps),
            callback=callbacks if callbacks else None,
            reset_num_timesteps=False,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Interrupted — saving current policy...")
    finally:
        out = _ROOT / args.save
        model.save(str(out))
        print(f"Saved: {out}.zip")
        env.close()


if __name__ == "__main__":
    main()
