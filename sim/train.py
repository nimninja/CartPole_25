from pathlib import Path

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from .fullvirenv import CartPoleEnv
except ImportError:
    from fullvirenv import CartPoleEnv

_ROOT = Path(__file__).resolve().parent.parent
_TB_LOG = _ROOT / "cartpole_tensorboard"
_SAVE_STEM = "cartpole_ppo_speed"


def _make_env():
    return TimeLimit(CartPoleEnv(render_mode=None), max_episode_steps=500)


def main() -> None:
    env = DummyVecEnv([_make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(_TB_LOG),
        learning_rate=3e-4,
        ent_coef=0.01,
        n_steps=2048,
        batch_size=64,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    )

    model.learn(total_timesteps=300_000)
    out = _ROOT / _SAVE_STEM
    model.save(str(out))
    print(f"Saved {_SAVE_STEM}.zip")

    model = PPO.load(str(out), env=env)
    obs = env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)
        if dones[0]:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
