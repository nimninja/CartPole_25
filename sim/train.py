from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from .fullvirenv import CartPoleEnv
except ImportError:
    from fullvirenv import CartPoleEnv

_ROOT = Path(__file__).resolve().parent.parent
_TB_LOG = _ROOT / "cartpole_tensorboard"


def _make_env():
    # mixed: half swing-up from hang, half balance from small upright perturbations
    return CartPoleEnv(render_mode=None, reset_mode="mixed")


def main() -> None:
    env = DummyVecEnv([_make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(_TB_LOG),
        # Slightly more decisive updates vs defaults (lr 3e-4, ent_coef 0)
        learning_rate=5e-4,
        ent_coef=0.01,
        n_steps=1024,
        batch_size=64,
        clip_range=0.2,
    )

    model.learn(total_timesteps=1_000_000)
    model_path = _ROOT / "asdasdasd4"
    model.save(str(model_path))

    # Short rollout on the vec env (DummyVecEnv returns obs, rewards, dones, infos)
    model = PPO.load(str(model_path), env=env)
    obs = env.reset()
    for _ in range(2000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        if dones[0]:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
