# CartPole (sim + real)

## Layout

| Path | Purpose |
|------|---------|
| `sim/fullvirenv.py` | Gymnasium CartPole used for training (inverted / customized dynamics). |
| `sim/virtualenv.py` | `RealisticCartPoleEnv` — classic Gymnasium CartPole with tuned physical parameters. |
| `sim/train.py` | Train PPO in simulation; writes models and TensorBoard logs under the repo root. |
| `sim/test.py` | Load a saved policy and roll out in sim with rendering. |
| `real/realenv.py` | Serial `CartPoleEnv` for the physical rig (Arduino). |
| `real/main.py` | Run a saved PPO on the real environment. |

## Run (from repository root)

```bash
python -m sim.train
python -m sim.test
python -m real.main
```

Models (`*.zip`) and `cartpole_tensorboard/` are expected next to this README unless you change paths in the scripts.

## Dependencies

```bash
pip install -r requirements.txt
```
