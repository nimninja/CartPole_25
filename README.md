# CartPole (sim + real)

## Layout

| Path | Purpose |
|------|---------|
| `sim/fullvirenv.py` | Gymnasium CartPole used for training (inverted / customized dynamics). |
| `sim/virtualenv.py` | `RealisticCartPoleEnv` — classic Gymnasium CartPole with tuned physical parameters. |
| `sim/train.py` | Train PPO in simulation; writes models and TensorBoard logs under the repo root. |
| `sim/test.py` | Load a saved policy and roll out in sim with rendering. |
| `real/realenv.py` | Serial `CartPoleEnv` for the physical rig (Arduino). |
| `real/main.py` | Run hybrid control: PPO swing-up + PID balance on hardware. |
| `real/balance.py` | Raw-encoder balance PID and phase handoff logic. |

## Run (from repository root)

```bash
python -m sim.train
python -m sim.test
python -m real.main
```

### Real hardware tuning (no obs mapping)

`real/main.py` uses **PPO for swing-up** and switches to a **PID balance controller** when the pole angle is within `CARTPOLE_BALANCE_BAND` counts of upright.

| Env var | Default | Purpose |
|---------|---------|---------|
| `CARTPOLE_SERIAL_PORT` | `COM3` | Arduino serial port |
| `CARTPOLE_UPRIGHT_ANGLE` | `-85` | Encoder counts at balance (tune on your rig) |
| `CARTPOLE_BALANCE_BAND` | `120` | Handoff zone ± counts around upright |
| `CARTPOLE_PULSE_MS` | `25` | Motor pulse length (ms), then STOP |
| `CARTPOLE_PID_KP_ANGLE` | `0.06` | Angle P gain |
| `CARTPOLE_PID_KD_ANGLE` | `0.0015` | Angle D gain |
| `CARTPOLE_PID_KP_BELT` | `0.00004` | Cart center P gain |
| `CARTPOLE_PID_KD_BELT` | `0.000008` | Cart center D gain |
| `CARTPOLE_PID_DEADBAND` | `0.08` | Below this → STOP (no pulse) |
| `CARTPOLE_VERBOSE` | `0` | `1` = print every serial line |
| `CARTPOLE_LOG_EVERY` | `10` | Status print interval in `main.py` |

Re-flash `real/cp_main.ino` after pulling (loop delay 20 ms for ~50 Hz).

`sim/train.py` now uses phase-aware rewards and mixed reset (swing-up + balance curriculum) for 1M steps.

Models (`*.zip`) and `cartpole_tensorboard/` are expected next to this README unless you change paths in the scripts.

## Dependencies

```bash
pip install -r requirements.txt
```
