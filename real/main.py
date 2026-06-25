import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

try:
    from .balance import HybridController
    from .realenv import CartPoleEnv
except ImportError:
    from balance import HybridController
    from realenv import CartPoleEnv

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    env = CartPoleEnv()
    model = PPO.load(str(_ROOT / "asdasdasd4.zip"))
    controller = HybridController()

    obs, info = env.reset()
    controller.reset()
    step = 0

    print(
        f"LATCH  |angle - {controller.nominal_upright}| ≤ {controller.latch_deg:.0f}°  "
        f"(encoder ~{controller.band_low:.0f}–{controller.band_high:.0f})"
    )
    print(
        f"RELEASE  >{controller.release_deg:.0f}° off upright, fast fall, or angle ≤{controller.hang_down:.0f}"
    )
    print(
        f"RAIL   belt soft=±{controller.rail.soft}  hard=±{controller.rail.hard}  "
        f"(STOP before hitting motor)"
    )
    print("Hold pole up — watch [angle] lines; balance fires when dist° ≤ latch°\n")

    try:
        while True:
            out = controller.act(obs, model)
            balance = out.phase == "balance"
            angle = float(obs[0])
            dist = controller.degrees_from_upright(angle)

            belt = float(obs[1])
            ang_vel = float(obs[2])
            if step % 20 == 0:
                in_zone = controller._should_latch(angle, ang_vel)
                print(
                    f"  [angle] step={step}  raw={angle:.0f}  belt={belt:.0f}  "
                    f"dist={dist:.1f}°  latch={'YES' if in_zone else 'no'}  "
                    f"phase={'BALANCE' if balance else 'swing'}"
                )

            if controller.just_latched:
                motor = {0: "LEFT", 1: "RIGHT", 2: "STOP"}.get(out.action, "?")
                print(
                    f">>> BALANCE ON   step={step}  angle={angle:.0f}  "
                    f"θ={out.theta_deg:+.1f}°  zone={out.zone}  "
                    f"motor={motor}  pulse={out.pulse_ms}ms  u={out.pid_u:+.1f}"
                )
            if balance and step % 5 == 0:
                motor = {0: "LEFT", 1: "RIGHT", 2: "STOP"}.get(out.action, "?")
                rail = f"  RAIL={out.rail}" if out.rail else ""
                print(
                    f"  [bal]  belt={belt:.0f}  θ={out.theta_deg:+.1f}°  zone={out.zone}  "
                    f"u={out.pid_u:+.1f}  {motor} {out.pulse_ms}ms{rail}"
                )
            if out.rail:
                print(f"  *** RAIL STOP  belt={belt:.0f}  {out.rail}  (blocked into motor)")
            if controller.just_released:
                print(f">>> BALANCE OFF  step={step}  angle={angle:.0f}  → swing-up")

            env.set_loop_mode(balance)
            obs, reward, terminated, truncated, info = env.step(
                out.action,
                pulse_ms=out.pulse_ms,
                balance=balance,
            )

            step += 1
            if terminated or truncated:
                obs, info = env.reset()
                controller.reset()
                env.set_loop_mode(False)
                step = 0
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
