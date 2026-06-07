"""Shared timing, sim physics, and checkpoint names for sim + real."""

STEP_DT = 0.06
MAX_EPISODE_STEPS = 600

# --- Sim physics (light cart + moderate force ≈ real motor, not Gymnasium 10 N on 1.4 kg) ---
SIM_MASSCART = 0.15
SIM_MASSPOLE = 0.06
SIM_POLE_LENGTH = 0.35  # half pole length (m)
SIM_FORCE_MAG = 2.5  # N; raise to 3.0 if swing-up too weak in sim test
SIM_X_THRESHOLD = 2.4  # matches real obs scale (belt / BELT_PER_SIM_X)

# Real motor pulse (s); 0 = motor on until next command
MOTOR_PULSE_S = 0.0

# --- Checkpoints ---
CHECKPOINT_SIM2REAL = "cartpole_sim2real"
LEGACY_CHECKPOINT = "asdasdasd5"

SIM_TRAIN_TIMESTEPS = 400_000
