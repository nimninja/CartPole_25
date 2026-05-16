"""Shared cart force action: one float in [-1, 1] (sign = direction, |a| = speed)."""

from __future__ import annotations

import numpy as np

FORCE_LOW = np.float32(-1.0)
FORCE_HIGH = np.float32(1.0)
DEADZONE = 0.05


def clip_force(action) -> float:
    """Map policy output to [-1, 1] with a small deadzone → 0 (coast)."""
    f = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
    f = float(np.clip(f, -1.0, 1.0))
    if abs(f) < DEADZONE:
        return 0.0
    return f


def force_to_serial_line(force: float) -> bytes:
    return f"{force:.4f}\n".encode("ascii")
