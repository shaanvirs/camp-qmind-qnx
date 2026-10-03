"""Daniel's synthetic motor sensor for the Camp QMIND MVP.

Contract: stream(fault="bearing", fault_at=5.0, severity=1.0) yields
{t, signal, true_state, fault_start_t}. 5,000 Hz, 1,024 float64 samples.
Amplitudes are arbitrary vibration units, not calibrated acceleration.

Healthy: 30 Hz sine + independent Gaussian noise. Imbalance: added 30 Hz
amplitude. Bearing: 120 impacts/s, each followed by 1,000 Hz damped ringing.
This is a controllable demonstration model, not a real-machine diagnosis model.

Time is simulated; this generator does not sleep. The consumer owns pacing.
A window's samples are available at t + WINDOW_SECONDS. A mixed window is
labelled faulty if ANY sample is at/after onset and severity is positive.
The two ground-truth fields must never be inputs to detection/shutoff decisions.
"""

from __future__ import annotations

from collections.abc import Iterator
import math
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 5_000
WINDOW_SIZE = 1_024
WINDOW_SECONDS = WINDOW_SIZE / SAMPLE_RATE
MOTOR_HZ = 30.0
HEALTHY_AMPLITUDE = 1.0
NOISE_STD = 0.05
IMBALANCE_AMPLITUDE = 1.5
BEARING_AMPLITUDE = 3.0
BEARING_IMPACT_HZ = 120.0
BEARING_RING_HZ = 1_000.0
BEARING_DECAY_SECONDS = 0.0015
# Eight impacts cover >44 decay time constants; older tails are negligible.
_RING_TAILS = 8

State = Literal["normal", "imbalance", "bearing"]


class Window(TypedDict):
    t: float
    signal: NDArray[np.float64]
    true_state: State
    fault_start_t: float | None


def _bearing_response(elapsed: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate ringing in absolute fault time, preserving tails across windows."""
    response = np.zeros_like(elapsed)
    latest_impact = np.floor(elapsed * BEARING_IMPACT_HZ)
    for lag in range(_RING_TAILS):
        impact = latest_impact - lag
        valid = (elapsed >= 0) & (impact >= 0)
        age = elapsed[valid] - impact[valid] / BEARING_IMPACT_HZ
        response[valid] += np.exp(-age / BEARING_DECAY_SECONDS) * np.sin(
            2 * np.pi * BEARING_RING_HZ * age
        )
    return BEARING_AMPLITUDE * response


def stream(
    fault: State = "bearing",
    fault_at: float = 5.0,
    severity: float = 1.0,
    *,
    seed: int | None = None,
    ramp_seconds: float = 0.0,
) -> Iterator[Window]:
    """Yield independent, consecutive vibration windows indefinitely.

    The three positional arguments match the team interface. Optional extras:
    seed fixes the noise sequence; None gives a fresh sequence for each run.
    ramp_seconds=0 switches on immediately; >0 grows linearly to full severity.
    fault="normal", severity=0, or fault_at=math.inf yields healthy data only.
    Onset may fall inside a window or between sample instants. fault_start_t is
    the requested continuous-time onset, not the window boundary/detection time.
    Validation occurs when the generator is first advanced.
    """
    if fault not in ("normal", "imbalance", "bearing"):
        raise ValueError("fault must be 'normal', 'imbalance', or 'bearing'")
    fault_at, severity, ramp_seconds = map(float, (fault_at, severity, ramp_seconds))
    if math.isnan(fault_at) or fault_at < 0:
        raise ValueError("fault_at must be nonnegative; +infinity means no fault")
    if not math.isfinite(severity) or severity < 0:
        raise ValueError("severity must be finite and nonnegative")
    if not math.isfinite(ramp_seconds) or ramp_seconds < 0:
        raise ValueError("ramp_seconds must be finite and nonnegative")
    # Avoid accepting finite inputs that overflow when fault amplitudes are applied.
    if severity > np.finfo(np.float64).max / (2 * BEARING_AMPLITUDE):
        raise ValueError("severity is too large to represent the signal")
    rng = np.random.default_rng(seed)
    sample_offset = np.arange(WINDOW_SIZE, dtype=np.float64)
    start_sample = 0
    enabled = fault != "normal" and severity > 0 and math.isfinite(fault_at)

    while True:
        times = (start_sample + sample_offset) / SAMPLE_RATE
        rotation = np.sin(2 * np.pi * MOTOR_HZ * times)
        signal = HEALTHY_AMPLITUDE * rotation + rng.normal(0, NOISE_STD, WINDOW_SIZE)
        active = (times >= fault_at) if enabled else np.zeros(WINDOW_SIZE, dtype=bool)
        affected = bool(np.any(active))
        if affected:
            elapsed = times[active] - fault_at
            envelope = (
                np.minimum(elapsed / ramp_seconds, 1.0)
                if ramp_seconds > 0
                else np.ones_like(elapsed)
            )
            contribution = (
                IMBALANCE_AMPLITUDE * rotation[active]
                if fault == "imbalance"
                else _bearing_response(elapsed)
            )
            signal[active] += severity * envelope * contribution
        yield {
            "t": start_sample / SAMPLE_RATE,
            "signal": signal,
            "true_state": fault if affected else "normal",
            "fault_start_t": fault_at if affected else None,
        }
        start_sample += WINDOW_SIZE


if __name__ == "__main__":
    from itertools import islice

    print("Synthetic sensor: 5,000 Hz; 1,024 samples/window; unpaced simulation")
    for window in islice(stream(seed=42), 30):
        rms = np.sqrt(np.mean(window["signal"] ** 2))
        print(
            f"t={window['t']:7.4f}s  state={window['true_state']:9s}  "
            f"rms={rms:.3f}  fault_start_t={window['fault_start_t']}"
        )
