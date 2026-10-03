# Sensor stream — Daniel

`stream.py` generates motor vibration windows for the laptop MVP. It needs only NumPy and keeps the agreed three-argument interface:

```python
from stream import stream, WINDOW_SECONDS

for window in stream(fault="bearing", fault_at=5.0, severity=1.0):
    features = extract(window["signal"])
    prediction = predict(features)
    # Integration owns the shutoff decision and live playback pacing.
```

Each yielded dictionary has `t` (window start in simulated seconds), `signal` (1,024 float64 samples at 5,000 Hz), `true_state`, and `fault_start_t`. A window lasts 0.2048 seconds. The generator is infinite and unpaced; stop with `break` or use `itertools.islice` for a finite dataset.

## Signals

| State | Signal |
| --- | --- |
| `normal` | 30 Hz sine, amplitude 1, Gaussian noise standard deviation 0.05 |
| `imbalance` | Baseline plus extra 30 Hz amplitude, scaled by severity |
| `bearing` | Baseline plus 120 impacts/second exciting damped 1,000 Hz ringing |

Amplitudes are arbitrary units. These are illustrative synthetic signals, not calibrated measurements or evidence of real-machine diagnostic performance.

The healthy label is **`normal`**, following the Notion Build Guide; the original placeholder comment said `healthy`. Consumers should use `normal`. Ground-truth labels and onset timestamps are for evaluation only and must never be model inputs or shutoff criteria.

## Optional controls

```python
stream("bearing", 5.0, 0.5, seed=42, ramp_seconds=2.0)
```

`seed` fixes the noise sequence for debugging; its default is fresh randomness. Use different seeds/runs for training and evaluation. `ramp_seconds` defaults to zero (step onset); a positive value grows the added fault linearly from zero to the chosen severity. `severity=0`, `fault="normal"`, or `fault_at=float("inf")` gives healthy data.

Only samples at/after onset are affected. A mixed window is labelled faulty once it contains any post-onset sample and severity is positive. For an onset at 5 seconds, the first affected window starts at 4.9152 and completes at 5.12; its `fault_start_t` stays 5.0. Base latency on window **completion**, then add processing/confirmation delay. The start of a ramp remains the ground-truth onset.

The stream's onset is preconfigured. Changing a running stream in response to a live dashboard button needs an agreed control interface; creating a new generator restarts its time at zero.

## Run and check

```bash
python stream.py
python -m unittest discover -s tests -v
```

The 15 tests cover the dictionary contract, sample timing, transition windows, both faults, severity scaling, ramps, repeatability, and signal continuity. Full team-pipeline detection/shutoff still needs integration with the other modules.

Optional plots and example export:

```bash
python -m pip install -r requirements-sensors.txt
python preview_sensor.py
```

This creates waveform/FFT comparisons, a fault-ramp plot, generation metadata, and `example-windows.npz` under `artifacts/`. Those three matched-noise examples are for illustration and loading checks, not a train/test dataset.

```python
import numpy as np
with np.load("artifacts/example-windows.npz", allow_pickle=False) as data:
    print(data["labels"], data["signals"].shape)  # (3, 1024)
```
