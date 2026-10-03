"""Render sensor comparisons and export examples; no model or team app required."""

from __future__ import annotations

import argparse
from itertools import islice
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stream import SAMPLE_RATE, WINDOW_SIZE, WINDOW_SECONDS, stream


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--fault-at", type=float, default=2.0)
    parser.add_argument("--ramp-seconds", type=float, default=2.0)
    parser.add_argument("--seconds", type=float, default=7.0)
    parser.add_argument("--severity", type=float, default=1.0)
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (args.seconds, args.fault_at, args.ramp_seconds, args.severity)):
        parser.error("all numeric options must be finite")
    if args.fault_at < 0 or args.ramp_seconds < 0 or args.severity <= 0:
        parser.error("onset/ramp must be nonnegative and severity must be positive")
    if not args.fault_at + args.ramp_seconds + WINDOW_SECONDS < args.seconds <= 120:
        parser.error("seconds must include one full window after the ramp and be <=120")
    args.out.mkdir(parents=True, exist_ok=True)
    count = math.ceil(args.seconds / WINDOW_SECONDS)
    states = ("normal", "imbalance", "bearing")
    # Matched noise isolates differences caused by faults. These demo examples
    # are deliberately NOT a train/test split. Training should use fresh seeds.
    runs = {
        state: list(islice(stream(state, args.fault_at, args.severity, seed=42,
                                 ramp_seconds=args.ramp_seconds), count))
        for state in states
    }
    colors = {"normal": "#23856d", "imbalance": "#c27515", "bearing": "#aa457e"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    comparison, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
    comparison.suptitle("QMIND sensor simulator · healthy, imbalance, bearing", fontsize=17)
    final_signals = []
    metrics = {}
    for row, state in enumerate(states):
        signal = runs[state][-1]["signal"]
        final_signals.append(signal)
        times = np.arange(WINDOW_SIZE) / SAMPLE_RATE
        axes[row, 0].plot(times * 1000, signal, color=colors[state], linewidth=1.0)
        axes[row, 0].set(title=state.title(), xlabel="Time within window (ms)",
                         ylabel="Vibration (arbitrary units)", ylim=(-5, 5))
        spectrum = np.abs(np.fft.rfft((signal - signal.mean()) * np.hanning(WINDOW_SIZE)))
        freq = np.fft.rfftfreq(WINDOW_SIZE, 1 / SAMPLE_RATE)
        axes[row, 1].plot(freq, spectrum, color=colors[state], linewidth=1.2)
        axes[row, 1].set(xlabel="Frequency (Hz)", ylabel="FFT magnitude", xlim=(0, 2500))
        axes[row, 1].axvline(30, color="#999999", linestyle=":", linewidth=0.8)
        axes[row, 1].axvline(1000, color="#999999", linestyle=":", linewidth=0.8)
        for ax in axes[row]:
            ax.grid(alpha=0.15)
        metrics[state] = {
            "rms": float(np.sqrt(np.mean(signal ** 2))),
            "peak_absolute": float(np.max(np.abs(signal))),
            "first_fault_window_t": next((w["t"] for w in runs[state] if w["true_state"] != "normal"), None),
        }
    comparison.savefig(args.out / "signal-comparison.png", dpi=170)
    plt.close(comparison)

    timeline, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True, sharex=True)
    timeline.suptitle("Fault growth over simulated time · known injection, not model detection", fontsize=15)
    healthy = np.concatenate([w["signal"] for w in runs["normal"]])
    sample_t = np.arange(healthy.size) / SAMPLE_RATE
    for ax, state in zip(axes, states[1:]):
        values = np.concatenate([w["signal"] for w in runs[state]])
        ax.plot(sample_t, values, color=colors[state], linewidth=0.35, alpha=0.8)
        ax.axvline(args.fault_at, color="#252525", linestyle="--", label="Fault begins")
        ax.axvline(args.fault_at + args.ramp_seconds, color="#777777", linestyle=":", label="Full severity")
        ax.set(title=state.title(), ylabel="Vibration (arbitrary units)", xlim=(0, args.seconds))
        ax.legend(loc="upper left")
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("Simulated time (seconds)")
    timeline.savefig(args.out / "fault-ramp.png", dpi=170)
    plt.close(timeline)

    np.savez_compressed(
        args.out / "example-windows.npz",
        signals=np.stack(final_signals), labels=np.array(states),
        sample_rate=SAMPLE_RATE, window_start=runs["normal"][-1]["t"],
        fault_at=args.fault_at, severity=args.severity, ramp_seconds=args.ramp_seconds,
    )
    summary = {
        "sample_rate_hz": SAMPLE_RATE, "window_samples": WINDOW_SIZE,
        "window_seconds": WINDOW_SECONDS, "fault_at": args.fault_at,
        "ramp_seconds": args.ramp_seconds, "severity": args.severity,
        "noise_seed": 42, "metrics": metrics,
        "purpose": "Sensor examples only; no detection, classifier, or shutoff is simulated here.",
    }
    (args.out / "preview-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved plots and example windows in {args.out.resolve()}")


if __name__ == "__main__":
    main()
