"""Features for one 1024-sample vibration window at 5000 samples/second.

Order: RMS, crest factor, excess kurtosis, 20-60 Hz RMS,
60-500 Hz RMS, 500-2000 Hz RMS, 500-2000 Hz excess kurtosis.

The 20-60 Hz band tracks the motor's ~30 Hz rotation (imbalance). The high
band and its kurtosis track the short, high-frequency taps from bearing wear.
All features are calculated from the signal alone; no ground-truth fields enter.
"""

import numpy as np


SAMPLE_RATE = 5000
WINDOW_SAMPLES = 1024
FEATURE_NAMES = (
    "rms",
    "crest_factor",
    "excess_kurtosis",
    "rms_20_60_hz",
    "rms_60_500_hz",
    "rms_500_2000_hz",
    "excess_kurtosis_500_2000_hz",
)


def _excess_kurtosis(x):
    power = np.mean(x * x)
    if power < 1e-24:
        return 0.0
    return float(np.mean(x**4) / (power * power) - 3.0)


def extract(signal):
    """Return seven finite floats in FEATURE_NAMES order."""
    x = np.asarray(signal, dtype=np.float64)
    if x.shape != (WINDOW_SAMPLES,) or not np.all(np.isfinite(x)):
        raise ValueError("signal must contain exactly 1024 finite samples")

    # Remove sensor DC offset before measuring vibration.
    x = x - np.mean(x)
    rms = float(np.sqrt(np.mean(x * x)))
    crest_factor = float(np.max(np.abs(x)) / rms) if rms > 1e-12 else 0.0

    # A Hann window reduces spectral leakage from the 30 Hz tone. Correcting
    # for its mean-square gain keeps the band measurements in signal RMS units.
    window = np.hanning(WINDOW_SAMPLES)
    spectrum = np.fft.rfft(x * window)
    frequencies = np.fft.rfftfreq(WINDOW_SAMPLES, d=1.0 / SAMPLE_RATE)
    one_sided_power = np.abs(spectrum) ** 2
    one_sided_power[1:-1] *= 2.0
    scale = WINDOW_SAMPLES * np.sum(window * window)

    def band_rms(low, high):
        selected = (frequencies >= low) & (frequencies < high)
        return float(np.sqrt(np.sum(one_sided_power[selected]) / scale))

    low_rms = band_rms(20, 60)
    mid_rms = band_rms(60, 500)
    high_rms = band_rms(500, 2000)

    # Isolate the high-frequency content before measuring impulsiveness. This
    # can expose weak bearing taps hidden under the stronger 30 Hz motor tone.
    high_spectrum = np.fft.rfft(x)
    high_spectrum[(frequencies < 500) | (frequencies >= 2000)] = 0
    high_signal = np.fft.irfft(high_spectrum, n=WINDOW_SAMPLES)

    return [
        rms,
        crest_factor,
        _excess_kurtosis(x),
        low_rms,
        mid_rms,
        high_rms,
        _excess_kurtosis(high_signal),
    ]
