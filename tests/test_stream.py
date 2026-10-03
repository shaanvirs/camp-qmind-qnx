"""Contract, timing and signal checks for the sensor handoff. Run with unittest."""

from itertools import islice
import math
import unittest

import numpy as np

from stream import (
    BEARING_AMPLITUDE, BEARING_DECAY_SECONDS, BEARING_IMPACT_HZ, BEARING_RING_HZ,
    HEALTHY_AMPLITUDE, IMBALANCE_AMPLITUDE, MOTOR_HZ, NOISE_STD,
    SAMPLE_RATE, WINDOW_SIZE, WINDOW_SECONDS, stream,
)


def take(fault="bearing", *, windows=35, **kwargs):
    return list(islice(stream(fault, **kwargs), windows))


def flat(windows):
    return np.concatenate([w["signal"] for w in windows])


class SensorTests(unittest.TestCase):
    def test_original_team_call_and_schema(self):
        for fault in ("imbalance", "bearing"):
            for k, window in enumerate(islice(stream(fault, 5.0, 1.0), 40)):
                self.assertEqual(set(window), {"t", "signal", "true_state", "fault_start_t"})
                self.assertIsInstance(window["t"], float)
                self.assertEqual(window["signal"].shape, (1024,))
                self.assertTrue(np.issubdtype(window["signal"].dtype, np.floating))
                self.assertTrue(np.isfinite(window["signal"]).all())
                self.assertAlmostEqual(window["t"], k * 1024 / 5000, places=13)

    def test_pre_onset_samples_are_unchanged_at_boundaries(self):
        healthy = flat(take("normal", seed=7))
        times = np.arange(len(healthy)) / SAMPLE_RATE
        for fault in ("imbalance", "bearing"):
            for onset in (0.0, 0.00013, WINDOW_SECONDS, 5.0, 5.00013, 9.0):
                with self.subTest(fault=fault, onset=onset):
                    windows = take(fault, fault_at=onset, seed=7)
                    np.testing.assert_array_equal(flat(windows)[times < onset], healthy[times < onset])
                    for window in windows:
                        last_sample_t = window["t"] + (WINDOW_SIZE - 1) / SAMPLE_RATE
                        active = last_sample_t >= onset
                        self.assertEqual(window["true_state"], fault if active else "normal")
                        self.assertEqual(window["fault_start_t"], onset if active else None)

    def test_default_onset_is_inside_window_and_keeps_exact_timestamp(self):
        windows = take(seed=1)
        first = next(w for w in windows if w["true_state"] == "bearing")
        self.assertAlmostEqual(first["t"], 4.9152)
        self.assertEqual(first["fault_start_t"], 5.0)
        self.assertAlmostEqual(first["t"] + WINDOW_SECONDS - 5.0, 0.12)

    def test_onset_between_last_sample_and_next_window(self):
        windows = take(fault_at=WINDOW_SECONDS - 0.0001, windows=2, seed=3)
        self.assertEqual(windows[0]["true_state"], "normal")
        self.assertEqual(windows[1]["true_state"], "bearing")

    def test_zero_severity_and_infinite_onset_are_healthy(self):
        expected = take("normal", windows=10, seed=18)
        for fault in ("imbalance", "bearing"):
            for options in ({"severity": 0, "fault_at": 0}, {"fault_at": math.inf}):
                actual = take(fault, windows=10, seed=18, **options)
                np.testing.assert_array_equal(flat(actual), flat(expected))
                self.assertTrue(all(w["true_state"] == "normal" and w["fault_start_t"] is None for w in actual))

    def test_severity_scales_only_fault_component(self):
        healthy = flat(take("normal", windows=8, seed=23))
        for fault in ("imbalance", "bearing"):
            base = flat(take(fault, windows=8, seed=23, fault_at=0.31)) - healthy
            self.assertGreater(np.linalg.norm(base), 1)
            for severity in (0.05, 0.25, 0.5, 2.0):
                with self.subTest(fault=fault, severity=severity):
                    measured = flat(take(fault, windows=8, seed=23, fault_at=0.31, severity=severity)) - healthy
                    np.testing.assert_allclose(measured, severity * base, atol=2e-15, rtol=1e-12)

    def test_healthy_phase_and_noise_continue_between_windows(self):
        values = flat(take("normal", windows=3, seed=12))
        expected_times = np.arange(values.size) / SAMPLE_RATE
        expected_noise = np.random.default_rng(12).normal(0, NOISE_STD, values.size)
        expected = HEALTHY_AMPLITUDE * np.sin(2 * np.pi * MOTOR_HZ * expected_times) + expected_noise
        np.testing.assert_allclose(values, expected, atol=1e-13, rtol=0)
        self.assertFalse(np.array_equal(values[:1024], values[1024:2048]))

    def test_imbalance_is_extra_rotation_frequency(self):
        healthy = flat(take("normal", windows=3, seed=12))
        actual = flat(take("imbalance", windows=3, fault_at=0, seed=12)) - healthy
        expected = IMBALANCE_AMPLITUDE * np.sin(2 * np.pi * MOTOR_HZ * np.arange(actual.size) / SAMPLE_RATE)
        np.testing.assert_allclose(actual, expected, atol=2e-13)

    def test_bearing_matches_direct_impulse_superposition_across_windows(self):
        onset = 0.173
        healthy = flat(take("normal", windows=4, seed=4))
        actual = flat(take("bearing", windows=4, fault_at=onset, seed=4)) - healthy
        times = np.arange(len(actual)) / SAMPLE_RATE
        expected = np.zeros_like(actual)
        # Independent reference: sum every individual impact since onset.
        for impact_t in np.arange(onset, times[-1], 1 / BEARING_IMPACT_HZ):
            active = times >= impact_t
            age = times[active] - impact_t
            expected[active] += BEARING_AMPLITUDE * np.exp(-age / BEARING_DECAY_SECONDS) * np.sin(2 * np.pi * BEARING_RING_HZ * age)
        np.testing.assert_allclose(actual, expected, atol=4e-11, rtol=0)

    def test_ramp_envelope_and_onset_metadata(self):
        windows = 12
        onset, ramp = 0.37, 0.81
        healthy = flat(take("normal", windows=windows, seed=9))
        times = np.arange(healthy.size) / SAMPLE_RATE
        envelope = np.clip((times - onset) / ramp, 0, 1)
        for fault in ("imbalance", "bearing"):
            step = flat(take(fault, windows=windows, fault_at=onset, seed=9)) - healthy
            ramped = take(fault, windows=windows, fault_at=onset, seed=9, ramp_seconds=ramp)
            np.testing.assert_allclose(flat(ramped) - healthy, envelope * step, atol=2e-15, rtol=1e-12)
            self.assertEqual(next(w["fault_start_t"] for w in ramped if w["true_state"] != "normal"), onset)

    def test_same_seed_repeats_and_different_seeds_vary(self):
        a = flat(take(seed=1, windows=5))
        np.testing.assert_array_equal(a, flat(take(seed=1, windows=5)))
        self.assertFalse(np.array_equal(a, flat(take(seed=2, windows=5))))
        self.assertFalse(np.array_equal(next(stream())["signal"], next(stream())["signal"]))

    def test_generator_does_not_change_global_random_state(self):
        np.random.seed(27)
        expected = np.random.random(4)
        np.random.seed(27)
        next(stream(seed=33))
        np.testing.assert_array_equal(np.random.random(4), expected)

    def test_caller_can_mutate_old_window_without_corrupting_stream(self):
        source = stream(seed=19)
        first = next(source)
        saved = first["signal"].copy()
        next(source)
        np.testing.assert_array_equal(first["signal"], saved)
        first["signal"][:] = 999
        third = next(source)
        reference = take(seed=19, windows=3)[2]
        np.testing.assert_array_equal(third["signal"], reference["signal"])

    def test_faults_have_distinct_frequency_signatures(self):
        signals = {f: flat(take(f, windows=20, seed=91, fault_at=0)) for f in ("normal", "imbalance", "bearing")}
        size = len(signals["normal"])
        freq = np.fft.rfftfreq(size, 1 / SAMPLE_RATE)
        powers = {f: np.abs(np.fft.rfft(x * np.hanning(size))) ** 2 for f, x in signals.items()}
        low = (freq > 20) & (freq < 40)
        high = (freq > 500) & (freq < 1800)
        self.assertGreater(powers["imbalance"][low].sum(), 5 * powers["normal"][low].sum())
        self.assertGreater(powers["bearing"][high].sum(), 20 * powers["normal"][high].sum())

    def test_invalid_parameters_fail(self):
        cases = [
            {"fault": "typo"}, {"fault_at": -1}, {"fault_at": -math.inf},
            {"fault_at": math.nan}, {"severity": -0.1}, {"severity": math.inf},
            {"severity": math.nan}, {"severity": 1e308}, {"ramp_seconds": -1},
            {"ramp_seconds": math.inf}, {"ramp_seconds": math.nan},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                next(stream(**case))


if __name__ == "__main__":
    unittest.main()
