"""End-to-end checks for the pipeline. Run with unittest."""

from itertools import islice
import unittest

from main import CONFIRM_WINDOWS, Pipeline, run
from stream import WINDOW_SECONDS


class PipelineTests(unittest.TestCase):
    def test_fault_is_named_and_motor_cut(self):
        for fault in ("imbalance", "bearing"):
            results = list(run(fault, fault_at=2.0, ramp_seconds=0.0))
            last = results[-1]
            self.assertTrue(last["motor_off"])
            self.assertEqual(last["fault"], fault)
            self.assertFalse(any(r["motor_off"] for r in results[:-1]))
            # Worst case: fault starts just after a window begins.
            self.assertGreater(last["latency"], 0)
            self.assertLessEqual(last["latency"], (CONFIRM_WINDOWS + 1) * WINDOW_SECONDS)

    def test_healthy_motor_stays_on(self):
        # Single flagged windows are allowed, CONFIRM_WINDOWS absorbs them.
        for r in islice(run("normal"), 300):
            self.assertFalse(r["motor_off"])
            self.assertIsNone(r["latency"])

    def test_inject_and_reset(self):
        pipeline = Pipeline()
        for _ in range(10):
            self.assertFalse(pipeline.step()["motor_off"])

        pipeline.inject("bearing")
        for _ in range(CONFIRM_WINDOWS):
            result = pipeline.step()
        self.assertTrue(result["motor_off"])
        self.assertAlmostEqual(result["t"], (10 + CONFIRM_WINDOWS - 1) * WINDOW_SECONDS)
        self.assertAlmostEqual(result["latency"], CONFIRM_WINDOWS * WINDOW_SECONDS)
        self.assertIs(pipeline.step(), result)

        pipeline.reset()
        result = pipeline.step()
        self.assertFalse(result["motor_off"])
        self.assertEqual(result["t"], 0.0)


if __name__ == "__main__":
    unittest.main()
