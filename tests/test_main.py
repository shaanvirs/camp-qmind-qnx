"""End-to-end checks for the pipeline. Run with unittest."""

from itertools import islice
import unittest

from main import CONFIRM_WINDOWS, Pipeline, run
from stream import WINDOW_SECONDS


class PipelineTests(unittest.TestCase):
    def test_fault_is_named_and_motor_cut(self):
        for fault in ("imbalance", "bearing"):
            results = list(run(fault, fault_at=2.0, realtime=False))
            last = results[-1]
            self.assertFalse(last["motor_on"])
            self.assertEqual(last["fault"], fault)
            self.assertTrue(all(r["motor_on"] for r in results[:-1]))
            # Worst case: fault starts just after a window begins.
            self.assertGreater(last["latency"], 0)
            self.assertLess(last["latency"], (CONFIRM_WINDOWS + 1) * WINDOW_SECONDS + 0.1)

    def test_healthy_motor_stays_on(self):
        for r in islice(run("normal", realtime=False), 300):
            self.assertTrue(r["motor_on"])
            self.assertFalse(r["anomaly"])
            self.assertIsNone(r["latency"])

    def test_inject_and_reset(self):
        pipeline = Pipeline()
        for _ in range(10):
            self.assertTrue(pipeline.step()["motor_on"])

        pipeline.inject("bearing")
        for _ in range(CONFIRM_WINDOWS):
            result = pipeline.step()
        self.assertFalse(result["motor_on"])
        self.assertAlmostEqual(result["t"], (10 + CONFIRM_WINDOWS - 1) * WINDOW_SECONDS)
        self.assertAlmostEqual(result["latency"], CONFIRM_WINDOWS * WINDOW_SECONDS, delta=0.1)
        self.assertIs(pipeline.step(), result)

        pipeline.reset()
        result = pipeline.step()
        self.assertTrue(result["motor_on"])
        self.assertEqual(result["t"], 0.0)


if __name__ == "__main__":
    unittest.main()
