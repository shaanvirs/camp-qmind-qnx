"""Integration: stream -> extract -> predict -> motor off, with latency.

app.py can use either:
    run(...)      generator, one dict per window, ends after shutoff
    Pipeline      step() / inject() / reset(), for dashboard buttons

Each window dict:
    t           window start time (s)
    signal      1024 vibration samples
    features    list from extract()
    anomaly     bool
    fault       "normal", "imbalance" or "bearing"
    confidence  float
    motor_off   bool
    shutoff_t   time the motor was cut, None until shutoff
    latency     seconds from fault start to shutoff, None until shutoff
    compute_ms  time spent in extract + predict for this window
    true_state, fault_start_t   ground truth, display only

Latency = window end - fault_start_t. A window's samples only exist once it
has filled, so the ~0.2 s window length is included. Compute time is reported
separately because it depends on how busy the laptop is.
"""

from itertools import islice
import time

from features import extract
import model
from stream import WINDOW_SECONDS, stream

# Anomalous windows in a row needed before shutoff. Each adds ~0.2 s latency.
CONFIRM_WINDOWS = 2

# Placeholder thresholds, healthy max is about 0.71 and 0.044.
IMBALANCE_RMS_20_60 = 0.78
BEARING_RMS_500_2000 = 0.06


def placeholder_predict(features):
    """Simple thresholds, used when no trained model file exists."""
    low_rms, high_rms = features[3], features[5]
    if high_rms > BEARING_RMS_500_2000:
        fault = "bearing"
    elif low_rms > IMBALANCE_RMS_20_60:
        fault = "imbalance"
    else:
        fault = "normal"
    return {"anomaly": fault != "normal", "fault": fault, "confidence": 1.0}


# False means the thresholds above are running instead of the trained model,
# because anomaly_model.joblib is missing or was trained on an older
# features.py. The dashboard shows a warning when this is False.
USING_REAL_MODEL = False
MODEL_PROBLEM = None


def _choose_predictor():
    """Pick the predictor once, at import, and say so if the model is unusable.

    The first real prediction also loads the model file, which takes a few
    seconds, so doing it here keeps it out of the middle of a demo run.
    Only model-file problems fall back. Once the model answers, later errors
    are real bugs and are allowed to raise.
    """
    global USING_REAL_MODEL, MODEL_PROBLEM
    probe = extract(next(stream("normal"))["signal"])
    try:
        model.predict(probe)
    except (FileNotFoundError, ValueError) as problem:
        MODEL_PROBLEM = str(problem)
        print(f"WARNING: using threshold placeholder, not the model. {problem}")
        return placeholder_predict
    USING_REAL_MODEL = True
    return model.predict


predict = _choose_predictor()


class Pipeline:
    def __init__(self, fault="normal", fault_at=5.0, severity=1.0, ramp_seconds=0.0):
        self.reset(fault, fault_at, severity, ramp_seconds)

    def reset(self, fault="normal", fault_at=5.0, severity=1.0, ramp_seconds=0.0):
        """Restart from t=0 with the motor on."""
        self.windows_done = 0
        self.bad_in_a_row = 0
        self.motor_on = True
        self.shutoff_t = None
        self.latency = None
        self.last = None
        self._stream = stream(fault, fault_at, severity, ramp_seconds=ramp_seconds)

    def inject(self, fault, severity=1.0, ramp_seconds=0.0):
        """Start a fault at the next window."""
        fault_at = self.windows_done * WINDOW_SECONDS
        new_stream = stream(fault, fault_at, severity, ramp_seconds=ramp_seconds)
        # A new stream starts at t=0, so skip ahead to where we are now.
        self._stream = islice(new_stream, self.windows_done, None)

    def step(self):
        """Process one window. After shutoff, keeps returning the last result."""
        if not self.motor_on:
            return self.last

        window = next(self._stream)
        self.windows_done += 1

        started = time.perf_counter()
        features = extract(window["signal"])
        result = predict(features)

        compute_ms = (time.perf_counter() - started) * 1000

        self.bad_in_a_row = self.bad_in_a_row + 1 if result["anomaly"] else 0
        if self.bad_in_a_row >= CONFIRM_WINDOWS:
            self.motor_on = False
            self.shutoff_t = window["t"] + WINDOW_SECONDS
            # fault_start_t is None on a false trip, so there is no latency.
            if window["fault_start_t"] is not None:
                self.latency = self.shutoff_t - window["fault_start_t"]

        self.last = {
            "t": window["t"],
            "signal": window["signal"],
            "features": features,
            "anomaly": result["anomaly"],
            "fault": result["fault"],
            "confidence": result["confidence"],
            "motor_off": not self.motor_on,
            "shutoff_t": self.shutoff_t,
            "latency": self.latency,
            "compute_ms": compute_ms,
            "true_state": window["true_state"],
            "fault_start_t": window["fault_start_t"],
        }
        return self.last


def run(fault="bearing", fault_at=5.0, severity=1.0, ramp_seconds=1.0, realtime=False):
    """Yield one dict per window and stop after the motor is cut.

    realtime=True sleeps so windows arrive at motor speed. The dashboard
    records a whole run and replays it, so it leaves this off.
    """
    pipeline = Pipeline(fault, fault_at, severity, ramp_seconds)
    start = time.perf_counter()
    while pipeline.motor_on:
        result = pipeline.step()
        if realtime:
            window_end = result["t"] + WINDOW_SECONDS
            time.sleep(max(0.0, window_end - (time.perf_counter() - start)))
        yield result


if __name__ == "__main__":
    for r in run(realtime=True):
        state = "OFF" if r["motor_off"] else "ON "
        print(f"t={r['t']:6.2f}s  motor={state}  fault={r['fault']:9s}  true={r['true_state']}")
    print(f"latency: {r['latency']:.3f} s  (+ {r['compute_ms']:.0f} ms compute)")
