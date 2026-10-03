# quick check that stream/features/model match the interface
# python check_interfaces.py
import numpy as np

ok = True


def check(name, cond, msg=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else "  " + msg))
    ok = ok and cond


win = None
try:
    from stream import stream
    win = next(iter(stream()))
    check("stream keys", set(win) >= {"t", "signal", "true_state", "fault_start_t"}, str(list(win)))
    check("stream 1024 samples", len(win["signal"]) == 1024, str(len(win["signal"])))
except Exception as e:
    check("stream", False, repr(e))

feats = None
try:
    from features import extract
    sig = win["signal"] if win else np.random.randn(1024)
    feats = extract(sig)
    check("features returns list of floats", isinstance(feats, list) and all(isinstance(x, float) for x in feats), type(feats).__name__)
    check("features same length each call", len(extract(np.random.randn(1024))) == len(feats))
except Exception as e:
    check("features", False, repr(e))

try:
    from model import predict
    out = predict(feats if isinstance(feats, list) else [0.0] * 8)
    check("model keys", isinstance(out, dict) and set(out) >= {"anomaly", "fault", "confidence"}, str(out))
    check("model types", isinstance(out["anomaly"], bool) and isinstance(out["fault"], str) and isinstance(out["confidence"], float), str(out))
except Exception as e:
    check("model", False, repr(e))

print("\nall good" if ok else "\nsomething's broken")
