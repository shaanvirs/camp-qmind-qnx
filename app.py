# streamlit dashboard, run with: streamlit run app.py
# Python runs the pipeline for a whole demo up front (the stream is simulated
# time, unpaced), then ui/dashboard.html plays it back with a three.js motor,
# live oscilloscope, status, MOTOR OFF banner and response time.
import json
import math
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from stream import WINDOW_SECONDS, stream

try:
    from features import extract
except Exception:
    extract = None

# =====================================================================
# 1. DATA SOURCE: real run() if main.py is ready, otherwise a placeholder
# =====================================================================
try:
    from main import run
    USING_FAKE = False
except Exception:
    USING_FAKE = True
    CONFIRM_WINDOWS = 2

    def _placeholder_predict(f):
        """Threshold rules on two bands, only until model.py is ready."""
        low, high = f[3], f[5]
        scores = {"imbalance": low / 0.85, "bearing": high / 0.09}
        fault = max(scores, key=scores.get)
        score = scores[fault]
        if score < 1:
            return {"anomaly": False, "fault": "healthy",
                    "confidence": round(min(0.99, 1.0 - 0.4 * score), 2)}
        return {"anomaly": True, "fault": fault,
                "confidence": round(min(0.99, 0.6 + 0.25 * (score - 1)), 2)}

    def run(fault="bearing", fault_at=5.0, severity=1.0):
        """Placeholder run(): same dict shape the real one should yield."""
        flagged = 0
        for w in stream(fault, fault_at, severity, ramp_seconds=1.0):
            p = _placeholder_predict(extract(w["signal"]))
            flagged = flagged + 1 if p["anomaly"] else 0
            off = flagged >= CONFIRM_WINDOWS
            ready_t = w["t"] + WINDOW_SECONDS
            yield {
                "t": w["t"], "signal": w["signal"], "anomaly": p["anomaly"],
                "fault": p["fault"], "confidence": p["confidence"],
                "motor_off": off,
                "shutoff_t": ready_t if off else None,
                "latency": (ready_t - fault_at) if off else None,
                "fault_start_t": w["fault_start_t"],
            }
            if off:
                return


# =====================================================================
# 2. RECORD A RUN for the browser to play back
# =====================================================================
TARGET_LATENCY = 0.5
MAX_SECONDS = 30
SAMPLES_PER_WINDOW = 256  # downsampled for display only
DASHBOARD = Path(__file__).parent / "ui" / "dashboard.html"


def _num(v):
    return None if v is None else float(v)


def _envelope(sig, bins=SAMPLES_PER_WINDOW):
    """Downsample keeping the largest-magnitude sample per bin, so impacts survive."""
    s = sig[: (len(sig) // bins) * bins].reshape(bins, -1)
    picked = s[np.arange(bins), np.abs(s).argmax(axis=1)]
    return np.round(picked, 3).tolist()


def _display_features(sig):
    """RMS, kurtosis, 20-60 Hz RMS, 500-2000 Hz RMS for the feature bars."""
    if extract is None:
        return None
    try:
        f = extract(sig)
    except Exception:
        return None
    return [round(float(f[i]), 4) for i in (0, 2, 3, 5)]


def record_run(fault, fault_at, severity):
    frames = []
    max_windows = math.ceil(MAX_SECONDS / WINDOW_SECONDS)
    for w in run(fault=fault, fault_at=fault_at, severity=severity):
        sig = np.asarray(w["signal"], dtype=np.float64)
        frames.append({
            "t": round(float(w["t"]), 4),
            "s": _envelope(sig),
            "a": bool(w.get("anomaly")),
            "f": str(w.get("fault") or "healthy"),
            "c": _num(w.get("confidence")),
            "off": bool(w.get("motor_off")),
            "lat": _num(w.get("latency")),
            "fs": _num(w.get("fault_start_t")),
            "sh": _num(w.get("shutoff_t")),
            "x": _display_features(sig),
        })
        if frames[-1]["off"] or len(frames) >= max_windows:
            break
    return frames


# =====================================================================
# 3. PAGE
# =====================================================================
st.set_page_config(page_title="QMIND Motor Health", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1500px;}
header[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #0b1222, #070b14);
                           border-right: 1px solid rgba(148,163,184,.12);}
[data-testid="stSidebar"] h2 {font-size: 1.05rem; letter-spacing: .3px;}
iframe {border-radius: 20px;}
.hist-title {font-size: .8rem; letter-spacing: 1.6px; text-transform: uppercase;
             color: #56627a; font-weight: 600; margin: 6px 0 4px;}
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "run" not in st.session_state:
    st.session_state.run = None

with st.sidebar:
    st.header("⚙️ Demo controls")
    fault = st.selectbox("Fault to inject", ["bearing", "imbalance"],
                         format_func=str.title)
    severity = st.slider("Severity", 0.1, 2.0, 1.0, 0.1)
    fault_at = st.slider("Fault starts at (s)", 2.0, 10.0, 5.0, 0.5)
    start = st.button("▶ Start demo", type="primary", use_container_width=True)
    if USING_FAKE:
        st.warning("Using a placeholder detector (main.py has no run() yet).")
    with st.expander("How it works"):
        st.markdown(
            "1. **stream.py** simulates a 30 Hz motor at 5 kHz.\n"
            "2. **features.py** turns each 0.2 s window into numbers.\n"
            "3. **model.py** flags and names the fault.\n"
            "4. **main.py** cuts power after consecutive bad windows.\n\n"
            "The fault start time is shown for the audience only; "
            "the model never sees it.")

if start:
    with st.spinner("Running pipeline…"):
        frames = record_run(fault, fault_at, severity)
    n = len(st.session_state.history) + 1
    st.session_state.run = {"id": n, "fault": fault, "severity": severity,
                            "fault_at": fault_at, "frames": frames}
    last = frames[-1] if frames else {}
    lat = last.get("lat")
    st.session_state.history.append({
        "run": n,
        "injected": fault,
        "severity": severity,
        "fault at (s)": fault_at,
        "detected": last.get("f") if last.get("a") else "—",
        "response (s)": round(lat, 3) if lat is not None else None,
        "met target": (lat is not None and lat <= TARGET_LATENCY),
    })

r = st.session_state.run
payload = {
    "runId": r["id"] if r else 0,
    "fault": r["fault"] if r else fault,
    "severity": r["severity"] if r else severity,
    "frames": r["frames"] if r else [],
    "win": WINDOW_SECONDS,
    "spw": SAMPLES_PER_WINDOW,
    "target": TARGET_LATENCY,
    "speed": 1,
    "fake": USING_FAKE,
}
html = DASHBOARD.read_text(encoding="utf-8").replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
components.html(html, height=830, scrolling=True)

st.markdown('<div class="hist-title">Past runs</div>', unsafe_allow_html=True)
if st.session_state.history:
    st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)
else:
    st.caption("No runs yet. Press ▶ Start demo in the sidebar.")
