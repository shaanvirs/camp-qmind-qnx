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
    import main
    USING_FAKE = False
    # main.py runs, but it may have fallen back to thresholds if the trained
    # model file is missing or stale. Say so rather than demoing silently.
    MODEL_WARNING = None if main.USING_REAL_MODEL else main.MODEL_PROBLEM
except Exception:
    USING_FAKE = True
    MODEL_WARNING = None
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
st.set_page_config(page_title="Predictive maintenance · QMIND × QNX", page_icon="◉", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --paper: #f6f4ee; --ink: #181b20; --coral: #ff493e; --rule: #d6d5ce; }
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); font-family: Inter, sans-serif; }
[data-testid="stHeader"] { display: none; }
.block-container { max-width: 1440px; padding: 2.3rem 3rem 1.5rem; }
[data-testid="stVerticalBlock"] { gap: 1rem; }
h1, h2, h3, h4, button { font-family: Manrope, Inter, sans-serif; }
.masthead { display: flex; align-items: center; justify-content: space-between; gap: 24px; border-bottom: 1px solid var(--ink); padding-bottom: 17px; }
.masthead .partner { font: 800 14px Manrope, sans-serif; letter-spacing: -.025em; }
.masthead .edition { color: #5e6268; font-size: 10px; letter-spacing: .08em; text-transform: uppercase; }
.hero { display: grid; grid-template-columns: 1fr auto; gap: 32px; align-items: center; padding: 21px 0 10px; }
.hero .eyebrow { font-size: 10px; font-weight: 600; letter-spacing: .1em; color: #5e6268; text-transform: uppercase; margin-bottom: 8px; }
.hero h1 { font: 800 clamp(34px, 4.3vw, 58px)/1.13 Manrope, sans-serif; letter-spacing: -.055em; padding: 0; color: var(--ink); margin: 0; }
.hero p { font: 400 12px/1.7 Inter, sans-serif; color: #5e6268; margin: 13px 0 0; }
.qnx-mark { width: 116px; height: 116px; display: grid; place-items: center; color: white; background: var(--coral); font: 400 33px Manrope, sans-serif; letter-spacing: -.07em; }
[data-testid="stForm"] { border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); border-radius: 0; padding: 17px 0 12px; }
[data-testid="stForm"] [data-testid="stWidgetLabel"] p { font: 500 11px Inter, sans-serif; color: #5e6268; }
[data-baseweb="select"] > div { border-radius: 2px; background: transparent; border-color: #b7b7b1; font: 500 13px Inter, sans-serif; min-height: 40px; }
[data-testid="stSlider"] { padding-top: 0; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { font-family: Inter, sans-serif; }
[data-testid="stFormSubmitButton"] button { min-height: 42px; border-radius: 2px; background: var(--coral); color: #181b20; border: 1px solid var(--coral); box-shadow: none; font-weight: 700; }
[data-testid="stFormSubmitButton"] button p { font: 700 12px Manrope, sans-serif; }
[data-testid="stFormSubmitButton"] button:hover { background: #ef3e34; border-color: #ef3e34; color: #181b20; }
[data-testid="stFormSubmitButton"] button:focus-visible { outline: 2px solid var(--ink); outline-offset: 3px; }
[data-testid="stAlert"] { border-radius: 0; font-size: 12px; }
iframe { display: block; border: none; border-radius: 0; }
.history-heading { display: flex; align-items: baseline; justify-content: space-between; border-top: 1px solid var(--ink); padding-top: 18px; margin-top: 6px; }
.history-heading h2 { font: 700 19px Manrope, sans-serif; letter-spacing: -.035em; padding: 0; margin: 0; }
.history-heading span { font-size: 10px; color: #5e6268; }
[data-testid="stExpander"] { border-radius: 0; border: 0; border-top: 1px solid var(--rule); }
.footer { display: flex; justify-content: space-between; border-top: 1px solid var(--rule); padding-top: 16px; color: #5e6268; font-size: 10px; }
@media (max-width: 816px) {
 [data-testid="stIFrame"], [data-testid="stElementContainer"]:has(> [data-testid="stIFrame"]) { height: 1600px; }
}
@media (max-width: 360px) {
 [data-testid="stIFrame"], [data-testid="stElementContainer"]:has(> [data-testid="stIFrame"]) { height: 1700px; }
}
@media (max-width: 720px) {
 [data-testid="stForm"] [data-testid="stHorizontalBlock"] { display: grid; grid-template-columns: 1fr 1fr; gap: 16px 22px; }
 [data-testid="stForm"] [data-testid="stColumn"] { width: auto; min-width: 0; }
 [data-testid="stForm"] [data-testid="stColumn"]:first-child, [data-testid="stForm"] [data-testid="stColumn"]:last-child { grid-column: 1 / -1; }
 .block-container { padding: 2rem 1rem 1rem; }
 .masthead .edition { font-size: 8px; text-align: right; }
 .masthead .partner { font-size: 12px; }
 .hero { gap: 15px; padding-top: 12px; }
 .hero h1 { font-size: 35px; }
 .hero p { font-size: 11px; }
 .qnx-mark { width: 65px; height: 85px; font-size: 25px; }
 .history-heading span { font-size: 9px; }
}
</style>
<div class="masthead"><div class="partner">QMIND × BlackBerry QNX</div><div class="edition">Camp QMIND · MVP / October 2026</div></div>
<div class="hero"><div><div class="eyebrow">Edge-AI motor monitoring</div><h1>Predictive maintenance.</h1><p>Inject a fault. Watch the signal change and the motor shut down.</p></div><div class="qnx-mark" aria-label="QNX">QNX</div></div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "run" not in st.session_state:
    st.session_state.run = None

with st.form("experiment", border=False):
    fault_col, severity_col, onset_col, run_col = st.columns([1.15, 1, 1, 1], gap="large", vertical_alignment="bottom")
    with fault_col:
        fault = st.selectbox("Fault to inject", ["bearing", "imbalance"], format_func=str.title)
    with severity_col:
        severity = st.slider("Fault severity", 0.1, 2.0, 1.0, 0.1)
    with onset_col:
        fault_at = st.slider("Fault onset · seconds", 2.0, 10.0, 5.0, 0.5)
    with run_col:
        start = st.form_submit_button("Run experiment →", type="primary", use_container_width=True)

if USING_FAKE:
    st.warning("Placeholder detector active. The demo pipeline is unavailable.")
elif MODEL_WARNING:
    st.warning(f"Threshold detector active. The trained model is unavailable: {MODEL_WARNING}")

if start:
    with st.spinner("Preparing experiment…"):
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
components.html(html, height=880, scrolling=True)

st.markdown('<div class="history-heading"><h2>Experiment history</h2><span>Results from this session</span></div>', unsafe_allow_html=True)
if st.session_state.history:
    st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)
else:
    st.caption("Your completed experiments will appear here.")

with st.expander("From signal to shutdown"):
    st.markdown(
        "**Sensor → Features → Model → Shutoff → Dashboard**\n\n"
        "A simulated 30 Hz motor produces 5,000 signal samples per second. "
        "Every 1,024 samples, the pipeline extracts vibration features and checks for a fault. "
        "Consecutive anomalous windows trigger a simulated power cut.\n\n"
        "The model never receives the fault onset time. Shutdown response measures the "
        "time from fault onset to the simulated power cut; the target is 0.50 seconds. "
        "The motor and charts replay the recorded experiment. Playback speed changes "
        "the presentation only, and Replay uses the same recorded result."
    )
st.markdown('<div class="footer"><span>QMIND × BlackBerry QNX</span><span>Predictive maintenance · MVP experiment</span></div>', unsafe_allow_html=True)
