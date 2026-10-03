# streamlit dashboard, run with: streamlit run app.py
# live chart, status, fault buttons, MOTOR OFF banner, latency
import time
import numpy as np
import streamlit as st

# =====================================================================
# 1. DATA SOURCE: real run() if main.py is ready, otherwise a fake one
# =====================================================================
try:
    from main import run
    USING_FAKE = False
except Exception:
    USING_FAKE = True

    def run(fault="bearing", fault_at=5.0, severity=1.0):
        """Fake run(): same dict shape the real one should yield."""
        fs, n = 5000, 1024
        win, t, flagged = n / fs, 0.0, 0
        while t < 20:
            tt = np.arange(n) / fs + t
            sig = np.sin(2 * np.pi * 30 * tt) + 0.1 * np.random.randn(n)
            faulty = t >= fault_at
            if faulty:
                ramp = min(1.0, (t - fault_at) / 1.0) * severity
                sig = sig * (1 + ramp) + ramp * 0.5 * np.random.randn(n)
            anomaly = faulty and t >= fault_at + 0.4
            flagged = flagged + 1 if anomaly else 0
            off = flagged >= 2
            yield {
                "t": t, "signal": sig, "anomaly": anomaly,
                "fault": fault if anomaly else "normal",
                "confidence": 0.9 if anomaly else 0.1,
                "motor_off": off,
                "shutoff_t": t + win if off else None,
                "latency": (t + win - fault_at) if off else None,
                "fault_start_t": fault_at if faulty else None,
            }
            if off:
                return
            t += win


# =====================================================================
# 2. LOOK AND FEEL: tweak colors and CSS here
# =====================================================================
COLORS = {
    "healthy": "#2EA043",   # green
    "warning": "#D29922",   # amber
    "off": "#DA3633",       # red
    "idle": "#30363D",      # grey
}
TARGET_LATENCY = 0.5
N = 1024

st.set_page_config(page_title="Motor Monitor", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.status-card {border-radius: 14px; padding: 22px 28px; margin-bottom: 18px;
              color: white; display: flex; justify-content: space-between;
              align-items: center;}
.status-title {font-size: 2rem; font-weight: 700; letter-spacing: .5px;}
.status-sub {font-size: 1rem; opacity: .85; margin-top: 2px;}
.metric-card {background: #161B22; border: 1px solid #30363D;
              border-radius: 12px; padding: 16px 20px;}
.metric-label {font-size: .8rem; text-transform: uppercase;
               letter-spacing: 1px; color: #8B949E;}
.metric-value {font-size: 2rem; font-weight: 700; margin-top: 4px;}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 3. COMPONENTS: one function per UI piece, easy to edit or swap
# =====================================================================
def status_card(slot, state, subtitle=""):
    titles = {"idle": "MOTOR IDLE", "healthy": "RUNNING: HEALTHY",
              "warning": "ANOMALY DETECTED", "off": "MOTOR OFF"}
    slot.markdown(
        f"""<div class="status-card" style="background:{COLORS[state]}">
              <div><div class="status-title">{titles[state]}</div>
              <div class="status-sub">{subtitle}</div></div>
            </div>""", unsafe_allow_html=True)


def metric_card(slot, label, value, color="#E6EDF3"):
    slot.markdown(
        f"""<div class="metric-card"><div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{value}</div></div>""",
        unsafe_allow_html=True)


def latency_card(slot, latency):
    if latency is None:
        metric_card(slot, "Response time", "-")
    else:
        ok = latency <= TARGET_LATENCY
        metric_card(slot, f"Response time (target {TARGET_LATENCY}s)",
                    f"{latency:.2f} s", COLORS["healthy"] if ok else COLORS["off"])


def history_table(slot):
    h = st.session_state.history
    if h:
        slot.dataframe(h, use_container_width=True, hide_index=True)
    else:
        slot.caption("No runs yet.")


# =====================================================================
# 4. PAGE LAYOUT
# =====================================================================
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("⚙️ Controls")
    fault = st.selectbox("Fault to inject", ["bearing", "imbalance"])
    severity = st.slider("Severity", 0.1, 2.0, 1.0, 0.1)
    fault_at = st.slider("Fault starts at (s)", 2.0, 10.0, 5.0, 0.5)
    start = st.button("▶ Start demo", type="primary", use_container_width=True)
    if USING_FAKE:
        st.warning("Using FAKE data (main.py not ready)")

st.title("Edge-AI Motor Health Monitor")
st.caption("Simulated motor → features → model → shutoff. "
           "Pick a fault, press Start, watch it get caught.")

status_slot = st.empty()
c1, c2, c3 = st.columns(3)
state_slot, fault_slot, lat_slot = c1.empty(), c2.empty(), c3.empty()

st.subheader("Live vibration")
chart_slot = st.empty()

st.subheader("Past runs")
history_slot = st.empty()

# =====================================================================
# 5. RUN LOOP
# =====================================================================
if not start:
    status_card(status_slot, "idle", "Press Start demo in the sidebar")
    metric_card(state_slot, "Status", "-")
    metric_card(fault_slot, "Detected fault", "-")
    latency_card(lat_slot, None)
    chart_slot.line_chart(np.zeros(N * 5), height=280)
    history_table(history_slot)
else:
    buf = np.zeros(N * 5)  # rolling window of ~1 second
    last = None
    for w in run(fault=fault, fault_at=fault_at, severity=severity):
        last = w
        buf = np.concatenate([buf[N:], np.asarray(w["signal"])[:N]])
        chart_slot.line_chart(buf, height=280)

        if w["motor_off"]:
            status_card(status_slot, "off", f"Fault confirmed at t = {w['t']:.2f} s")
            metric_card(state_slot, "Status", "STOPPED", COLORS["off"])
        elif w["anomaly"]:
            status_card(status_slot, "warning", "Confirming before shutoff...")
            metric_card(state_slot, "Status", "SUSPICIOUS", COLORS["warning"])
        else:
            status_card(status_slot, "healthy", f"t = {w['t']:.1f} s")
            metric_card(state_slot, "Status", "HEALTHY", COLORS["healthy"])

        metric_card(fault_slot, "Detected fault",
                    w["fault"].title() if w["anomaly"] else "None")
        latency_card(lat_slot, w["latency"])
        time.sleep(0.1)  # pacing for humans; remove if main.py paces itself

    if last and last["latency"] is not None:
        st.session_state.history.append({
            "fault": fault, "severity": severity,
            "latency (s)": round(last["latency"], 3),
            "met target": last["latency"] <= TARGET_LATENCY,
        })
    history_table(history_slot)