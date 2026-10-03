# fake motor vibration data
#
# stream() yields dicts like:
#   {"t": float, "signal": 1024 samples, "true_state": str, "fault_start_t": float or None}
#
# 5 kHz, so each window is ~0.2 s
# true_state: "healthy", "imbalance" or "bearing"


def stream():
    pass
