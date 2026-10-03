# camp-qmind-qnx

Camp QMIND MVP (Oct 3-4). Same pipeline as the real QNX project:

```
fake motor data -> features -> model -> shutoff -> dashboard
   stream.py      features.py  model.py   main.py    app.py
```

Demo: healthy motor on the dashboard, we inject a fault and let it ramp up, the model catches it, names it, cuts the motor, and the response time shows on screen.

## Setup

```
git clone https://github.com/shaanvirs/camp-qmind-qnx.git
cd camp-qmind-qnx
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pieces

Owners are suggestions. The ideas in the last column are just a starting point, use something else if you think it's better.

| File | What it does | Owner | Ideas |
|---|---|---|---|
| stream.py | Fake vibration, healthy + faults (imbalance, bearing) | Daniel | sine waves + noise, fault = extra frequency or spikes, severity slider |
| features.py | Raw signal to numbers | | RMS, kurtosis, FFT bands |
| model.py | Detect a fault, then name it | | Isolation Forest on healthy data, then a classifier |
| main.py | Wire it together, time fault to shutoff | | shutoff after a few bad windows in a row |
| app.py | Dashboard | | Streamlit is already in requirements |

## Interface

Keep these the same so everything connects at 4:00. Use dummy inputs until the real pieces are ready.

```python
stream()           # yields {"t", "signal", "true_state", "fault_start_t"}
                   # signal = 1024 samples @ 5 kHz (~0.2 s)
extract(signal)    # -> list of floats (same length and order every call)
predict(features)  # -> {"anomaly": bool, "fault": str, "confidence": float}
```

Run `python check_interfaces.py` to see if everything still matches.

## Schedule

Sat 4:00 full pipeline running end to end (even if ugly)
Sun 11:30 code freeze, 12:30 pitch
