# camp-qmind-qnx

Camp QMIND MVP (Oct 3-4). Same pipeline as the real QNX project:

```
fake motor data -> features -> model -> shutoff -> dashboard
   stream.py      features.py  model.py   main.py    app.py
```

Demo: healthy motor on the dashboard, we inject a fault and let it ramp up, the model catches it, names it, cuts the motor, and the response time shows on screen.

## Setup

```
git clone <this repo>
cd qnx-camp-qmind
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pieces

Owners are suggestions, we pick at the 2:30 huddle. The ideas in the last column are just a starting point, use something else if you think it's better.

| File | What it does | Owner | Ideas |
|---|---|---|---|
| stream.py | Fake vibration, healthy + faults (imbalance, bearing) | Daniel | sine waves + noise, fault = extra frequency or spikes, severity slider |
| features.py | Raw signal to numbers | Alamjeet | RMS, kurtosis, FFT bands |
| model.py | Detect a fault, then name it | Aariz | Isolation Forest on healthy data, then a classifier |
| main.py | Wire it together, time fault to shutoff | Aryan | shutoff after a few bad windows in a row |
| app.py | Dashboard | Adam | Streamlit is already in requirements |

## Interface

Keep these the same so everything connects at 4:00. Use dummy inputs until the real pieces are ready.

```python
stream()           # yields {"t", "signal", "true_state", "fault_start_t"}
                   # signal = 1024 samples @ 5 kHz (~0.2 s)
extract(signal)    # -> list of floats (same length and order every call)
predict(features)  # -> {"anomaly": bool, "fault": str, "confidence": float}
```

Run `python check_interfaces.py` to see if everything still matches.

## Quick answers

- Not sure how to do your piece? Do the simplest version first, make it better later.
- Need something from someone else's file? Use the dummy version, don't wait on them.
- Want to change the interface? Ask in the channel first, it affects everyone.
- Stuck 20+ min? Post in the channel.
- Only edit your own file, small commits, pull before you push.

## Schedule

Sat 4:00 full pipeline running end to end (even if ugly)
Sun 11:30 code freeze, 12:30 pitch
