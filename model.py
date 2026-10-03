# takes the list from extract(), returns:
#   {"anomaly": bool, "fault": str, "confidence": float}
#
# fault is "healthy", "imbalance" or "bearing"
# idea: Isolation Forest for anomaly, then a classifier (your call)

# takes the list from extract(), returns:
#   {"anomaly": bool, "fault": str, "confidence": float}
#
# fault is "healthy", "imbalance" or "bearing"
# idea: Isolation Forest for anomaly, then a classifier (your call)


"""Integration imports predict(features). No shutdown decisions here."""
from pathlib import Path
import hashlib
import joblib
import numpy as np
from features import FEATURE_NAMES, SAMPLE_RATE, WINDOW_SAMPLES

ROOT = Path(__file__).resolve().parent
_bundle = None

def predict(features):
    global _bundle
    if _bundle is None:
        path = ROOT / 'anomaly_model.joblib'
        if not path.exists():
            raise FileNotFoundError('Run python3 train_model.py first.')
        bundle = joblib.load(path)  # Load only trusted team-produced artifacts.
        fingerprint = hashlib.sha256((ROOT / 'features.py').read_bytes()).hexdigest()
        if (bundle['feature_names'] != list(FEATURE_NAMES)
                or bundle['sample_rate'] != SAMPLE_RATE
                or bundle['window_samples'] != WINDOW_SAMPLES
                or bundle['features_sha256'] != fingerprint):
            raise ValueError('features.py changed since training. Retrain the model.')
        _bundle = bundle
    x = np.asarray(features, dtype=float)
    if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
        raise ValueError(f'Expected {len(FEATURE_NAMES)} finite features in FEATURE_NAMES order.')
    anomaly = bool(_bundle['detector'].predict(x.reshape(1, -1))[0] == -1)
    # Confidence is unavailable until a classifier is added. Display N/A.
    return {'anomaly': anomaly, 'fault': 'unknown' if anomaly else 'normal', 'confidence': 0.0}
