# takes the list from extract(), returns:
#   {"anomaly": bool, "fault": str, "confidence": float}
#
# fault is "normal", "imbalance" or "bearing"
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
            raise FileNotFoundError('No anomaly_model.joblib. Run: python3 make_data.py && python3 train_model.py')
        bundle = joblib.load(path)  # Load only trusted team-produced artifacts.
        fingerprint = hashlib.sha256((ROOT / 'features.py').read_bytes()).hexdigest()
        if (bundle['feature_names'] != list(FEATURE_NAMES)
                or bundle['sample_rate'] != SAMPLE_RATE
                or bundle['window_samples'] != WINDOW_SAMPLES
                or bundle['features_sha256'] != fingerprint):
            raise ValueError('features.py changed since training. Retrain the model.')
        if 'classifier' not in bundle:
            raise ValueError('Model file predates the classifier. Rerun python3 train_model.py')
        _bundle = bundle
    x = np.asarray(features, dtype=float)
    if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
        raise ValueError(f'Expected {len(FEATURE_NAMES)} finite features in FEATURE_NAMES order.')
    row = x.reshape(1, -1)
    anomaly = bool(_bundle['detector'].predict(row)[0] == -1)
    if not anomaly:
        # The classifier never saw normal windows, so don't ask it about one.
        return {'anomaly': False, 'fault': 'normal', 'confidence': 0.0}
    classifier = _bundle['classifier']
    probabilities = classifier.predict_proba(row)[0]
    best = int(np.argmax(probabilities))
    # Confidence is how sure the classifier is of the name, not of the anomaly.
    # A flagged window must be named, so low confidence means "faulty, unsure which".
    return {'anomaly': True,
            'fault': str(classifier.classes_[best]),
            'confidence': float(probabilities[best])}