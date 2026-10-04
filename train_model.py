"""Train on the training rows, then evaluate the untouched test split.

Two stages: an Isolation Forest sees normal windows only and answers
"is this abnormal", then a Random Forest trained on the labelled fault
rows answers "which fault". The classifier never sees normal windows, so
predict() only consults it once the detector has flagged a window.
"""
from pathlib import Path
import argparse
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from features import extract, FEATURE_NAMES, SAMPLE_RATE, WINDOW_SAMPLES

ROOT = Path(__file__).resolve().parent
FAULT_LABELS = ('imbalance', 'bearing')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "vibration_data.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    sample_cols = [f"s{i}" for i in range(WINDOW_SAMPLES)]
    required = ["split", "label", "severity"] + sample_cols
    if any(c not in df.columns for c in required):
        raise ValueError("CSV must contain split,label,severity,s0,...,s1023")
    if not df['label'].isin(['normal', 'imbalance', 'bearing']).all():
        raise ValueError("Unexpected motor label")
    normal_train = df[(df['split'] == 'train') & (df['label'] == 'normal')]
    fault_train = df[(df['split'] == 'train') & (df['label'].isin(FAULT_LABELS))]
    test = df[df['split'] == 'test']
    if len(normal_train) < 100 or test.empty:
        raise ValueError("Need at least 100 normal training windows and a test split")
    if set(fault_train['label']) != set(FAULT_LABELS):
        raise ValueError("Need training rows for both faults. Rerun python3 make_data.py")
    def feature_matrix(rows):
        result = np.asarray([extract(s) for s in rows[sample_cols].to_numpy(dtype=float)])
        if result.shape != (len(rows), len(FEATURE_NAMES)) or not np.isfinite(result).all():
            raise ValueError("Invalid feature output")
        return result
    x_train = feature_matrix(normal_train)
    x_fault_train = feature_matrix(fault_train)
    x_test = feature_matrix(test)
    detector = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=1)
    detector.fit(x_train)
    classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
    classifier.fit(x_fault_train, fault_train['label'].to_numpy())
    # Both models are fixed here. Do not tune either on the test split.
    flagged = detector.predict(x_test) == -1
    named = classifier.predict(x_test)
    report = {'normal_training_windows': len(normal_train),
              'fault_training_windows': len(fault_train),
              'test_windows': len(test), 'by_label': {}}
    for label in ['normal', 'imbalance', 'bearing']:
        mask = test['label'].to_numpy() == label
        total = int(mask.sum())
        count = int(flagged[mask].sum())
        entry = {'windows': total, 'flagged': count, 'flagged_rate': count / total if total else None}
        if label in FAULT_LABELS and count:
            # Naming is only judged on windows the detector actually caught,
            # because those are the only ones predict() names in the demo.
            caught = mask & flagged
            entry['named_correctly'] = int((named[caught] == label).sum())
            entry['naming_accuracy'] = float((named[caught] == label).mean())
        report['by_label'][label] = entry
    bundle = {'detector': detector, 'classifier': classifier,
              'fault_labels': list(FAULT_LABELS), 'feature_names': list(FEATURE_NAMES),
              'sample_rate': SAMPLE_RATE, 'window_samples': WINDOW_SAMPLES,
              'features_sha256': hashlib.sha256((ROOT / 'features.py').read_bytes()).hexdigest(),
              'sklearn_version': sklearn.__version__}
    joblib.dump(bundle, ROOT / 'anomaly_model.joblib')
    (ROOT / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Detector: {len(normal_train)} normal windows. '
          f'Classifier: {len(fault_train)} fault windows. {len(FEATURE_NAMES)} features each.')
    for label, r in report['by_label'].items():
        meaning = 'false alarms' if label == 'normal' else 'detected'
        line = f"{label}: {r['flagged']}/{r['windows']} {meaning}"
        if 'naming_accuracy' in r:
            line += f", named right {r['named_correctly']}/{r['flagged']}"
        print(line)
    print('Saved anomaly_model.joblib and evaluation.json')

if __name__ == '__main__':
    main()
