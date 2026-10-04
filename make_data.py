"""Build the vibration_data.csv that train_model.py reads.

    python3 make_data.py          # then: python3 train_model.py

Windows come from stream.py, so the model trains on the same signal the
demo runs on. The detector trains on normal rows only; the classifier
trains on the labelled fault rows. The test split holds normal plus both
faults at three severities, so evaluation.json shows how they do on weak
early faults as well as obvious ones.
Separate seeds per block keep train and test noise independent.
"""

from itertools import islice
from pathlib import Path
import argparse

import pandas as pd

from stream import WINDOW_SIZE, stream

TRAIN_NORMAL_WINDOWS = 400
TRAIN_WINDOWS_PER_BLOCK = 150
TEST_WINDOWS_PER_BLOCK = 200
# The classifier sees weak faults in training so it can still name them.
TRAIN_SEVERITIES = (1.0, 0.6, 0.3, 0.1)
TEST_SEVERITIES = (1.0, 0.5, 0.2)
SAMPLE_COLUMNS = [f"s{i}" for i in range(WINDOW_SIZE)]


def _block(split, label, severity, count, seed):
    """Collect count windows of one label at one severity."""
    # fault_at=0 makes every window faulty; normal blocks never switch on.
    windows = stream(fault=label, fault_at=0.0, severity=severity, seed=seed)
    rows = []
    for window in islice(windows, count):
        row = {"split": split, "label": label, "severity": severity}
        row.update(zip(SAMPLE_COLUMNS, window["signal"]))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "vibration_data.csv")
    args = parser.parse_args()

    rows = _block("train", "normal", 0.0, TRAIN_NORMAL_WINDOWS, seed=1)
    rows += _block("test", "normal", 0.0, TEST_WINDOWS_PER_BLOCK, seed=2)
    seed = 10
    for severity in TRAIN_SEVERITIES:
        for label in ("imbalance", "bearing"):
            rows += _block("train", label, severity, TRAIN_WINDOWS_PER_BLOCK, seed)
            seed += 1
    for severity in TEST_SEVERITIES:
        for label in ("imbalance", "bearing"):
            rows += _block("test", label, severity, TEST_WINDOWS_PER_BLOCK, seed)
            seed += 1

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} windows to {args.out.name}")
    print("Next: python3 train_model.py")


if __name__ == "__main__":
    main()
