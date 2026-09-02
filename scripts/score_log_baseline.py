"""Score a real, unlabeled flight-log CSV against ITS OWN behavior (no training data,
no pretrained model) -- an alternative to scripts/score_log.py for when the supervised
model doesn't transfer to real logs (see drone_tamper/baseline.py for the method).

Usage:
    uv run python scripts/score_log_baseline.py data/logs/2026-01-13_11-49-05.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from drone_tamper import adapt, features
from drone_tamper.baseline import CONSISTENCY_FEATURES, score_baseline
from drone_tamper.config import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_csv", type=Path, help="path to the raw flight-log CSV to score")
    parser.add_argument("--threshold", type=float, default=4.0, help="robust z-score flag threshold")
    args = parser.parse_args()

    raw = pd.read_csv(args.log_csv)
    adapted = adapt.adapt_generic_log(raw)
    df, feats = features.prepare(adapted)

    print("flight phases:", df["phase"].value_counts().to_dict())
    score, flagged, worst_feature = score_baseline(
        feats, CONSISTENCY_FEATURES, args.threshold, phase=df["phase"]
    )

    out = pd.DataFrame({
        "row_idx": df["row_idx"],
        "timestamp": df["timestamp"],
        "latitude": df["latitude"],
        "longitude": df["longitude"],
        "altitude": df["altitude"],
        "speed": df["speed"],
        "heading": df["heading"],
        "anomaly_score": score,
        "worst_feature": worst_feature,
        "flagged": flagged,
    })

    out_dir = ROOT / "results" / "scored"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.log_csv.stem}_baseline_scored.csv"
    out.to_csv(out_path, index=False)

    n = len(out)
    n_flagged = int(flagged.sum())
    print(f"method: same-flight robust z-score (threshold={args.threshold})")
    print(f"rows: {n}   flagged: {n_flagged} ({n_flagged / n:.1%})")
    print(f"wrote per-row scores -> {out_path}")

    if n_flagged:
        print("\nmost suspicious rows:")
        print(out.sort_values("anomaly_score", ascending=False).head(10).to_string(index=False))

    print(
        "\nCAVEAT: this method only detects departures from THIS flight's own typical "
        "behavior. Tampering applied uniformly across the whole flight (e.g. a constant "
        "offset) defines its own 'normal' and will not be flagged -- there is no external "
        "reference with a single log."
    )


if __name__ == "__main__":
    main()
