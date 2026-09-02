"""Score MULTIPLE real, unlabeled flight logs against a POOLED cross-flight baseline
(no training data, no pretrained model) -- fixes the single-flight version's blind spot:
with only one flight, a legitimate recurring maneuver (e.g. turns on a grid survey) looks
like a rare, "anomalous" minority. Pooling several flights' same-phase rows together gives
a broader, more honest sense of what's actually typical.

Usage:
    uv run python scripts/score_logs_baseline.py data/logs/*.csv
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
    parser.add_argument("log_csvs", type=Path, nargs="+", help="paths to the raw flight-log CSVs to pool and score")
    parser.add_argument("--threshold", type=float, default=4.0, help="robust z-score flag threshold")
    args = parser.parse_args()

    adapted_frames = []
    for case_id, path in enumerate(args.log_csvs):
        raw = pd.read_csv(path)
        adapted = adapt.adapt_generic_log(raw, case_id=case_id)
        adapted["source"] = path.stem
        adapted_frames.append(adapted)
    combined = pd.concat(adapted_frames, ignore_index=True)

    # engineer_features groups by case_id internally, so deltas never leak across flights
    df, feats = features.prepare(combined)

    print("flights:", {p.stem: len(a) for p, a in zip(args.log_csvs, adapted_frames)})
    print("pooled phase counts:", df["phase"].value_counts().to_dict())

    score, flagged, worst_feature = score_baseline(
        feats, CONSISTENCY_FEATURES, args.threshold, phase=df["phase"]
    )

    out = pd.DataFrame({
        "source": df["source"],
        "row_idx": df["row_idx"],
        "timestamp": df["timestamp"],
        "latitude": df["latitude"],
        "longitude": df["longitude"],
        "altitude": df["altitude"],
        "speed": df["speed"],
        "heading": df["heading"],
        "phase": df["phase"],
        "anomaly_score": score,
        "worst_feature": worst_feature,
        "flagged": flagged,
    })

    out_dir = ROOT / "results" / "scored"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pooled_baseline_scored.csv"
    out.to_csv(out_path, index=False)

    print(f"\nmethod: pooled cross-flight robust z-score (threshold={args.threshold})")
    print(f"wrote per-row scores -> {out_path}\n")
    print("per-flight breakdown:")
    summary = out.groupby("source")["flagged"].agg(["size", "sum"])
    summary["pct"] = (summary["sum"] / summary["size"] * 100).round(1)
    print(summary.rename(columns={"size": "rows", "sum": "flagged"}).to_string())

    print("\nmost suspicious rows overall:")
    print(out.sort_values("anomaly_score", ascending=False).head(10).to_string(index=False))

    print(
        "\nCAVEAT: pooling flights gives a broader sense of 'typical' than one flight alone, "
        "but still only detects departures from what's common ACROSS these flights. Tampering "
        "applied consistently to every flight would define its own 'normal' and go undetected."
    )


if __name__ == "__main__":
    main()
