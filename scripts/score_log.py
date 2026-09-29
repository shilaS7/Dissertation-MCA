"""Score a real, unlabeled flight-log CSV for possible tampering using a saved model.

Usage:
    uv run python scripts/score_log.py data/logs/2026-01-13_11-49-05.csv

This is inference only: no label/tamper_type is required or produced. It reuses the
Gradient Boosting model trained by `run_pipeline.py`'s main stage (saved to
results/models/gb_rep00.joblib) and the same feature engineering as the rest of the
pipeline.

CAVEAT: the saved model was trained on tampered variants of ONE synthetic source
flight. It has been validated at detecting different tampering magnitudes of that one
flight, not at generalizing to a genuinely different real flight, drone, or sensor
setup. Treat any result on a new, real log as exploratory, not proven.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd

from drone_tamper import adapt, features
from drone_tamper.config import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_csv", type=Path, help="path to the raw flight-log CSV to score")
    parser.add_argument("--model", type=Path, default=ROOT / "results/models/gb_rep00.joblib")
    parser.add_argument("--out", type=Path, default=None,
                        help="output CSV path (default: results/scored/<log>__<model>_scored.csv)")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"No saved model at {args.model}. Run `uv run python scripts/run_pipeline.py --only main` "
            "first -- it trains and saves the model this script needs."
        )

    bundle = joblib.load(args.model)
    model, threshold, feature_names = bundle["model"], bundle["threshold"], bundle["feature_names"]

    raw = pd.read_csv(args.log_csv)
    adapted = adapt.adapt_generic_log(raw)
    df, feats = features.prepare(adapted)
    feats = feats[feature_names]

    scores = model.predict_proba(feats)[:, 1]
    flagged = scores >= threshold

    out = pd.DataFrame({
        "row_idx": df["row_idx"],
        "timestamp": df["timestamp"],
        "latitude": df["latitude"],
        "longitude": df["longitude"],
        "altitude": df["altitude"],
        "speed": df["speed"],
        "heading": df["heading"],
        "anomaly_score": scores,
        "flagged": flagged,
    })

    out_dir = ROOT / "results" / "scored"
    out_dir.mkdir(parents=True, exist_ok=True)
    # the model name is part of the filename so scoring the same log with a different
    # model never silently overwrites an earlier run's per-row scores
    out_path = args.out or (out_dir / f"{args.log_csv.stem}__{args.model.stem}_scored.csv")
    out.to_csv(out_path, index=False)

    n = len(out)
    n_flagged = int(flagged.sum())
    print(f"model: {args.model}  (threshold={threshold:.4f})")
    print(f"rows: {n}   flagged: {n_flagged} ({n_flagged / n:.1%})")
    print(f"wrote per-row scores -> {out_path}")

    if n_flagged:
        print("\nmost suspicious rows:")
        print(out.sort_values("anomaly_score", ascending=False).head(10).to_string(index=False))

    print(
        "\nCAVEAT: this model was trained on ONE synthetic source flight and its tampered "
        "variants. It has not been validated on real, independent flights -- treat this "
        "score as exploratory, not a proven verdict."
    )


if __name__ == "__main__":
    main()
