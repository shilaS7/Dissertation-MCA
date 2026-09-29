"""Consolidate the real-flight-log external check into one comparable table.

The real logs are UNLABELLED, so no accuracy metric (PR-AUC, F1, ...) can be computed.
What can be compared is how much of each flight each detector flags -- and, in particular,
how much it flags while the aircraft is stationary on the ground, where tampering is
implausible and a high flag rate therefore indicates the detector is mis-firing.

Usage:
    uv run python scripts/summarise_real_logs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from drone_tamper.config import ROOT

SCORED = ROOT / "results" / "scored"
STATIONARY_MAX_SPEED = 0.5  # m/s -- essentially not moving

DETECTORS = {
    "gb_rep00": "Gradient Boosting (main model)",
    "gb_transfer_rep00": "Gradient Boosting (transfer variant)",
}


def rates(df: pd.DataFrame) -> dict:
    still = df["speed"].abs() <= STATIONARY_MAX_SPEED
    return {
        "rows": len(df),
        "flagged": int(df["flagged"].sum()),
        "flag_rate": round(float(df["flagged"].mean()), 4),
        "stationary_rows": int(still.sum()),
        "stationary_flag_rate": round(float(df.loc[still, "flagged"].mean()), 4) if still.any() else None,
        "moving_flag_rate": round(float(df.loc[~still, "flagged"].mean()), 4) if (~still).any() else None,
    }


def main() -> None:
    rows = []
    for model_stem, label in DETECTORS.items():
        for path in sorted(SCORED.glob(f"*__{model_stem}_scored.csv")):
            flight = path.name.replace(f"__{model_stem}_scored.csv", "")
            rows.append({"detector": label, "flight": flight, **rates(pd.read_csv(path))})

    pooled = SCORED / "pooled_baseline_scored.csv"
    if pooled.exists():
        df = pd.read_csv(pooled)
        for flight, g in df.groupby("source"):
            rows.append({"detector": "Pooled cross-flight baseline", "flight": flight, **rates(g)})

    if not rows:
        raise SystemExit(f"No scored CSVs found in {SCORED}. Run score_log.py / score_logs_baseline.py first.")

    out = pd.DataFrame(rows).sort_values(["detector", "flight"]).reset_index(drop=True)
    out_path = ROOT / "results" / "real_log_summary.csv"
    out.to_csv(out_path, index=False)

    show = out.copy()
    for c in ["flag_rate", "stationary_flag_rate", "moving_flag_rate"]:
        show[c] = (show[c] * 100).map(lambda v: "-" if pd.isna(v) else f"{v:.1f}%")
    print(show.to_string(index=False))
    print(f"\nwrote {out_path}")
    print(
        "\nNOTE: these logs carry no tamper labels, so none of these rates is an accuracy "
        "figure. A high stationary flag rate means the detector flags the aircraft sitting "
        "still on the ground, which is evidence the detector is mis-firing rather than "
        "evidence of tampering."
    )


if __name__ == "__main__":
    main()
