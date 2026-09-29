"""Generate a ground-track figure for a single flight log, colored by the same-flight
baseline detector's flagged/normal classification (see drone_tamper/baseline.py).

Usage:
    uv run python scripts/make_flight_figure.py data/logs/some_flight.csv
    uv run python scripts/make_flight_figure.py data/logs/some_flight.csv --output my_chart.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from drone_tamper import adapt, features
from drone_tamper.baseline import CONSISTENCY_FEATURES, score_baseline
from drone_tamper.config import ROOT

NORMAL_COLOR = "#176b87"
FLAGGED_COLOR = "#d88732"
PATH_COLOR = "#999999"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_csv", type=Path, help="path to the raw flight-log CSV to plot")
    parser.add_argument("--threshold", type=float, default=4.0, help="robust z-score flag threshold")
    parser.add_argument("--output", type=Path, default=None, help="output PNG path")
    args = parser.parse_args()

    raw = pd.read_csv(args.log_csv)
    adapted = adapt.adapt_generic_log(raw)
    df, feats = features.prepare(adapted)
    score, flagged, worst = score_baseline(feats, CONSISTENCY_FEATURES, args.threshold, phase=df["phase"])
    df = df.assign(flagged=flagged.to_numpy(), score=score.to_numpy())

    def renderable(s: str) -> bool:
        try:
            s.encode("latin-1")
            return True
        except UnicodeEncodeError:
            return False  # matplotlib's default font can't render this (e.g. Hangul, CJK)

    mission = None
    for col in ("MissionName", "FlightLogName"):
        if col in raw.columns and raw[col].notna().any():
            candidate = str(raw[col].dropna().iloc[0]).strip()
            if candidate and renderable(candidate):
                mission = candidate
                break
    title = mission or args.log_csv.stem

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(df["longitude"], df["latitude"], color=PATH_COLOR, linewidth=0.6, alpha=0.5, zorder=1)
    normal = df[~df["flagged"]]
    flagged_pts = df[df["flagged"]]
    ax.scatter(normal["longitude"], normal["latitude"], s=10, color=NORMAL_COLOR,
               label=f"Normal ({len(normal)})", zorder=2)
    ax.scatter(flagged_pts["longitude"], flagged_pts["latitude"], s=16, color=FLAGGED_COLOR,
               label=f"Flagged ({len(flagged_pts)})", zorder=3)
    ax.set_title(f"{title}\n{len(df)} rows, {flagged.mean():.1%} flagged (same-flight baseline)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="datalim")
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    # "baseline" in the name: this figure comes from the same-flight baseline detector,
    # not from either trained model
    out_path = args.output or (ROOT / "results" / "scored" / f"{args.log_csv.stem}__baseline_ground_track.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
