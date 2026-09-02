"""Same-flight baseline anomaly detector.

Unlike the supervised Gradient Boosting model (trained on a different, synthetic
dataset and shown not to transfer to real logs -- see scripts/score_log.py), this
detector needs no training data at all: it scores each row against the statistics of
the SAME flight, using the engineered kinematic/temporal consistency features from
`features.py` (residuals and deltas, not raw absolute position/speed/altitude).

Method: robust z-scores. For each consistency feature, compare every row to that
feature's own median and MAD (median absolute deviation) across the flight -- robust
to a handful of true outliers, unlike mean/std. A row is flagged if any one feature's
robust z-score exceeds `threshold` (default 4.0, i.e. a strong departure from this
flight's own typical behavior).

Limitation: this can only catch anomalies that stand out WITHIN this flight. Tampering
applied uniformly across the whole flight (e.g. a constant altitude offset) defines its
own "normal" and will not be flagged -- there is no external reference to compare
against with a single log.
"""

import numpy as np
import pandas as pd

CONSISTENCY_FEATURES = [
    "dt_s",
    "speed_resid_mps",
    "accel_mps2",
    "d_heading_deg",
    "roll_dev_speed",
    "roll_dev_latitude",
    "roll_dev_longitude",
]
# Deliberately excluded: alt_rate_mps, roll_dev_altitude, roll_std_altitude. These measure a
# MAGNITUDE (how fast is altitude changing) that legitimately varies within one phase -- e.g. a
# mission's initial climb-to-altitude leg is still labeled the same phase as level cruise in
# most ground-station logs, so it isn't a phase-invariant "is this internally consistent"
# signal the way speed_resid_mps (recorded speed vs. GPS-derived speed should agree regardless
# of how fast you're going) is.

MAD_SCALE = 1.4826  # scales MAD to be comparable to a normal std for Gaussian data
MIN_GROUP_SIZE = 20  # phases with fewer rows can't reliably characterize their own "normal"

# Altitude rate needs a different treatment than the z-score features above: comparing it to
# a phase label (e.g. "WAYLINE") doesn't work because a mission's climb-to-altitude leg often
# shares the same phase label as level cruise, so a same-flight z-score flags every real climb
# and landing as an anomaly. Instead of judging "how far from the median" (which assumes climbs
# are rare), judge "is this rate physically plausible for what THIS aircraft actually did" --
# a bound set from the flight's own fastest genuine climb/descent, with headroom.
ALTITUDE_SAFETY_FACTOR = 2.0  # plausibility bound = this many times the flight's own p99.5 rate
ALTITUDE_MIN_FLOOR_MPS = 3.0  # floor for very calm flights, so the bound isn't near-zero


def _center_scale(x: pd.Series) -> tuple[float, float]:
    """Robust (median, scale) for one column, falling back to std when MAD collapses to ~0
    (common for rate-like features that sit at exactly 0 for most of a flight, e.g. altitude
    rate during level flight/hover) -- a tiny epsilon floor there would blow up any nonzero
    reading into a meaningless score. scale is NaN if the column has no variation at all."""
    median = x.median()
    mad = (x - median).abs().median()
    scale = mad * MAD_SCALE
    if scale < 1e-6:
        std = x.std()
        scale = std if std > 1e-6 else np.nan
    return median, scale


def robust_zscores(feats: pd.DataFrame, columns: list[str], phase: pd.Series | None = None) -> pd.DataFrame:
    """Robust z-score per column. If `phase` is given, each row is compared only to rows in the
    SAME phase (e.g. TAKEOFF_AUTO / WAYLINE / LANDING_AUTO) -- a climb's altitude rate should
    not be judged against a hover's. Phases smaller than MIN_GROUP_SIZE fall back to the whole
    flight's stats (too few samples to characterize their own "normal"). Without a phase
    column, every row shares one group."""
    groups = phase if phase is not None else pd.Series("flight", index=feats.index)
    z = pd.DataFrame(index=feats.index)
    for col in columns:
        x = feats[col]
        g_median, g_scale = _center_scale(x)
        col_z = pd.Series(index=x.index, dtype=float)
        for _, idx in groups.groupby(groups).groups.items():
            if len(idx) >= MIN_GROUP_SIZE:
                median, scale = _center_scale(x.loc[idx])
            else:
                median, scale = g_median, g_scale
            col_z.loc[idx] = (x.loc[idx] - median) / scale if scale == scale else 0.0
        z[col] = col_z
    return z


def altitude_plausibility_zscore(alt_rate_mps: pd.Series, threshold: float = 4.0) -> pd.Series:
    """A row scores `threshold` exactly at the plausibility bound and scales linearly past it,
    so it combines directly with the z-score features via a simple max() -- both use the same
    "flagged if > threshold" convention."""
    bound = max(ALTITUDE_MIN_FLOOR_MPS, ALTITUDE_SAFETY_FACTOR * alt_rate_mps.abs().quantile(0.995))
    return threshold * alt_rate_mps.abs() / bound


def score_baseline(
    feats: pd.DataFrame,
    columns: list[str] = CONSISTENCY_FEATURES,
    threshold: float = 4.0,
    phase: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (score, flagged, worst_feature) -- score is the max |z| across columns per row."""
    z = robust_zscores(feats, columns, phase)
    # NaN means "no predecessor row to compare against" (e.g. the first row of the flight),
    # not evidence of an anomaly.
    abs_z = z.abs().fillna(0.0)
    abs_z["alt_rate_mps"] = altitude_plausibility_zscore(feats["alt_rate_mps"], threshold).fillna(0.0)
    score = abs_z.max(axis=1)
    worst_feature = abs_z.idxmax(axis=1)
    flagged = score > threshold
    return score, flagged, worst_feature
