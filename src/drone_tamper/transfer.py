"""Transfer variant: make the synthetic training data resemble real flight-log exports.

The main model is trained on 10 Hz synthetic telemetry from one location and uses absolute
position as features, so every row of a real 0.5 Hz log from elsewhere looks out of
distribution. This variant (a) thins each synthetic case to the target sampling interval and
(b) drops absolute features that encode *where* and *on which route* the flight happened,
keeping only relative / kinematic-consistency features.
"""

import pandas as pd

# Absolute values that tie the model to the one synthetic source flight's place and route.
LOCATION_FEATURES = ["latitude", "longitude", "altitude", "heading"]

TARGET_DT_S = 2.0  # typical DJI / ground-station export interval


def median_dt_s(df: pd.DataFrame) -> float:
    dt = df.sort_values(["case_id", "row_idx"]).groupby("case_id")["timestamp"].diff().dt.total_seconds()
    return float(dt.median())


def resample_step(df: pd.DataFrame, target_dt_s: float = TARGET_DT_S) -> int:
    """Keep every `step`-th row so the median interval approximates `target_dt_s`."""
    return max(1, round(target_dt_s / median_dt_s(df)))


def thin(df: pd.DataFrame, step: int) -> pd.DataFrame:
    """Keep every `step`-th row within each case and renumber row_idx contiguously.

    Row labels are kept as-is: a tampered row that survives thinning stays tampered.
    """
    df = df.sort_values(["case_id", "row_idx"]).reset_index(drop=True)
    pos = df.groupby("case_id").cumcount()
    out = df[pos % step == 0].copy()
    out["row_idx"] = out.groupby("case_id").cumcount().astype("int32")
    return out.reset_index(drop=True)


def select_features(feats: pd.DataFrame) -> pd.DataFrame:
    return feats.drop(columns=[c for c in LOCATION_FEATURES if c in feats.columns])
