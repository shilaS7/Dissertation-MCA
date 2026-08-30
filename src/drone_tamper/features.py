"""Temporal / kinematic feature engineering (Step 2), computed within each case."""

import numpy as np
import pandas as pd


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must be sorted by [case_id, row_idx] with a fresh RangeIndex."""
    g = df.groupby("case_id", sort=False)
    prev_ts = g["timestamp"].shift()
    prev_lat = g["latitude"].shift()
    prev_lon = g["longitude"].shift()
    prev_alt = g["altitude"].shift()
    prev_speed = g["speed"].shift()
    prev_heading = g["heading"].shift()

    dt_s = (df["timestamp"] - prev_ts).dt.total_seconds()
    dist_m = haversine_m(prev_lat, prev_lon, df["latitude"], df["longitude"])
    speed_gps = dist_m / dt_s
    speed_resid = df["speed"] - speed_gps
    d_heading = ((df["heading"] - prev_heading + 540.0) % 360.0) - 180.0
    row_max = g["row_idx"].transform("max").astype("float32")

    feats = pd.DataFrame(index=df.index)
    feats["dt_s"] = dt_s.clip(upper=3600.0)
    feats["ts_gap_flag"] = (dt_s > 60.0).astype("int8")
    feats["d_lat_deg"] = df["latitude"] - prev_lat
    feats["d_lon_deg"] = df["longitude"] - prev_lon
    feats["dist_m"] = dist_m
    feats["d_alt_m"] = df["altitude"] - prev_alt
    feats["speed_gps_mps"] = speed_gps
    feats["speed_resid_mps"] = speed_resid
    feats["accel_mps2"] = (df["speed"] - prev_speed) / dt_s
    feats["alt_rate_mps"] = feats["d_alt_m"] / dt_s
    feats["d_heading_deg"] = d_heading
    feats["row_frac"] = df["row_idx"].astype("float32") / row_max
    for col in ["altitude", "speed", "latitude", "longitude"]:
        rm = g[col].rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
        rs = g[col].rolling(10, min_periods=1).std().reset_index(level=0, drop=True)
        feats[f"roll_dev_{col}"] = df[col] - rm
        feats[f"roll_std_{col}"] = rs
    feats["altitude"] = df["altitude"]
    feats["speed"] = df["speed"]
    feats["heading"] = df["heading"]
    feats["latitude"] = df["latitude"]
    feats["longitude"] = df["longitude"]

    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort a loaded profile by case and engineer its feature matrix."""
    df = df.sort_values(["case_id", "row_idx"]).reset_index(drop=True)
    feats = engineer_features(df)
    feats = feats.astype({c: "float32" for c in feats.columns if c != "ts_gap_flag"})
    return df, feats
