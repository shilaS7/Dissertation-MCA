"""Adapt a real, unlabeled flight-log CSV (arbitrary DJI/ground-station export format)
into the internal row schema the feature engineering expects.

Only the columns needed for `features.engineer_features` are required:
case_id, row_idx, timestamp, latitude, longitude, altitude, speed, heading.
No label/tamper_type is needed or produced here -- this is inference-only.
"""

import pandas as pd

# Each internal column maps to a list of candidate source column names, tried in order.
COLUMN_CANDIDATES = {
    "latitude": ["Latitude", "latitude", "lat"],
    "longitude": ["Longitude", "longitude", "lon", "lng"],
    "altitude": ["Altitude(meters)", "Altitude", "altitude", "alt"],
    "speed": ["Speed(m/s)", "Speed", "speed"],
    "heading": ["Yaw", "Heading", "heading", "yaw"],
}

# Optional flight-phase column (e.g. TAKEOFF_AUTO / WAYLINE / LANDING_AUTO). Not required --
# falls back to a single constant phase when absent. Used by baseline.py to avoid comparing
# a row's behavior (e.g. altitude rate) against the whole flight when different phases are
# expected to look very different (a climb and a hover are not equally "normal" altitude rate).
PHASE_CANDIDATES = ["ModeCode", "FlightMode", "Mode", "flight_mode", "mode"]


def _find_column(df: pd.DataFrame, candidates: list[str], internal_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a column for '{internal_name}' -- tried {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def _find_timestamp(df: pd.DataFrame) -> pd.Series:
    if "Timestamp" in df.columns:
        # epoch milliseconds, as produced by common DJI/ground-station log exports
        return pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    if "Time(millis)" in df.columns:
        # relative milliseconds since log start -- no absolute time, use an arbitrary epoch
        return pd.to_datetime(df["Time(millis)"], unit="ms", utc=True)
    raise ValueError(
        "Could not find a timestamp column -- tried 'Timestamp' (epoch ms) and "
        f"'Time(millis)' (relative ms). Available columns: {list(df.columns)}"
    )


def adapt_generic_log(df: pd.DataFrame, case_id: int = 0) -> pd.DataFrame:
    """Map a raw flight-log export to the internal schema for one single flight."""
    out = pd.DataFrame(index=df.index)
    out["case_id"] = case_id
    out["row_idx"] = range(len(df))
    out["timestamp"] = _find_timestamp(df)
    for internal_name, candidates in COLUMN_CANDIDATES.items():
        col = _find_column(df, candidates, internal_name)
        out[internal_name] = pd.to_numeric(df[col], errors="coerce")

    phase_col = next((c for c in PHASE_CANDIDATES if c in df.columns), None)
    out["phase"] = df[phase_col].ffill().bfill() if phase_col else "flight"
    return out
