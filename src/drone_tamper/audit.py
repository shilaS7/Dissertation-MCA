"""Per-profile data audit (Step 1)."""

import numpy as np
import pandas as pd


def audit(df: pd.DataFrame, profile: str) -> dict:
    info = {"profile": profile, "rows": int(len(df))}

    lab = df["label"].value_counts().sort_index()
    info["label_counts"] = {int(k): int(v) for k, v in lab.items()}
    info["anomaly_rate"] = round(float(df["label"].mean()), 5)
    info["cases"] = int(df["case_id"].nunique())

    case_label = df.groupby("case_id")["label"].max()
    info["case_label_counts"] = {
        int(k): int(v) for k, v in case_label.value_counts().sort_index().items()
    }

    tt = df.groupby(["case_id", "tamper_type"]).size().reset_index()
    per_case = tt.groupby("tamper_type")["case_id"].nunique()
    info["tamper_type_cases"] = {k: int(v) for k, v in per_case.sort_index().items()}

    miss = df.isna().sum()
    info["missing"] = {k: int(v) for k, v in miss[miss > 0].items()}

    ts = df["timestamp"]
    info["ts_min"] = str(ts.min())
    info["ts_max"] = str(ts.max())
    info["epoch_rows"] = int((ts.dt.year <= 1971).sum())

    s = df.sort_values(["case_id", "row_idx"])
    delta = s["timestamp"].diff().dt.total_seconds()
    new_case = s["case_id"] != s["case_id"].shift()
    delta[new_case] = np.nan
    info["ts_decrease_frac"] = round(float((delta < 0).mean()), 6)
    info["ts_max_gap_s"] = round(float(delta.max()), 3)
    info["ts_gaps_gt_60s"] = int((delta > 60).sum())

    info["exact_duplicate_rows"] = int(df.duplicated().sum())
    sub = df.drop(columns=["case_id", "row_idx", "original_row_idx"])
    info["duplicate_rows_ignoring_idx"] = int(sub.duplicated().sum())
    return info
