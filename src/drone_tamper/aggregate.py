"""Case-level aggregation and per-tamper-type breakdowns (Steps 3c, 4, 4b)."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import TAMPER_TYPES


def case_agg(case_id: np.ndarray, label: np.ndarray, score_h: np.ndarray, score_s: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"case_id": case_id, "label": label, "sh": score_h, "ss": score_s})
    return out.groupby("case_id", sort=False).agg(
        label=("label", "max"), sh=("sh", "max"), ss=("ss", "max"),
    ).reset_index()


def per_type_row(
    tamper_type: np.ndarray, score_h: np.ndarray, score_s: np.ndarray,
    thr_h: float, thr_s: float, dataset: str,
) -> pd.DataFrame:
    r = pd.DataFrame({"tamper_type": tamper_type, "sh": score_h, "ss": score_s})
    pred_h = r["sh"] >= thr_h
    pred_s = r["ss"] >= thr_s

    rows = []
    for t in [*TAMPER_TYPES, "normal"]:
        pos = r["tamper_type"] == t
        if t != "normal":
            sel = r["tamper_type"].isin([t, "normal"]).to_numpy()
            y = pos[sel].to_numpy()
            ph, ps = pred_h[sel].to_numpy(), pred_s[sel].to_numpy()
            pa_h = average_precision_score(y, r["sh"][sel].to_numpy())
            pa_s = average_precision_score(y, r["ss"][sel].to_numpy())
            tp_h = int((ph & y).sum()); fp_h = int((ph & ~y).sum())
            tp_s = int((ps & y).sum()); fp_s = int((ps & ~y).sum())
            prec_h = tp_h / max(tp_h + fp_h, 1); prec_s = tp_s / max(tp_s + fp_s, 1)
            rec_h = tp_h / max(int(y.sum()), 1); rec_s = tp_s / max(int(y.sum()), 1)
            f1_h = 2 * prec_h * rec_h / max(prec_h + rec_h, 1e-12)
            f1_s = 2 * prec_s * rec_s / max(prec_s + rec_s, 1e-12)
            flag_h, flag_s, pa_h, pa_s = rec_h, rec_s, round(pa_h, 4), round(pa_s, 4)
            f1_h, f1_s = round(f1_h, 4), round(f1_s, 4)
        else:
            n = int(pos.sum())
            flag_h = round(float((pred_h & pos).sum()) / n, 4)
            flag_s = round(float((pred_s & pos).sum()) / n, 4)
            pa_h, pa_s, f1_h, f1_s = np.nan, np.nan, np.nan, np.nan
        rows.append({
            "dataset": dataset, "tamper_type": t, "n_rows": int(pos.sum()),
            "gb_flag_rate": flag_h, "svm_flag_rate": flag_s,
            "gb_pr_auc": pa_h, "svm_pr_auc": pa_s, "gb_f1": f1_h, "svm_f1": f1_s,
        })
    return pd.DataFrame(rows)


def cases_of(case_id: np.ndarray, case_name: np.ndarray, score_h: np.ndarray, score_s: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"case_id": case_id, "name": case_name, "sh": score_h, "ss": score_s})
    c = out.groupby("case_id", sort=False).agg(
        name=("name", "first"), sh=("sh", "max"), ss=("ss", "max"),
    ).reset_index()
    c["type"] = c["name"].str.split("_", n=2).str[2]
    return c


def per_type_case(c: pd.DataFrame, thr_h: float, thr_s: float, dataset: str) -> pd.DataFrame:
    pred_h = c["sh"] >= thr_h
    pred_s = c["ss"] >= thr_s
    rows = []
    for t in [*TAMPER_TYPES, "normal"]:
        sel = c["type"] == t
        n = int(sel.sum())
        rows.append({
            "dataset": dataset, "tamper_type": t, "n_cases": n,
            "gb_flag_rate": round(float((pred_h & sel).sum()) / n, 4) if n else np.nan,
            "svm_flag_rate": round(float((pred_s & sel).sum()) / n, 4) if n else np.nan,
        })
    return pd.DataFrame(rows)
