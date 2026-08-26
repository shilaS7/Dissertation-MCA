"""Build notebooks/01_gradient_boosting_vs_svm.ipynb (Step 1: data audit).

Run: uv run python scripts/make_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)

SRC = r'''
# --- imports ---
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

# Resolve project root no matter where the kernel was launched (jupyter / nbconvert)
def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start

ROOT = _find_root(Path.cwd())
os.chdir(ROOT)
(ROOT / "results").mkdir(exist_ok=True)

print("python", sys.version.split()[0])
print("pandas", pd.__version__)
print("numpy", np.__version__)
print("scikit-learn", sklearn.__version__)
'''

AUDIT_CODE = r'''
# --- Step 1: data audit for replicate 0 (rep_00) ---
# We deliberately work with ONE replicate first so that repeated synthetic
# variants do not dominate, and the notebook stays lightweight.

PROFILES = ["balanced", "strong", "subtle"]
RAW = ROOT / "data/raw/rep_00"

DTYPES = {
    "case_id": "int16",
    "row_idx": "int32",
    "label": "int8",
    "latitude": "float32",
    "longitude": "float32",
    "altitude": "float32",
    "speed": "float32",
    "heading": "float32",
    "original_row_idx": "int32",
}


def load(profile: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"{profile}.csv", dtype=DTYPES)
    # pandas 3.0 needs an explicit ISO8601 format for these 6-digit-microsecond strings
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", errors="coerce")
    return df


data = {p: load(p) for p in PROFILES}
for p, df in data.items():
    print(f"{p:9s} rows={len(df):>8,d}  cases={df['case_id'].nunique():>3}  "
          f"anomaly_rate={df['label'].mean():.4f}")
'''

AUDIT_DETAIL_CODE = r'''
# --- detailed audit ---
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


audit_summary = {p: audit(df, p) for p, df in data.items()}

print(json.dumps(audit_summary, indent=2))
with open(ROOT / "data/index/rep_00/audit_rep00.json", "w") as f:
    json.dump(audit_summary, f, indent=2)
'''

ARTIFACT_CODE = r'''
# --- key artifact: one sentinel 1970-01-01 row at the start of every case ---
epoch = data["balanced"][data["balanced"]["timestamp"].dt.year <= 1971]
print("sentinel rows:", len(epoch), "| all row_idx==0?", (epoch["row_idx"] == 0).all())
print("sentinel label distribution:", epoch["label"].value_counts().to_dict())

# label <-> tamper_type is clean and one-to-one:
print(data["balanced"].groupby("tamper_type")["label"].mean().round(3).to_string())
'''

FINDINGS_MD = r'''
## Step 1 findings (replicate 0)

- **No missing values** in any profile; no exact duplicate rows.
- **Anomaly rate is ~27–30%**, not the 14% suggested by `anom_rate_target` in the
  dataset metadata. The label distribution is imbalanced but not extreme.
- **Case-level:** each profile has 60 cases; ~92% of cases (55/60 balanced) contain at
  least one tampered segment. A case-level "contains any anomaly" label is therefore
  strongly imbalanced toward tampered.
- **Timestamp artifact:** every case begins with a sentinel row at `1970-01-01`. It is
  **not** always benign: in 8/60 balanced cases that first row is itself tampered
  (`label=1`). Sentinel rows must therefore be **kept** (do not drop). Feature
  engineering will give the first row of each case no temporal predecessor
  (`delta = NaN`) and clip the artificial ~54-year gap.
- **Timestamp drift:** ~1.5–1.8% of intra-case timestamp deltas are negative; these are
  tampering-related and must be captured by features, not silently dropped.
- **~0.5% of rows duplicate telemetry when ignoring row indices** — likely stationary
  drone records; do NOT auto-deduplicate.
- Every profile shares the same `case_id`/`case_name` space and the same source log, so
  strong/subtle evaluation measures generalization across tampering **magnitude**, not
  across new flights.
'''

FEAT_CODE = r'''
# --- temporal / kinematic feature engineering (within each case) ---
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def eng(df: pd.DataFrame) -> pd.DataFrame:
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


def prepare(profile: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = data[profile].sort_values(["case_id", "row_idx"]).reset_index(drop=True)
    feats = eng(df)
    feats = feats.astype({c: "float32" for c in feats.columns if c != "ts_gap_flag"})
    print(f"{profile:9s} feats={feats.shape[1]}  NaN_cols={int(feats.isna().any().sum())}")
    return df, feats


bal_s, feats_bal = prepare("balanced")
str_s, feats_str = prepare("strong")
sub_s, feats_sub = prepare("subtle")

FEATURE_NAMES = [c for c in feats_bal.columns]
print("features:", FEATURE_NAMES)
'''

SPLIT_CODE = r'''
# --- case-level 80/20 split on balanced, stratified by case tamper status ---
from sklearn.model_selection import train_test_split

case_label = (
    bal_s.groupby("case_id")["label"].max().rename("case_label").reset_index()
)
tr_cases, va_cases = train_test_split(
    case_label, test_size=0.2, stratify=case_label["case_label"],
    random_state=42,
)
tr_mask = bal_s["case_id"].isin(tr_cases["case_id"]).to_numpy()
va_mask = bal_s["case_id"].isin(va_cases["case_id"]).to_numpy()

print("train cases:", len(tr_cases), tr_cases["case_label"].value_counts().to_dict())
print("holdout cases:", len(va_cases), va_cases["case_label"].value_counts().to_dict())
print("train rows:", int(tr_mask.sum()), " holdout rows:", int(va_mask.sum()))
'''

MODEL_CODE = r'''
# --- Step 3: train Gradient Boosting vs LinearSVC on the SAME features/cases ---
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42
X_tr = feats_bal.loc[tr_mask]
y_tr = bal_s["label"].to_numpy()[tr_mask]

# Gradient Boosting (NaN-safe, no scaling needed)
hgb = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20,
    l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=15, random_state=SEED,
)
hgb.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))

# LinearSVC (needs imputation + scaling; class weights balance classes)
svm = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("svc", LinearSVC(class_weight="balanced", dual="auto", max_iter=3000, random_state=SEED)),
])
svm.fit(X_tr, y_tr)

print("HistGB fitted; n_iter used:", hgb.n_iter_)
print("LinearSVC fitted; n_iter reached:", svm.named_steps["svc"].n_iter_)
'''

EVAL_ROW_CODE = r'''
# --- Step 3b: row-level evaluation ---
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score, balanced_accuracy_score,
    matthews_corrcoef,
)

def row_scores(df, feats):
    s_hgb = hgb.predict_proba(feats)[:, 1]
    s_svm = svm.decision_function(feats)
    return df["label"].to_numpy(), s_hgb, s_svm

def tune_threshold(y, s):
    prec, rec, thr = precision_recall_curve(y, s)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[int(np.argmax(f1s[:-1]))])

def row_metrics(y, s, thr):
    yp = (s >= thr).astype(int)
    return {
        "n": int(len(y)), "pos": int(y.sum()), "roc_auc": round(roc_auc_score(y, s), 4),
        "pr_auc": round(average_precision_score(y, s), 4),
        "f1": round(f1_score(y, yp), 4), "prec": round(precision_score(y, yp), 4),
        "rec": round(recall_score(y, yp), 4),
        "bal_acc": round(balanced_accuracy_score(y, yp), 4),
        "mcc": round(matthews_corrcoef(y, yp), 4),
    }

# tune a threshold per model on the balanced holdout ONLY
results_row = []
for name, df, feats in [("balanced_holdout", bal_s, feats_bal), ("strong", str_s, feats_str), ("subtle", sub_s, feats_sub)]:
    y, sh, ss = row_scores(df, feats)
    if name == "balanced_holdout":
        thr_h, thr_s = tune_threshold(y[va_mask], sh[va_mask]), tune_threshold(y[va_mask], ss[va_mask])
        y, sh, ss = y[va_mask], sh[va_mask], ss[va_mask]
        print(f"thresholds -> hgb={thr_h:.4f}  svm={thr_s:.4f}")
    results_row.append({"dataset": name, "model": "GradientBoosting", **row_metrics(y, sh, thr_h)})
    results_row.append({"dataset": name, "model": "LinearSVC", **row_metrics(y, ss, thr_s)})

res_row = pd.DataFrame(results_row)
print(res_row.to_string(index=False))
res_row.to_csv(ROOT / "results/row_metrics_rep00.csv", index=False)
'''

EVAL_CASE_CODE = r'''
# --- Step 3c: case-level evaluation (aggregate row scores to each flight) ---
def case_agg(df, feats, score_h, score_s):
    out = pd.DataFrame({
        "case_id": df["case_id"].to_numpy(),
        "label": df["label"].to_numpy(),
        "sh": score_h, "ss": score_s,
    })
    return out.groupby("case_id", sort=False).agg(
        label=("label", "max"), sh=("sh", "max"), ss=("ss", "max"),
    ).reset_index()

# case scores for the balanced holdout cases
case_va = case_agg(bal_s, feats_bal, hgb.predict_proba(feats_bal)[:, 1], svm.decision_function(feats_bal))
case_va = case_va[case_va["case_id"].isin(va_cases["case_id"])]

thr_c_h = tune_threshold(case_va["label"], case_va["sh"])
thr_c_s = tune_threshold(case_va["label"], case_va["ss"])
print(f"case thresholds -> hgb={thr_c_h:.4f} svm={thr_c_s:.4f}")

results_case = []
for name, df, feats in [("balanced_holdout", bal_s, feats_bal), ("strong", str_s, feats_str), ("subtle", sub_s, feats_sub)]:
    cs = case_agg(df, feats, hgb.predict_proba(feats)[:, 1], svm.decision_function(feats))
    if name == "balanced_holdout":
        cs = cs[cs["case_id"].isin(va_cases["case_id"])]
    results_case.append({"dataset": name, "model": "GradientBoosting", **row_metrics(cs["label"], cs["sh"], thr_c_h)})
    results_case.append({"dataset": name, "model": "LinearSVC", **row_metrics(cs["label"], cs["ss"], thr_c_s)})

res_case = pd.DataFrame(results_case)
print(res_case.to_string(index=False))
res_case.to_csv(ROOT / "results/case_metrics_rep00.csv", index=False)
'''

PER_TYPE_CODE = r'''
# --- per-tamper-type row-level breakdown on strong / subtle (external) ---
TAMPER_TYPES = [
    "altitude_spike", "combined", "coordinate_jump", "deletion_gap",
    "heading_inconsistency", "injection", "precision_rounding",
    "speed_inconsistency", "timestamp_drift",
]


def scores_of(df, feats):
    return hgb.predict_proba(feats)[:, 1], svm.decision_function(feats)


def per_type_row(df, feats, thr_h, thr_s, dataset):
    sh, ss = scores_of(df, feats)
    r = pd.DataFrame({
        "tamper_type": df["tamper_type"].to_numpy(),
        "sh": sh, "ss": ss,
    })
    pred_h = r["sh"] >= thr_h
    pred_s = r["ss"] >= thr_s

    rows = []
    for t in TAMPER_TYPES + ["normal"]:
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


per_type_tables = []
for name, df, feats in [("strong", str_s, feats_str), ("subtle", sub_s, feats_sub)]:
    per_type_tables.append(per_type_row(df, feats, thr_h, thr_s, name))

res_type_row = pd.concat(per_type_tables, ignore_index=True)
print(res_type_row.to_string(index=False))
res_type_row.to_csv(ROOT / "results/per_type_row_rep00.csv", index=False)
'''

PER_TYPE_CASE_CODE = r'''
# --- case-level detection by tamper type ---
def cases_of(df, feats):
    sh, ss = scores_of(df, feats)
    out = pd.DataFrame({
        "case_id": df["case_id"].to_numpy(),
        "name": df["case_name"].to_numpy(),
        "sh": sh, "ss": ss,
    })
    c = out.groupby("case_id", sort=False).agg(
        name=("name", "first"), sh=("sh", "max"), ss=("ss", "max"),
    ).reset_index()
    c["type"] = c["name"].str.split("_", n=2).str[2]
    return c


def per_type_case(c, thr_h, thr_s, dataset):
    pred_h = c["sh"] >= thr_h
    pred_s = c["ss"] >= thr_s
    rows = []
    for t in TAMPER_TYPES + ["normal"]:
        sel = c["type"] == t
        n = int(sel.sum())
        rows.append({
            "dataset": dataset, "tamper_type": t, "n_cases": n,
            "gb_flag_rate": round(float((pred_h & sel).sum()) / n, 4) if n else np.nan,
            "svm_flag_rate": round(float((pred_s & sel).sum()) / n, 4) if n else np.nan,
        })
    return pd.DataFrame(rows)


per_type_case_tables = []
for name, df, feats in [("strong", str_s, feats_str), ("subtle", sub_s, feats_sub)]:
    per_type_case_tables.append(per_type_case(cases_of(df, feats), thr_c_h, thr_c_s, name))

res_type_case = pd.concat(per_type_case_tables, ignore_index=True)
print(res_type_case.to_string(index=False))
res_type_case.to_csv(ROOT / "results/per_type_case_rep00.csv", index=False)
'''

ROBUST_CODE = r'''
# --- Step 5: identical protocol repeated over replicates 0..3 ---
def load_profile(raw_dir: Path, profile: str) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / f"{profile}.csv", dtype=DTYPES)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", errors="coerce")
    return df


def prepare_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["case_id", "row_idx"]).reset_index(drop=True)
    feats = eng(df)
    feats = feats.astype({c: "float32" for c in feats.columns if c != "ts_gap_flag"})
    return df, feats


def run_replicate(repl: int) -> pd.DataFrame:
    raw = ROOT / f"data/raw/rep_{repl:02d}"
    bal_s, feats_bal = prepare_df(load_profile(raw, "balanced"))
    str_s, feats_str = prepare_df(load_profile(raw, "strong"))
    sub_s, feats_sub = prepare_df(load_profile(raw, "subtle"))

    case_label = bal_s.groupby("case_id")["label"].max().rename("case_label").reset_index()
    tr_cases, va_cases = train_test_split(
        case_label, test_size=0.2, stratify=case_label["case_label"], random_state=42,
    )
    tr_mask = bal_s["case_id"].isin(tr_cases["case_id"]).to_numpy()
    va_mask = bal_s["case_id"].isin(va_cases["case_id"]).to_numpy()

    X_tr = feats_bal.loc[tr_mask]
    y_tr = bal_s["label"].to_numpy()[tr_mask]

    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=15, random_state=42,
    )
    hgb.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
    svm = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("svc", LinearSVC(class_weight="balanced", dual="auto", max_iter=3000, random_state=42)),
    ])
    svm.fit(X_tr, y_tr)

    yva = bal_s["label"].to_numpy()[va_mask]
    thr_h = tune_threshold(yva, hgb.predict_proba(feats_bal.loc[va_mask])[:, 1])
    thr_s = tune_threshold(yva, svm.decision_function(feats_bal.loc[va_mask]))

    rows = []
    for name, df, feats, mask in [
        ("balanced_holdout", bal_s, feats_bal, va_mask),
        ("strong", str_s, feats_str, None),
        ("subtle", sub_s, feats_sub, None),
    ]:
        y = df["label"].to_numpy()[mask] if mask is not None else df["label"].to_numpy()
        F = feats.loc[mask] if mask is not None else feats
        sh = hgb.predict_proba(F)[:, 1]
        ss = svm.decision_function(F)
        rows.append({"replicate": repl, "dataset": name, "model": "GradientBoosting", **row_metrics(y, sh, thr_h)})
        rows.append({"replicate": repl, "dataset": name, "model": "LinearSVC", **row_metrics(y, ss, thr_s)})
    return pd.DataFrame(rows)


# free the rep_00 workspace frames before re-running all replicates
del data, bal_s, feats_bal, str_s, feats_str, sub_s, feats_sub

robust = pd.concat([run_replicate(r) for r in range(4)], ignore_index=True)
robust.to_csv(ROOT / "results/robustness_row_rep00_03.csv", index=False)

print("---- per replicate (subtle) ----")
piv = robust[robust["dataset"] == "subtle"][["replicate", "model", "pr_auc", "roc_auc", "f1"]]
print(piv.to_string(index=False))

print("\n---- mean ± std across replicates ----")
summ = robust.groupby(["dataset", "model"]).agg(
    pr_auc=("pr_auc", lambda x: f"{x.mean():.4f} ± {x.std(ddof=0):.4f}"),
    roc_auc=("roc_auc", lambda x: f"{x.mean():.4f} ± {x.std(ddof=0):.4f}"),
    f1=("f1", lambda x: f"{x.mean():.4f} ± {x.std(ddof=0):.4f}"),
    bal_acc=("bal_acc", lambda x: f"{x.mean():.4f} ± {x.std(ddof=0):.4f}"),
).reset_index()
print(summ.to_string(index=False))
summ.to_csv(ROOT / "results/robustness_summary_rep00_03.csv", index=False)
'''

RBF_CODE = r'''
# --- Step 6: nonlinear RBF SVM (sensitivity experiment) ---
# RBF SVC is O(n^2)-ish, so we train on a deterministic, label-stratified sample of the
# balanced training rows (same features, same seed). Test sets stay untouched.
import time as _time
from sklearn.svm import SVC

RBF_SAMPLE = 20000


def run_rbf_replicate(repl: int) -> pd.DataFrame:
    raw = ROOT / f"data/raw/rep_{repl:02d}"
    bal_s, feats_bal = prepare_df(load_profile(raw, "balanced"))
    str_s, feats_str = prepare_df(load_profile(raw, "strong"))
    sub_s, feats_sub = prepare_df(load_profile(raw, "subtle"))

    case_label = bal_s.groupby("case_id")["label"].max().rename("case_label").reset_index()
    tr_cases, va_cases = train_test_split(
        case_label, test_size=0.2, stratify=case_label["case_label"], random_state=42,
    )
    tr_mask = bal_s["case_id"].isin(tr_cases["case_id"]).to_numpy()
    va_mask = bal_s["case_id"].isin(va_cases["case_id"]).to_numpy()
    X_tr = feats_bal.loc[tr_mask]
    y_tr = bal_s["label"].to_numpy()[tr_mask]

    X_s, _, y_s, _ = train_test_split(
        X_tr, y_tr, train_size=RBF_SAMPLE, stratify=y_tr, random_state=42,
    )
    t0 = _time.time()
    rbf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                    cache_size=500, random_state=42)),
    ])
    rbf.fit(X_s, y_s)
    fit_s = round(_time.time() - t0, 1)

    yva = bal_s["label"].to_numpy()[va_mask]
    thr_r = tune_threshold(yva, rbf.decision_function(feats_bal.loc[va_mask]))

    rows = []
    for name, df, feats, mask in [
        ("balanced_holdout", bal_s, feats_bal, va_mask),
        ("strong", str_s, feats_str, None),
        ("subtle", sub_s, feats_sub, None),
    ]:
        y = df["label"].to_numpy()[mask] if mask is not None else df["label"].to_numpy()
        F = feats.loc[mask] if mask is not None else feats
        sr = rbf.decision_function(F)
        rows.append({"replicate": repl, "dataset": name, "model": "RBF-SVC",
                     "train_rows": RBF_SAMPLE, "fit_s": fit_s,
                     **row_metrics(y, sr, thr_r)})
    return pd.DataFrame(rows)


rbf_res = pd.concat([run_rbf_replicate(r) for r in range(4)], ignore_index=True)
rbf_res.to_csv(ROOT / "results/rbf_row_rep00_03.csv", index=False)

all3 = pd.concat([robust, rbf_res], ignore_index=True)
all3.to_csv(ROOT / "results/all_models_row_rep00_03.csv", index=False)

print("---- subtle PR-AUC per replicate ----")
print(all3[all3["dataset"] == "subtle"].pivot(
    index="replicate", columns="model", values="pr_auc").to_string())

print("\n---- mean ± std across replicates (subtle) ----")
for m in ["GradientBoosting", "LinearSVC", "RBF-SVC"]:
    s = all3[(all3["dataset"] == "subtle") & (all3["model"] == m)]
    print(f"{m:16s} pr_auc={s['pr_auc'].mean():.4f} ± {s['pr_auc'].std(ddof=0):.4f}"
          f"   f1={s['f1'].mean():.4f} ± {s['f1'].std(ddof=0):.4f}")

print("\n---- full summary (mean ± std) + fit time ----")
summ3 = all3.groupby(["dataset", "model"]).agg(
    pr_auc=("pr_auc", lambda x: f"{x.mean():.4f} ± {x.std(ddof=0):.4f}"),
    roc_auc=("roc_auc", lambda x: f"{x.mean():.4f} ± {x.std(ddof=0):.4f}"),
    f1=("f1", lambda x: f"{x.mean():.4f} ± {x.std(ddof=0):.4f}"),
    fit_s=("fit_s", "mean"),
).reset_index()
print(summ3.to_string(index=False))
summ3.to_csv(ROOT / "results/all_models_summary_rep00_03.csv", index=False)
'''

RBF_LC_CODE = r'''
# --- Step 7: RBF learning curve — is 20,000 rows enough? (rep_00) ---
# Same protocol at increasing sample sizes. If PR-AUC plateaus, 20k is justified.
# The probe evaluates on the balanced holdout + a deterministic 120k-row sample of subtle
# (prediction cost of RBF on the full 700k-row test is prohibitive for a curve).

SIZES = [5000, 10000, 20000, 40000]
REPL = 0
raw = ROOT / f"data/raw/rep_{REPL:02d}"
bal_s, feats_bal = prepare_df(load_profile(raw, "balanced"))
sub_s, feats_sub = prepare_df(load_profile(raw, "subtle"))

case_label = bal_s.groupby("case_id")["label"].max().rename("case_label").reset_index()
tr_cases, va_cases = train_test_split(
    case_label, test_size=0.2, stratify=case_label["case_label"], random_state=42,
)
tr_mask = bal_s["case_id"].isin(tr_cases["case_id"]).to_numpy()
va_mask = bal_s["case_id"].isin(va_cases["case_id"]).to_numpy()
X_tr = feats_bal.loc[tr_mask]
y_tr = bal_s["label"].to_numpy()[tr_mask]
yva = bal_s["label"].to_numpy()[va_mask]

# deterministic 120k-row slice of the subtle test profile
lc_idx = np.random.default_rng(42).choice(len(sub_s), size=120000, replace=False)
sub_s_lc = sub_s.iloc[lc_idx]
feats_sub_lc = feats_sub.iloc[lc_idx]

lc_rows = []
for n in SIZES:
    X_s, _, y_s, _ = train_test_split(X_tr, y_tr, train_size=n, stratify=y_tr, random_state=42)
    t0 = _time.time()
    rbf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                    cache_size=500, random_state=42)),
    ])
    rbf.fit(X_s, y_s)
    fit_s = round(_time.time() - t0, 1)
    nsv = len(rbf.named_steps["svc"].support_)
    thr = tune_threshold(yva, rbf.decision_function(feats_bal.loc[va_mask]))
    for name, df, feats in [("balanced_holdout", bal_s, feats_bal),
                            ("subtle_120k", sub_s_lc, feats_sub_lc)]:
        mask = va_mask if name == "balanced_holdout" else np.ones(len(df), dtype=bool)
        y = df["label"].to_numpy()[mask]
        s = rbf.decision_function(feats.loc[mask])
        lc_rows.append({"sample_size": n, "dataset": name, "fit_s": fit_s,
                        "n_support_vectors": nsv, **row_metrics(y, s, thr)})

lc = pd.DataFrame(lc_rows)
lc.to_csv(ROOT / "results/rbf_learning_curve_rep00.csv", index=False)

print("---- RBF learning curve: PR-AUC vs sample size (rep_00) ----")
print(lc.pivot(index="sample_size", columns="dataset", values="pr_auc").to_string())
print("\nF1:")
print(lc.pivot(index="sample_size", columns="dataset", values="f1").to_string())
print("\nfit_s / support vectors:")
print(lc.groupby("sample_size")[["fit_s", "n_support_vectors"]].first().to_string())
'''

cells = [
    ("markdown",
     "# Gradient Boosting vs Support Vector Machine for Drone Flight-Log Tamper Detection\n\n"
     "Dataset: **Drone Telemetry Tampering Dataset v2** (Kaggle, CC BY-SA 4.0, synthetic).\n"
     "This notebook is step-by-step and reproducible via `uv`. Step 1 (this file): data audit "
     "for one replicate (`rep_00`)."),
    ("markdown", "## Setup\nEnvironment is managed by `uv` (see `pyproject.toml` / `uv.lock`). "
     "Everything below is run inside the project virtual environment."),
    ("code", SRC),
    ("markdown",
     "## Dataset overview\n\n"
     "The archive contains profiles `balanced`, `strong`, `subtle`, each with 4 replicates "
     "(`rep_00`–`rep_03`) of 60 flight cases generated from the same decoded source log.\n\n"
     "**Design decision:** run the primary experiment on `replicate=0`:\n"
     "- `balanced/rep_00` -> development (train + in-profile holdout)\n"
     "- `strong/rep_00`, `subtle/rep_00` -> untouched external tests\n\n"
     "Replicates 1–3 are reserved for a robustness pass later."),
    ("code", AUDIT_CODE),
    ("markdown", "## Detailed audit"),
    ("code", AUDIT_DETAIL_CODE),
    ("markdown", "## Known artifacts & label semantics"),
    ("code", ARTIFACT_CODE),
    ("markdown", FINDINGS_MD),
    ("markdown",
     "## Step 2 — Temporal / kinematic feature engineering\n\n"
     "Convert each flight sequence into per-row temporal and kinematic features. All "
     "differences are computed **within each case** (grouped by `case_id`), the first row "
     "of each case has no predecessor (`NaN`), and the sentinel gap is clipped.\n\n"
     "**Excluded columns (leakage):** `label`, `tamper_type`, `case_name`, `profile`, "
     "`replicate`, `case_id`, `row_idx`, `source`, `original_row_idx`, and raw `timestamp`."),
    ("code", FEAT_CODE),
    ("markdown",
     "## Step 2b — Case-level 80/20 split (balanced only)\n\n"
     "Split the **60 balanced cases** into train (80%) and in-profile holdout (20%), "
     "stratified by whether a case contains any tampered row. Rows are never split "
     "individually. `strong` and `subtle` remain untouched external tests."),
    ("code", SPLIT_CODE),
    ("markdown",
     "## Step 3 — Models\n\n"
     "- **Gradient Boosting:** `HistGradientBoostingClassifier` (scalable, NaN-safe) with "
     "balanced training sample weights.\n"
     "- **Support Vector Machine:** `LinearSVC` (linear kernel, scales to millions of "
     "rows) wrapped in `SimpleImputer` + `StandardScaler`, with `class_weight='balanced'`.\n\n"
     "Both models see the **same features**, the **same training cases**, and the same "
     "random seed. `balanced` severity magnitude is 1.0 by construction."),
    ("code", MODEL_CODE),
    ("markdown",
     "## Step 3b — Row-level evaluation\n\n"
     "Scores: Gradient Boosting uses `predict_proba`; `LinearSVC` uses `decision_function` "
     "(both are monotone anomaly scores). Threshold-free metrics (ROC-AUC, PR-AUC) plus a "
     "per-model threshold tuned on the balanced holdout to maximize F1, then applied "
     "unchanged to `strong` and `subtle`."),
    ("code", EVAL_ROW_CODE),
    ("markdown",
     "## Step 3c — Case-level evaluation\n\n"
     "Aggregate row anomaly scores to each flight case: `case_score = max(row score)`, "
     "`case_label = max(row label)` (case contains any tampering)."),
    ("code", EVAL_CASE_CODE),
    ("markdown",
     "## Step 4 — Per-tamper-type breakdown (replicate 0)\n\n"
     "Which tampering techniques does each model detect well or poorly? For every tamper "
     "type we treat it as the positive class vs. normal rows only, using the fixed "
     "thresholds tuned in Step 3b (no re-tuning on the test sets). `flag_rate` = recall "
     "for tampered rows, false-positive rate for normal rows. PR-AUC is threshold-free "
     "detectability of the type against normal telemetry."),
    ("code", PER_TYPE_CODE),
    ("markdown",
     "## Step 4b — Case-level detection by tamper type\n\n"
     "Aggregate to cases (`max` row score), flag each case with the Step 3c thresholds, "
     "then break the flag rate down by tamper type. For normal cases this is the "
     "false-positive rate."),
    ("code", PER_TYPE_CASE_CODE),
    ("markdown",
     "## Step 5 — Robustness pass (replicates 0–3)\n\n"
     "Repeat the **exact same protocol** for every replicate: same features, same "
     "`random_state=42` split, same model hyperparameters, and a threshold tuned on each "
     "replicate's own balanced holdout (model selection never touches `strong`/`subtle`). "
     "Report mean ± std across replicates to quantify sensitivity to the synthetic "
     "perturbation arrangement."),
    ("code", ROBUST_CODE),
    ("markdown",
     "## Step 6 — Nonlinear (RBF) SVM sensitivity experiment\n\n"
     "The comparison so far is Gradient Boosting vs a **linear** SVM. To test whether the "
     "conclusion is an artifact of the linear kernel, train an **RBF-kernel SVM** on a "
     "deterministic, label-stratified sample of 20,000 balanced training rows (same "
     "features, same seed, `C=1`, `gamma='scale'`). Test sets remain untouched; thresholds "
     "are tuned on each replicate's balanced holdout only. Fit time is recorded."),
    ("code", RBF_CODE),
    ("markdown",
     "## Step 7 — RBF learning curve\n\n"
     "Is the 20,000-row training sample enough? Repeat the identical RBF protocol on "
     "`rep_00` at 5k / 10k / 20k / 40k rows (same features, same seed, thresholds tuned on "
     "the holdout). To keep the curve fast, the probe evaluates on the balanced holdout "
     "plus a deterministic 120k-row sample of `subtle` (full 700k-row RBF scoring is "
     "prohibitively slow for a multi-point curve). If PR-AUC plateaus, 20k is defensible."),
    ("code", RBF_LC_CODE),
]


def build():
    nb = {
        "cells": [
            {
                "cell_type": ctype,
                "metadata": {},
                "source": [content if ctype == "markdown" else content],
                "outputs": [] if ctype == "code" else None,
                "execution_count": None if ctype == "code" else None,
            }
            for ctype, content in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = NOTEBOOKS / "01_gradient_boosting_vs_svm.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    build()