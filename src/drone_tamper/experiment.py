"""Orchestration for the tamper-detection experiment (Steps 1-7).

Each `run_*` function is an independent, rerunnable block, mirroring the structure the
project originally had as separate notebook cells: `run_main` reuses one in-memory load of
replicate 0, while `run_robustness`, `run_rbf`, and `run_rbf_learning_curve` each reload
the replicates they need from disk.
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from . import aggregate, features, io, models
from .audit import audit as audit_profile
from .config import (
    DATA_INDEX, DATA_RAW, PROFILES, RBF_LC_SIZES, RBF_SAMPLE, REPLICATES, RESULTS, SEED,
)
from .metrics import row_metrics, tune_threshold
from .splits import case_split


def load_and_prepare(raw_dir, profile: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return features.prepare(io.load_profile(raw_dir, profile))


# --- Steps 1-4: main experiment on replicate 0 ---------------------------------


def run_main(replicate: int = 0) -> dict:
    raw = DATA_RAW / f"rep_{replicate:02d}"
    out_dir = DATA_INDEX / f"rep_{replicate:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Step 1: Data audit (replicate {replicate}) ===")
    data = {p: io.load_profile(raw, p) for p in PROFILES}
    for p, df in data.items():
        print(f"{p:9s} rows={len(df):>8,d}  cases={df['case_id'].nunique():>3}  "
              f"anomaly_rate={df['label'].mean():.4f}")

    audit_summary = {p: audit_profile(df, p) for p, df in data.items()}
    audit_path = out_dir / f"audit_rep{replicate:02d}.json"
    audit_path.write_text(json.dumps(audit_summary, indent=2))
    print(f"wrote {audit_path}")

    print("\n=== Step 2: Feature engineering ===")
    bal_s, feats_bal = features.prepare(data["balanced"])
    str_s, feats_str = features.prepare(data["strong"])
    sub_s, feats_sub = features.prepare(data["subtle"])
    print(f"features: {list(feats_bal.columns)}")

    print("\n=== Step 2b: Case-level 80/20 split (balanced only) ===")
    tr_mask, va_mask, tr_cases, va_cases = case_split(bal_s)
    print("train cases:", len(tr_cases), tr_cases["case_label"].value_counts().to_dict())
    print("holdout cases:", len(va_cases), va_cases["case_label"].value_counts().to_dict())
    print("train rows:", int(tr_mask.sum()), " holdout rows:", int(va_mask.sum()))

    print("\n=== Step 3: Train Gradient Boosting vs LinearSVC ===")
    X_tr = feats_bal.loc[tr_mask]
    y_tr = bal_s["label"].to_numpy()[tr_mask]

    hgb = models.build_hgb(SEED)
    hgb.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))

    svm = models.build_linear_svc(SEED)
    svm.fit(X_tr, y_tr)
    print("HistGB fitted; n_iter used:", hgb.n_iter_)
    print("LinearSVC fitted; n_iter reached:", svm.named_steps["svc"].n_iter_)

    def scores_of(df, feats):
        return hgb.predict_proba(feats)[:, 1], svm.decision_function(feats)

    print("\n=== Step 3b: Row-level evaluation ===")
    datasets = [("balanced_holdout", bal_s, feats_bal), ("strong", str_s, feats_str), ("subtle", sub_s, feats_sub)]
    results_row = []
    thr_h = thr_s = None
    for name, df, feats in datasets:
        sh, ss = scores_of(df, feats)
        y = df["label"].to_numpy()
        if name == "balanced_holdout":
            thr_h, thr_s = tune_threshold(y[va_mask], sh[va_mask]), tune_threshold(y[va_mask], ss[va_mask])
            y, sh, ss = y[va_mask], sh[va_mask], ss[va_mask]
            print(f"thresholds -> hgb={thr_h:.4f}  svm={thr_s:.4f}")
        results_row.append({"dataset": name, "model": "GradientBoosting", **row_metrics(y, sh, thr_h)})
        results_row.append({"dataset": name, "model": "LinearSVC", **row_metrics(y, ss, thr_s)})
    res_row = pd.DataFrame(results_row)
    print(res_row.to_string(index=False))
    res_row.to_csv(RESULTS / f"row_metrics_rep{replicate:02d}.csv", index=False)

    print("\n=== Step 3c: Case-level evaluation ===")
    sh_bal, ss_bal = scores_of(bal_s, feats_bal)
    case_va = aggregate.case_agg(bal_s["case_id"].to_numpy(), bal_s["label"].to_numpy(), sh_bal, ss_bal)
    case_va = case_va[case_va["case_id"].isin(va_cases["case_id"])]
    thr_c_h = tune_threshold(case_va["label"], case_va["sh"])
    thr_c_s = tune_threshold(case_va["label"], case_va["ss"])
    print(f"case thresholds -> hgb={thr_c_h:.4f} svm={thr_c_s:.4f}")

    results_case = []
    for name, df, feats in datasets:
        sh, ss = scores_of(df, feats)
        cs = aggregate.case_agg(df["case_id"].to_numpy(), df["label"].to_numpy(), sh, ss)
        if name == "balanced_holdout":
            cs = cs[cs["case_id"].isin(va_cases["case_id"])]
        results_case.append({"dataset": name, "model": "GradientBoosting", **row_metrics(cs["label"], cs["sh"], thr_c_h)})
        results_case.append({"dataset": name, "model": "LinearSVC", **row_metrics(cs["label"], cs["ss"], thr_c_s)})
    res_case = pd.DataFrame(results_case)
    print(res_case.to_string(index=False))
    res_case.to_csv(RESULTS / f"case_metrics_rep{replicate:02d}.csv", index=False)

    print("\n=== Step 4: Per-tamper-type breakdown (row level) ===")
    per_type_tables = []
    for name, df, feats in [("strong", str_s, feats_str), ("subtle", sub_s, feats_sub)]:
        sh, ss = scores_of(df, feats)
        per_type_tables.append(aggregate.per_type_row(df["tamper_type"].to_numpy(), sh, ss, thr_h, thr_s, name))
    res_type_row = pd.concat(per_type_tables, ignore_index=True)
    print(res_type_row.to_string(index=False))
    res_type_row.to_csv(RESULTS / f"per_type_row_rep{replicate:02d}.csv", index=False)

    print("\n=== Step 4b: Per-tamper-type breakdown (case level) ===")
    per_type_case_tables = []
    for name, df, feats in [("strong", str_s, feats_str), ("subtle", sub_s, feats_sub)]:
        sh, ss = scores_of(df, feats)
        c = aggregate.cases_of(df["case_id"].to_numpy(), df["case_name"].to_numpy(), sh, ss)
        per_type_case_tables.append(aggregate.per_type_case(c, thr_c_h, thr_c_s, name))
    res_type_case = pd.concat(per_type_case_tables, ignore_index=True)
    print(res_type_case.to_string(index=False))
    res_type_case.to_csv(RESULTS / f"per_type_case_rep{replicate:02d}.csv", index=False)

    return {"thr_h": thr_h, "thr_s": thr_s, "thr_c_h": thr_c_h, "thr_c_s": thr_c_s}


# --- Step 5: robustness pass across replicates ----------------------------------


def _run_robustness_replicate(repl: int) -> pd.DataFrame:
    raw = DATA_RAW / f"rep_{repl:02d}"
    bal_s, feats_bal = load_and_prepare(raw, "balanced")
    str_s, feats_str = load_and_prepare(raw, "strong")
    sub_s, feats_sub = load_and_prepare(raw, "subtle")

    tr_mask, va_mask, _, _ = case_split(bal_s)
    X_tr = feats_bal.loc[tr_mask]
    y_tr = bal_s["label"].to_numpy()[tr_mask]

    hgb = models.build_hgb(SEED)
    hgb.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
    svm = models.build_linear_svc(SEED)
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


def run_robustness(replicates=REPLICATES) -> pd.DataFrame:
    print(f"\n=== Step 5: Robustness pass (replicates {list(replicates)}) ===")
    robust = pd.concat([_run_robustness_replicate(r) for r in replicates], ignore_index=True)
    robust.to_csv(RESULTS / "robustness_row_rep00_03.csv", index=False)

    print("---- per replicate (subtle) ----")
    piv = robust[robust["dataset"] == "subtle"][["replicate", "model", "pr_auc", "roc_auc", "f1"]]
    print(piv.to_string(index=False))

    summ = robust.groupby(["dataset", "model"]).agg(
        pr_auc=("pr_auc", lambda x: f"{x.mean():.4f} ± {x.std(ddof=0):.4f}"),
        roc_auc=("roc_auc", lambda x: f"{x.mean():.4f} ± {x.std(ddof=0):.4f}"),
        f1=("f1", lambda x: f"{x.mean():.4f} ± {x.std(ddof=0):.4f}"),
        bal_acc=("bal_acc", lambda x: f"{x.mean():.4f} ± {x.std(ddof=0):.4f}"),
    ).reset_index()
    print("\n---- mean ± std across replicates ----")
    print(summ.to_string(index=False))
    summ.to_csv(RESULTS / "robustness_summary_rep00_03.csv", index=False)
    return robust


# --- Step 6: nonlinear RBF SVM sensitivity experiment ---------------------------


def _run_rbf_replicate(repl: int) -> pd.DataFrame:
    raw = DATA_RAW / f"rep_{repl:02d}"
    bal_s, feats_bal = load_and_prepare(raw, "balanced")
    str_s, feats_str = load_and_prepare(raw, "strong")
    sub_s, feats_sub = load_and_prepare(raw, "subtle")

    tr_mask, va_mask, _, _ = case_split(bal_s)
    X_tr = feats_bal.loc[tr_mask]
    y_tr = bal_s["label"].to_numpy()[tr_mask]

    X_s, _, y_s, _ = train_test_split(X_tr, y_tr, train_size=RBF_SAMPLE, stratify=y_tr, random_state=SEED)

    t0 = time.time()
    rbf = models.build_rbf_svc(SEED)
    rbf.fit(X_s, y_s)
    fit_s = round(time.time() - t0, 1)

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
                     "train_rows": RBF_SAMPLE, "fit_s": fit_s, **row_metrics(y, sr, thr_r)})
    return pd.DataFrame(rows)


def run_rbf(replicates=REPLICATES) -> pd.DataFrame:
    print(f"\n=== Step 6: Nonlinear (RBF) SVM sensitivity experiment (replicates {list(replicates)}) ===")
    rbf_res = pd.concat([_run_rbf_replicate(r) for r in replicates], ignore_index=True)
    rbf_res.to_csv(RESULTS / "rbf_row_rep00_03.csv", index=False)

    robustness_path = RESULTS / "robustness_row_rep00_03.csv"
    if not robustness_path.exists():
        raise FileNotFoundError(
            f"{robustness_path} not found — run Step 5 (run_robustness) first, "
            "since the combined model comparison needs its results."
        )
    robust = pd.read_csv(robustness_path)
    all3 = pd.concat([robust, rbf_res], ignore_index=True)
    all3.to_csv(RESULTS / "all_models_row_rep00_03.csv", index=False)

    print("---- subtle PR-AUC per replicate ----")
    print(all3[all3["dataset"] == "subtle"].pivot(
        index="replicate", columns="model", values="pr_auc").to_string())

    print("\n---- mean ± std across replicates (subtle) ----")
    for m in ["GradientBoosting", "LinearSVC", "RBF-SVC"]:
        s = all3[(all3["dataset"] == "subtle") & (all3["model"] == m)]
        print(f"{m:16s} pr_auc={s['pr_auc'].mean():.4f} ± {s['pr_auc'].std(ddof=0):.4f}"
              f"   f1={s['f1'].mean():.4f} ± {s['f1'].std(ddof=0):.4f}")

    summ3 = all3.groupby(["dataset", "model"]).agg(
        pr_auc=("pr_auc", lambda x: f"{x.mean():.4f} ± {x.std(ddof=0):.4f}"),
        roc_auc=("roc_auc", lambda x: f"{x.mean():.4f} ± {x.std(ddof=0):.4f}"),
        f1=("f1", lambda x: f"{x.mean():.4f} ± {x.std(ddof=0):.4f}"),
        fit_s=("fit_s", "mean"),
    ).reset_index()
    print("\n---- full summary (mean ± std) + fit time ----")
    print(summ3.to_string(index=False))
    summ3.to_csv(RESULTS / "all_models_summary_rep00_03.csv", index=False)
    return rbf_res


# --- Step 7: RBF learning curve --------------------------------------------------


def run_rbf_learning_curve(replicate: int = 0) -> pd.DataFrame:
    print(f"\n=== Step 7: RBF learning curve (replicate {replicate}) ===")
    raw = DATA_RAW / f"rep_{replicate:02d}"
    bal_s, feats_bal = load_and_prepare(raw, "balanced")
    sub_s, feats_sub = load_and_prepare(raw, "subtle")

    tr_mask, va_mask, _, _ = case_split(bal_s)
    X_tr = feats_bal.loc[tr_mask]
    y_tr = bal_s["label"].to_numpy()[tr_mask]
    yva = bal_s["label"].to_numpy()[va_mask]

    lc_idx = np.random.default_rng(SEED).choice(len(sub_s), size=120000, replace=False)
    sub_s_lc = sub_s.iloc[lc_idx]
    feats_sub_lc = feats_sub.iloc[lc_idx]

    lc_rows = []
    for n in RBF_LC_SIZES:
        X_s, _, y_s, _ = train_test_split(X_tr, y_tr, train_size=n, stratify=y_tr, random_state=SEED)
        t0 = time.time()
        rbf = models.build_rbf_svc(SEED)
        rbf.fit(X_s, y_s)
        fit_s = round(time.time() - t0, 1)
        nsv = len(rbf.named_steps["svc"].support_)
        thr = tune_threshold(yva, rbf.decision_function(feats_bal.loc[va_mask]))
        for name, df, feats in [("balanced_holdout", bal_s, feats_bal), ("subtle_120k", sub_s_lc, feats_sub_lc)]:
            mask = va_mask if name == "balanced_holdout" else np.ones(len(df), dtype=bool)
            y = df["label"].to_numpy()[mask]
            s = rbf.decision_function(feats.loc[mask])
            lc_rows.append({"sample_size": n, "dataset": name, "fit_s": fit_s,
                            "n_support_vectors": nsv, **row_metrics(y, s, thr)})

    lc = pd.DataFrame(lc_rows)
    lc.to_csv(RESULTS / f"rbf_learning_curve_rep{replicate:02d}.csv", index=False)

    print("---- RBF learning curve: PR-AUC vs sample size ----")
    print(lc.pivot(index="sample_size", columns="dataset", values="pr_auc").to_string())
    print("\nF1:")
    print(lc.pivot(index="sample_size", columns="dataset", values="f1").to_string())
    print("\nfit_s / support vectors:")
    print(lc.groupby("sample_size")[["fit_s", "n_support_vectors"]].first().to_string())
    return lc
