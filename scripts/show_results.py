"""Print the Chapter 4 result tables to the console, formatted as they appear in the report.

Reads the saved CSVs in results/ -- it does not re-run anything, so it is instant.
Output is ASCII-only (the plus-minus sign is written as +/-), so it renders correctly
on any Windows console code page.

Usage:
    uv run python scripts/show_results.py            # every table
    uv run python scripts/show_results.py 4.3 4.4    # only those
    uv run python scripts/show_results.py --list     # table names only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from drone_tamper.config import RESULTS

DATASETS = {"balanced_holdout": "Balanced holdout", "strong": "Strong", "subtle": "Subtle"}
MODELS = {"GradientBoosting": "Gradient Boosting", "RBF-SVC": "RBF SVM",
          "LinearSVC": "Linear SVM", "GradientBoosting-transfer": "GB (transfer)"}
DS_ORDER = {k: i for i, k in enumerate(DATASETS)}
M_ORDER = {k: i for i, k in enumerate(MODELS)}


def load(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    if not path.exists():
        print(f"  (missing: {path.name} -- run scripts/run_pipeline.py first)\n")
        return None
    return pd.read_csv(path)


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by dataset then model, map to report labels, strip non-ASCII."""
    if "dataset" in df and "model" in df:
        df = df.sort_values(
            ["dataset", "model"],
            key=lambda s: s.map(DS_ORDER if s.name == "dataset" else M_ORDER))
    for col, mapping in (("dataset", DATASETS), ("model", MODELS)):
        if col in df:
            df[col] = df[col].map(lambda v: mapping.get(v, v))
    return df.map(lambda v: v.replace("±", "+/-") if isinstance(v, str) else v)


def show(number: str, caption: str, df: pd.DataFrame, cols: dict) -> None:
    print(f"Table {number}: {caption}")
    out = tidy(df)[list(cols)].rename(columns=cols)
    print(out.to_string(index=False))
    print()


def t42() -> None:
    d = load("row_metrics_rep00.csv")
    if d is not None:
        show("4.2", "Row-level results, single replicate.", d,
             {"dataset": "Dataset", "model": "Model", "pr_auc": "PR-AUC",
              "roc_auc": "ROC-AUC", "f1": "F1", "mcc": "MCC"})


def t43() -> None:
    d = load("all_models_summary_rep00_03.csv")
    if d is not None:
        show("4.3", "Row-level PR-AUC across four replicates (mean +/- standard deviation).", d,
             {"dataset": "Dataset", "model": "Model", "pr_auc": "PR-AUC",
              "roc_auc": "ROC-AUC", "f1": "F1"})


def t44() -> None:
    d = load("case_metrics_rep00.csv")
    if d is not None:
        show("4.4", "Case-level results at the tuned case threshold, single replicate.", d,
             {"dataset": "Dataset", "model": "Model", "n": "Cases", "f1": "F1",
              "rec": "Recall", "mcc": "MCC"})


def t45(profile: str = "subtle") -> None:
    d = load("per_type_row_rep00.csv")
    if d is None:
        return
    d = d[d.dataset == profile].sort_values("gb_pr_auc", ascending=False)
    d["n_rows"] = d["n_rows"].map("{:,}".format)
    for c in ("gb_flag_rate", "svm_flag_rate"):
        d[c] = (d[c] * 100).map("{:.1f}%".format)
    for c in ("gb_pr_auc", "svm_pr_auc"):
        d[c] = d[c].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}")
    show("4.5", f"Row-level detection by tamper type, {profile} profile, single replicate.", d,
         {"tamper_type": "Tamper type", "n_rows": "Records", "gb_pr_auc": "GB PR-AUC",
          "svm_pr_auc": "SVM PR-AUC", "gb_flag_rate": "GB flagged", "svm_flag_rate": "SVM flagged"})


def t46() -> None:
    d = load("rbf_learning_curve_rep00.csv")
    if d is None:
        return
    wide = d.pivot(index=["sample_size", "fit_s", "n_support_vectors"],
                   columns="dataset", values="pr_auc").reset_index()
    wide["sample_size"] = wide["sample_size"].map("{:,}".format)
    wide["n_support_vectors"] = wide["n_support_vectors"].map("{:,}".format)
    show("4.6", "RBF SVM learning curve, single replicate.", wide,
         {"sample_size": "Training rows", "fit_s": "Fit time (s)",
          "n_support_vectors": "Support vectors",
          "balanced_holdout": "PR-AUC (holdout)", "subtle_120k": "PR-AUC (subtle)"})


def t47() -> None:
    d = load("real_log_summary.csv")
    if d is None:
        return
    g = d.groupby("detector", sort=False).agg(
        flag_lo=("flag_rate", "min"), flag_hi=("flag_rate", "max"),
        st_lo=("stationary_flag_rate", "min"), st_hi=("stationary_flag_rate", "max")).reset_index()
    g["flagged"] = g.apply(lambda r: f"{r.flag_lo:.1%} - {r.flag_hi:.1%}", axis=1)
    g["stationary"] = g.apply(lambda r: f"{r.st_lo:.1%} - {r.st_hi:.1%}", axis=1)
    show("4.7", "External check on three real flight logs (unlabelled).", g,
         {"detector": "Detector", "flagged": "Records flagged",
          "stationary": "Flagged while stationary"})


def fig41() -> None:
    d = load("all_models_summary_rep00_03.csv")
    if d is None:
        return
    d = tidy(d)
    d["mean"] = d.pr_auc.str.split().str[0].astype(float)
    d["std"] = d.pr_auc.str.split().str[2].astype(float)
    print("Figure 4.1: plotted values (bar height and error bar).")
    print(d[["dataset", "model", "mean", "std"]]
          .rename(columns={"dataset": "Dataset", "model": "Model",
                           "mean": "PR-AUC (bar)", "std": "Std (error bar)"})
          .to_string(index=False))
    print()


TABLES = {"4.2": t42, "4.3": t43, "4.4": t44, "4.5": t45,
          "4.6": t46, "4.7": t47, "fig4.1": fig41}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", help=f"one or more of: {', '.join(TABLES)}")
    ap.add_argument("--list", action="store_true", help="list available tables and exit")
    args = ap.parse_args()

    if args.list:
        print("Available:", ", ".join(TABLES))
        return
    wanted = args.which or list(TABLES)
    unknown = [w for w in wanted if w not in TABLES]
    if unknown:
        raise SystemExit(f"unknown table(s) {unknown}; choose from {list(TABLES)}")
    for w in wanted:
        TABLES[w]()


if __name__ == "__main__":
    main()
