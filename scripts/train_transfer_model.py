"""Train the transfer variant of the Gradient Boosting model for scoring real flight logs.

Usage:
    uv run python scripts/train_transfer_model.py
    uv run python scripts/score_log.py data/logs/some_flight.csv --model results/models/gb_transfer_rep00.joblib

Same protocol as the main model (case-level 80/20 split of `balanced`, F1-tuned threshold on
the holdout, `strong`/`subtle` untouched) but on synthetic data thinned to the real logs'
~2 s sampling interval, and without absolute position/route features (see transfer.py).
This does NOT replace the main experiment -- the results reported for the model comparison
are unchanged. It is a separate model for exploratory scoring of real logs.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from drone_tamper import features, io, models, transfer
from drone_tamper.config import DATA_RAW, PROFILES, RESULTS, SEED, ensure_dirs
from drone_tamper.metrics import row_metrics, tune_threshold
from drone_tamper.splits import case_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--target-dt", type=float, default=transfer.TARGET_DT_S,
                        help="sampling interval (s) to thin the synthetic data to")
    args = parser.parse_args()
    ensure_dirs()

    raw = DATA_RAW / f"rep_{args.replicate:02d}"
    loaded = {p: io.load_profile(raw, p) for p in PROFILES}
    step = transfer.resample_step(loaded["balanced"], args.target_dt)
    print(f"synthetic median interval {transfer.median_dt_s(loaded['balanced']):.3f}s "
          f"-> keeping every {step}th row (target {args.target_dt}s)")

    data = {}
    for p, df in loaded.items():
        thinned = transfer.thin(df, step)
        df_s, feats = features.prepare(thinned)
        data[p] = (df_s, transfer.select_features(feats))
        print(f"{p:9s} rows {len(df):>8,d} -> {len(thinned):>7,d}  "
              f"anomaly_rate={thinned['label'].mean():.4f}")

    bal_s, feats_bal = data["balanced"]
    print(f"features ({feats_bal.shape[1]}): {list(feats_bal.columns)}")

    tr_mask, va_mask, _, _ = case_split(bal_s)
    X_tr, y_tr = feats_bal.loc[tr_mask], bal_s["label"].to_numpy()[tr_mask]
    hgb = models.build_hgb(SEED)
    hgb.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))

    y_va = bal_s["label"].to_numpy()[va_mask]
    thr = tune_threshold(y_va, hgb.predict_proba(feats_bal.loc[va_mask])[:, 1])
    print(f"threshold (F1 on balanced holdout) = {thr:.4f}")

    rows = []
    for name, (df_s, feats) in [("balanced_holdout", data["balanced"]), ("strong", data["strong"]),
                                ("subtle", data["subtle"])]:
        y, F = df_s["label"].to_numpy(), feats
        if name == "balanced_holdout":
            y, F = y[va_mask], feats.loc[va_mask]
        rows.append({"dataset": name, "model": "GradientBoosting-transfer",
                     **row_metrics(y, hgb.predict_proba(F)[:, 1], thr)})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    metrics_path = RESULTS / f"transfer_row_metrics_rep{args.replicate:02d}.csv"
    res.to_csv(metrics_path, index=False)
    print(f"wrote {metrics_path}")

    model_path = RESULTS / "models" / f"gb_transfer_rep{args.replicate:02d}.joblib"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": hgb, "threshold": thr, "feature_names": list(feats_bal.columns),
        "replicate": args.replicate, "thin_step": step, "target_dt_s": args.target_dt,
        "dropped_features": transfer.LOCATION_FEATURES,
    }, model_path)
    print(f"saved model -> {model_path}")


if __name__ == "__main__":
    main()
