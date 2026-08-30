"""Threshold tuning and row/case-level metrics."""

import numpy as np
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def tune_threshold(y, s) -> float:
    prec, rec, thr = precision_recall_curve(y, s)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[int(np.argmax(f1s[:-1]))])


def row_metrics(y, s, thr: float) -> dict:
    yp = (s >= thr).astype(int)
    return {
        "n": int(len(y)), "pos": int(np.sum(y)), "roc_auc": round(roc_auc_score(y, s), 4),
        "pr_auc": round(average_precision_score(y, s), 4),
        "f1": round(f1_score(y, yp), 4), "prec": round(precision_score(y, yp), 4),
        "rec": round(recall_score(y, yp), 4),
        "bal_acc": round(balanced_accuracy_score(y, yp), 4),
        "mcc": round(matthews_corrcoef(y, yp), 4),
    }
