# Drone Telemetry Tamper Detection

Comparison of Gradient Boosting, linear SVM, and nonlinear RBF-SVM for synthetic UAV
flight-log tamper detection.

## Quick Start

The Kaggle archive must be available as `drone.zip` in the project root.

```bash
uv sync
uv run python scripts/extract_replicates.py
uv run jupyter lab
```

Open `notebooks/01_gradient_boosting_vs_svm.ipynb` and run it top-to-bottom. To execute
headlessly:

```bash
uv run jupyter nbconvert --to notebook --execute \
  --inplace notebooks/01_gradient_boosting_vs_svm.ipynb
uv run python scripts/make_figures.py
```

## Deliverables

- `REPORT.md` — final methods, results, limitations, and conclusions.
- `notebooks/01_gradient_boosting_vs_svm.ipynb` — executed reproducible analysis.
- `results/` — row-level, case-level, per-type, robustness, RBF, and learning-curve tables.
- `figures/` — report figures.

The primary result is Gradient Boosting > RBF-SVM > LinearSVC. The RBF-SVM uses a
deterministic 20,000-row training sample because a full-data RBF fit is not tractable.
All cases are synthetic variants of one source flight log, so results measure tampering
magnitude generalization rather than generalization to new flights.
