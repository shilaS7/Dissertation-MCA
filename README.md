# Drone Telemetry Tamper Detection

Comparison of Gradient Boosting, linear SVM, and nonlinear RBF-SVM for synthetic UAV
flight-log tamper detection.

## Quick Start

The Kaggle archive must be available as `drone.zip` in the project root.

```bash
uv sync
uv run python scripts/extract_replicates.py
uv run python scripts/run_pipeline.py
uv run python scripts/make_figures.py
```

`scripts/run_pipeline.py` runs the full experiment (data audit, feature engineering,
model training, row/case/per-type evaluation, the 4-replicate robustness pass, and the
RBF sensitivity/learning-curve experiments) as plain Python — no notebook or Jupyter
involved. It prints progress to the terminal and tees the full run to
`results/pipeline_run.log`. To iterate on just the early steps without re-running the
slower robustness/RBF passes:

```bash
uv run python scripts/run_pipeline.py --only main
uv run python scripts/run_pipeline.py --skip rbf,rbf_lc
```

The pipeline logic lives in the importable `drone_tamper` package under `src/`.

## Deliverables

- `REPORT.md` — final methods, results, limitations, and conclusions.
- `scripts/run_pipeline.py` — reproducible end-to-end analysis (replaces the old notebook).
- `results/` — row-level, case-level, per-type, robustness, RBF, and learning-curve tables.
- `figures/` — report figures.

The primary result is Gradient Boosting > RBF-SVM > LinearSVC. The RBF-SVM uses a
deterministic 20,000-row training sample because a full-data RBF fit is not tractable.
All cases are synthetic variants of one source flight log, so results measure tampering
magnitude generalization rather than generalization to new flights.
