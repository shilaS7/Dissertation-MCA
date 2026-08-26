"""Generate report figures from the saved experiment results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def split_pm(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    parts = values.astype(str).str.split()
    return parts.str[0].astype(float).to_numpy(), parts.str[2].astype(float).to_numpy()


def model_summary() -> None:
    df = pd.read_csv(RESULTS / "all_models_summary_rep00_03.csv")
    models = ["GradientBoosting", "RBF-SVC", "LinearSVC"]
    datasets = ["balanced_holdout", "strong", "subtle"]
    labels = {
        "balanced_holdout": "Balanced holdout",
        "strong": "Strong",
        "subtle": "Subtle",
    }
    colors = {
        "GradientBoosting": "#176b87",
        "RBF-SVC": "#d88732",
        "LinearSVC": "#8a5a9e",
    }

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(datasets))
    width = 0.24
    for i, model in enumerate(models):
        rows = df[df["model"] == model].set_index("dataset").loc[datasets]
        means, errors = split_pm(rows["pr_auc"])
        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=errors,
            capsize=4,
            label=model,
            color=colors[model],
            edgecolor="white",
            linewidth=0.8,
        )
    ax.set_xticks(x, [labels[d] for d in datasets])
    ax.set_ylim(0, 1)
    ax.set_ylabel("PR-AUC (mean +/- std across replicates)")
    ax.set_title("Tamper detection performance by model and profile")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_pr_auc_robustness.png", dpi=180)
    plt.close(fig)


def type_breakdown() -> None:
    df = pd.read_csv(RESULTS / "per_type_row_rep00.csv")
    df = df[df["tamper_type"] != "normal"].copy()
    types = list(df["tamper_type"].drop_duplicates())
    columns = [
        ("strong", "gb_pr_auc", "GB / strong"),
        ("strong", "svm_pr_auc", "Linear SVM / strong"),
        ("subtle", "gb_pr_auc", "GB / subtle"),
        ("subtle", "svm_pr_auc", "Linear SVM / subtle"),
    ]
    matrix = np.array(
        [
            [
                float(df[(df["dataset"] == dataset) & (df["tamper_type"] == t)][metric].iloc[0])
                for dataset, metric, _ in columns
            ]
            for t in types
        ]
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), [label for _, _, label in columns], rotation=25, ha="right")
    ax.set_yticks(range(len(types)), types)
    ax.set_title("Per-tamper-type PR-AUC (replicate 0)")
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            color = "white" if matrix[r, c] < 0.55 else "black"
            ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, label="PR-AUC")
    fig.tight_layout()
    fig.savefig(FIGURES / "per_type_pr_auc.png", dpi=180)
    plt.close(fig)


def learning_curve() -> None:
    df = pd.read_csv(RESULTS / "rbf_learning_curve_rep00.csv")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    styles = {
        "balanced_holdout": ("#176b87", "Balanced holdout"),
        "subtle_120k": ("#d88732", "Subtle sample (120k rows)"),
    }
    for dataset, (color, label) in styles.items():
        rows = df[df["dataset"] == dataset].sort_values("sample_size")
        ax.plot(
            rows["sample_size"],
            rows["pr_auc"],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
    ax.set_xlabel("RBF-SVM training rows")
    ax.set_ylabel("PR-AUC")
    ax.set_title("RBF-SVM sample-size sensitivity")
    ax.set_xticks([5000, 10000, 20000, 40000], ["5k", "10k", "20k", "40k"])
    ax.set_ylim(0.5, 0.75)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "rbf_learning_curve.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    model_summary()
    type_breakdown()
    learning_curve()
    print(f"Wrote figures to {FIGURES}")
