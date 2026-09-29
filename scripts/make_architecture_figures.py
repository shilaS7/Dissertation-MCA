"""Draw architecture diagrams for the report: end-to-end pipeline, Gradient Boosting, and
linear vs RBF SVM. Hyperparameters shown match drone_tamper/models.py.

Usage:
    uv run python scripts/make_architecture_figures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.datasets import make_moons
from sklearn.svm import SVC, LinearSVC

from drone_tamper.config import FIGURES

INK = "#1f2933"
MUTED = "#52606d"
DATA = "#e3f0f7"      # data / inputs
PROC = "#f4f4f4"      # processing steps
GB = "#d6ebe6"        # gradient boosting
LIN = "#e8e1f2"       # linear SVM
RBF = "#fbe7d3"       # RBF SVM
OUT = "#fdf3c4"       # outputs / decisions
EDGE = "#7b8794"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

BW = False


def set_bw():
    """Switch the shared palette to black-and-white for print versions."""
    global BW, INK, MUTED, DATA, PROC, GB, LIN, RBF, OUT, EDGE
    BW = True
    INK = MUTED = EDGE = "black"
    DATA = PROC = GB = LIN = RBF = OUT = "white"


def box(ax, x, y, w, h, title, body="", fc=PROC, title_size=10.5, body_size=8.8):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=EDGE, lw=1.1))
    if body:
        ax.text(x, y + h / 2 - 0.16, title, ha="center", va="top", fontsize=title_size, weight="bold", color=INK)
        ax.text(x, y + h / 2 - 0.48, body, ha="center", va="top", fontsize=body_size, color=MUTED, linespacing=1.35)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=title_size, weight="bold", color=INK)


def arrow(ax, x1, y1, x2, y2, label="", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                                 color=EDGE, lw=1.3, linestyle=ls, shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2 + 0.08, (y1 + y2) / 2, label, fontsize=8.3, color=MUTED, va="center")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def pipeline_figure():
    fig, ax = canvas(11, 8.6)
    ax.text(5.5, 8.35, "End-to-end tamper-detection architecture", ha="center", fontsize=14, weight="bold", color=INK)

    box(ax, 5.5, 7.55, 6.2, 0.95, "Drone flight-log tampering dataset(s)",
        "tampering profiles (balanced / strong / subtle)  ·  multiple replicates", fc=DATA)
    arrow(ax, 5.5, 7.07, 5.5, 6.72)
    box(ax, 5.5, 6.2, 6.2, 1.0, "Feature engineering (within each case)",
        "Features: time gaps, GPS-derived speed & residual, acceleration,\nclimb rate, heading change, rolling deviation / std, raw telemetry", fc=PROC)
    arrow(ax, 5.5, 5.7, 5.5, 5.35)
    box(ax, 5.5, 4.85, 6.2, 0.95, "Case-level split (never by row)",
        "balanced: 80 / 20 train / holdout by case  ·  strong, subtle: external test", fc=PROC)

    xs = [1.9, 5.5, 9.1]
    specs = [
        ("Gradient Boosting", "HistGradientBoosting\nall training rows\nbalanced sample weights", GB),
        ("Linear SVM", "Impute → Scale → LinearSVC\nall training rows\nclass_weight = balanced", LIN),
        ("RBF SVM", "Impute → Scale → SVC(rbf)\nstratified subsample\nclass_weight = balanced", RBF),
    ]
    for x, (t, b, fc) in zip(xs, specs):
        arrow(ax, 5.5, 4.37, x, 3.72)
        box(ax, x, 3.15, 3.2, 1.1, t, b, fc=fc)
        arrow(ax, x, 2.6, x, 2.28)
    ax.text(xs[0], 2.1, "score = P(tampered)", ha="center", fontsize=8.8, color=MUTED)
    ax.text(xs[1], 2.1, "score = wᵀx + b", ha="center", fontsize=8.8, color=MUTED)
    ax.text(xs[2], 2.1, "score = Σ αᵢyᵢK(xᵢ,x) + b", ha="center", fontsize=8.8, color=MUTED)
    for x in xs:
        arrow(ax, x, 1.95, 5.5, 1.48)

    box(ax, 5.5, 1.0, 7.4, 0.95, "Threshold (max F1 on balanced holdout) → evaluation",
        "row, case and per-tamper-type metrics  ·  PR-AUC primary\nmean ± std across replicates", fc=OUT)
    out = FIGURES / "architecture_pipeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def gradient_boosting_figure(suffix=""):
    fig, ax = canvas(11, 7.4)
    ax.text(5.5, 7.15, "Gradient Boosting (HistGradientBoostingClassifier)", ha="center",
            fontsize=14, weight="bold", color=INK)

    box(ax, 1.35, 5.6, 2.2, 1.35, "Input x", "engineered features\nbinned into ≤255\nhistogram bins", fc=DATA)
    box(ax, 1.35, 3.55, 2.2, 1.3, "Initial model", "F₀ = log-odds of\nweighted class prior", fc=PROC)
    arrow(ax, 1.35, 4.92, 1.35, 4.2)

    # sequence of trees
    tx = [3.75, 5.85, 8.55]
    labels = ["Tree 1", "Tree 2", "Tree M"]
    for i, (x, lab) in enumerate(zip(tx, labels)):
        box(ax, x, 3.55, 1.65, 1.3, lab, "≤ 31 leaves\n≥ 20 rows / leaf", fc=GB)
    ax.text(7.2, 3.55, "· · ·", ha="center", va="center", fontsize=18, color=MUTED)
    arrow(ax, 2.45, 3.55, 2.92, 3.55)
    arrow(ax, 4.58, 3.55, 5.02, 3.55)
    arrow(ax, 6.68, 3.55, 6.95, 3.55)
    arrow(ax, 7.45, 3.55, 7.72, 3.55)

    # gradients feeding each tree
    for x in tx:
        box(ax, x, 5.6, 1.9, 1.25, "Fit to gradients", "gᵢ = pᵢ − yᵢ\nhᵢ = pᵢ(1 − pᵢ)", fc=PROC, body_size=8.5)
        arrow(ax, x, 4.97, x, 4.22)
    arrow(ax, 2.45, 5.6, 2.8, 5.6)
    ax.text(5.85, 6.55, "each tree corrects the errors of the ensemble built so far", ha="center",
            fontsize=9, style="italic", color=MUTED)

    # sum
    box(ax, 5.85, 1.75, 5.8, 1.15, "Additive update  (learning rate η = 0.1)",
        "Fₘ(x) = Fₘ₋₁(x) + η · treeₘ(x)\nleaf value = −Σg / (Σh + λ),   λ = 1.0 (L2)", fc=PROC, body_size=8.8)
    for x in tx:
        arrow(ax, x, 2.9, x, 2.33)
    box(ax, 9.9, 1.75, 1.7, 1.05, "Output", "p = σ(F_M(x))", fc=OUT)
    arrow(ax, 8.75, 1.75, 9.05, 1.75)

    box(ax, 1.35, 1.75, 2.2, 1.05, "Early stopping", "10% validation\nstop after 15 rounds\nno gain (max 300)", fc=PROC, body_size=8.3)
    arrow(ax, 2.45, 1.75, 2.95, 1.75, style="<|-", ls="--")
    ax.text(2.7, 2.0, "monitors", ha="center", fontsize=7.8, color=MUTED)

    out = FIGURES / f"architecture_gradient_boosting{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def boundary_panel(ax, clf, title, fc):
    X, y = make_moons(n_samples=220, noise=0.22, random_state=42)
    clf.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(-1.6, 2.6, 300), np.linspace(-1.2, 1.7, 300))
    zz = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    region = ["white", "#e6e6e6"] if BW else ["#dfeaf3", "#f8e1cf"]
    ax.contourf(xx, yy, zz, levels=[-100, 0, 100], colors=region, alpha=0.9)
    ax.contour(xx, yy, zz, levels=[-1, 0, 1], colors=[EDGE, INK, EDGE], linestyles=["--", "-", "--"], linewidths=[1, 1.8, 1])
    if BW:  # distinguish classes by marker shape, not colour
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=14, marker="o", facecolors="white", edgecolors="black",
                   linewidths=0.7, label="normal")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=16, marker="x", c="black", linewidths=0.8, label="tampered")
    else:
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=14, c="#176b87", label="normal", edgecolors="none")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=14, c="#d88732", label="tampered", edgecolors="none")
    if hasattr(clf, "support_vectors_"):
        sv = clf.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=60 if BW else 40, marker="s" if BW else "o", facecolors="none",
                   edgecolors=INK, linewidths=0.6, label="support vectors")
    ax.set_title(title, fontsize=10.5, weight="bold", color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(EDGE)
    ax.set_facecolor(fc)


def svm_figure(suffix=""):
    fig = plt.figure(figsize=(11, 10.2))
    top = fig.add_axes([0, 0.36, 1, 0.64])
    top.set_xlim(0, 11)
    top.set_ylim(0, 6.5)
    top.axis("off")
    top.text(5.5, 6.25, "Support Vector Machine: linear vs non-linear (RBF)", ha="center",
             fontsize=14, weight="bold", color=INK)

    for cx, fc, head, data, solver, score in [
        (2.85, LIN, "Linear SVM (LinearSVC)",
         "all training rows\nengineered features",
         "primal: min ½‖w‖² + C·Σ cᵢ·max(0, 1 − yᵢ(wᵀxᵢ + b))²\nsquared hinge loss, C = 1.0 (liblinear)",
         "f(x) = wᵀx + b"),
        (8.15, RBF, "Non-linear SVM (SVC, RBF kernel)",
         "label-stratified subsample\n(kernel SVM cost ≈ O(n²)–O(n³))",
         "dual: max Σαᵢ − ½ΣΣ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)\n0 ≤ αᵢ ≤ C·cᵢ,  C = 1.0 (libsvm)",
         "f(x) = Σ_SV αᵢyᵢ·K(xᵢ, x) + b"),
    ]:
        top.text(cx, 5.7, head, ha="center", fontsize=12, weight="bold", color=INK)
        box(top, cx, 5.0, 4.6, 0.8, "Training data", data, fc=DATA, body_size=8.5)
        arrow(top, cx, 4.6, cx, 4.33)
        box(top, cx, 3.95, 4.6, 0.7, "SimpleImputer(median) → StandardScaler", fc=PROC, title_size=9.5)
        arrow(top, cx, 3.6, cx, 3.33)
        if "RBF" in head:
            box(top, cx, 2.85, 4.6, 0.9, "RBF kernel", "K(x, x′) = exp(−γ‖x − x′‖²),  γ = 'scale' = 1 / (d · Var X)",
                fc=fc, body_size=8.5)
        else:
            box(top, cx, 2.85, 4.6, 0.9, "Linear kernel", "K(x, x′) = xᵀx′  (hyperplane in feature space)",
                fc=fc, body_size=8.5)
        arrow(top, cx, 2.4, cx, 2.13)
        box(top, cx, 1.55, 4.6, 1.1, "Optimisation (class_weight = balanced → cᵢ)", solver, fc=PROC, body_size=8.3)
        arrow(top, cx, 1.0, cx, 0.73)
        box(top, cx, 0.4, 4.6, 0.6, score, fc=OUT, title_size=10)

    a1 = fig.add_axes([0.07, 0.02, 0.39, 0.27])
    a2 = fig.add_axes([0.55, 0.02, 0.39, 0.27])
    boundary_panel(a1, LinearSVC(C=1.0, random_state=42), "Linear: straight boundary", "white")
    boundary_panel(a2, SVC(kernel="rbf", C=1.0, gamma="scale"), "RBF: curved boundary", "white")
    a2.legend(loc="lower right", fontsize=8, frameon=True)
    fig.text(0.5, 0.345, "Illustration on toy 2-D data (solid = decision boundary f(x)=0, dashed = margin f(x)=±1)",
             ha="center", fontsize=9, style="italic", color=MUTED)

    out = FIGURES / f"architecture_svm{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def flowchart_bw():
    """Black-and-white flowchart of the full methodology, for print."""
    def bw_box(ax, x, y, w, h, title, body="", dashed=False, tsize=10.5):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="white", ec="black", lw=1.2,
                                    linestyle=(0, (4, 2.5)) if dashed else "-"))
        if body:
            ax.text(x, y + 0.15, title, ha="center", va="center", fontsize=tsize, weight="bold", color="black")
            ax.text(x, y - 0.21, body, ha="center", va="center", fontsize=8.8, color="black", linespacing=1.35)
        else:
            ax.text(x, y, title, ha="center", va="center", fontsize=tsize, weight="bold", color="black")

    def bw_arrow(ax, x1, y1, x2, y2, dashed=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                     color="black", lw=1.2, shrinkA=0, shrinkB=0,
                                     linestyle=(0, (4, 2.5)) if dashed else "-"))

    fig, ax = canvas(15.4, 12.6)
    W, H, cx = 6.8, 0.85, 5.0
    steps = [
        ("Drone flight-log tampering dataset(s)",
         "tampering profiles (balanced / strong / subtle) · multiple replicates"),
        ("Load & audit", "parse timestamps · verify labels · keep sentinel rows"),
        ("Feature engineering (within each flight case)", "temporal / kinematic features"),
        ("Case-level split", "balanced: 80 / 20 train / holdout by case · strong & subtle: external test"),
    ]
    y = 11.55
    for i, (t, b) in enumerate(steps):
        bw_box(ax, cx, y, W, H, t, b)
        if i < len(steps) - 1:
            bw_arrow(ax, cx, y - H / 2, cx, y - 1.2 + H / 2)
        y -= 1.2
    split_bottom = y + 1.2 - H / 2

    ym = y - 0.35
    models_ = [
        (1.75, "Gradient Boosting", "HistGradientBoosting\nall training rows"),
        (5.0, "SVM – Linear", "impute → scale → LinearSVC\nall training rows"),
        (8.25, "SVM – RBF", "impute → scale → SVC (RBF)\nstratified subsample"),
    ]
    mh, mw = 1.15, 3.0
    for x, t, b in models_:
        bw_arrow(ax, cx, split_bottom, x, ym + mh / 2)
        ax.add_patch(FancyBboxPatch((x - mw / 2, ym - mh / 2), mw, mh, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="white", ec="black", lw=1.2))
        ax.text(x, ym + 0.25, t, ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(x, ym - 0.17, b, ha="center", va="center", fontsize=8.8, linespacing=1.3)

    y = ym - 1.55
    for x, _, _ in models_:
        bw_arrow(ax, x, ym - mh / 2, cx, y + H / 2)
    later = [
        ("Threshold selection", "max F1 on balanced holdout → applied unchanged to strong / subtle"),
        ("Evaluation", "row, case & per-tamper-type · PR-AUC (primary), ROC-AUC, precision, recall, F1, MCC"),
        ("Robustness", "repeat across replicates (mean ± std) · RBF learning curve"),
        ("Recommended model", "Gradient Boosting"),
    ]
    for i, (t, b) in enumerate(later):
        bw_box(ax, cx, y, W + 1.4 if t == "Evaluation" else W, H, t, b)
        if i < len(later) - 1:
            bw_arrow(ax, cx, y - H / 2, cx, y - 1.2 + H / 2)
        y -= 1.2

    # --- side branch: real, unlabelled flight logs (inference only) -------------
    rx, rw = 12.85, 4.7
    ax.text(rx, 12.28, "external check (not used for training)", ha="center", fontsize=9.5,
            style="italic", color="black")
    branch = [
        (11.55, 1.05, "Real flight logs", "ground-station exports\nno ground-truth labels"),
        (10.05, 1.05, "Schema adapter", "map export columns →\ninternal schema"),
        (8.55, 0.95, "Same feature engineering", "temporal / kinematic features"),
        (4.85, 1.05, "Score with saved model", "inference only\n(Gradient Boosting + threshold)"),
        (2.9, 1.05, "Qualitative inspection", "flag rates, ground-track plots\nno metrics — unlabelled"),
    ]
    for i, (ry, rh, t, b) in enumerate(branch):
        bw_box(ax, rx, ry, rw, rh, t, b, dashed=True, tsize=10)
        if i < len(branch) - 1:
            ny, nh = branch[i + 1][0], branch[i + 1][1]
            bw_arrow(ax, rx, ry - rh / 2, rx, ny + nh / 2, dashed=True)
    # the fitted model and its tuned threshold cross over into the scoring step
    sy = branch[3][0]
    bw_arrow(ax, cx + W / 2, sy, rx - rw / 2, sy, dashed=True)
    ax.text((cx + W / 2 + rx - rw / 2) / 2, sy + 0.12, "saved model\n+ threshold", ha="center",
            va="bottom", fontsize=8.2, color="black", linespacing=1.2)

    out = FIGURES / "flowchart_bw.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    outputs = [pipeline_figure(), gradient_boosting_figure(), svm_figure(), flowchart_bw()]
    set_bw()
    outputs += [gradient_boosting_figure("_bw"), svm_figure("_bw")]
    for f in outputs:
        print("wrote", f)


if __name__ == "__main__":
    main()
