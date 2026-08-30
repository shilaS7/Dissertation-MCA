"""Model builders — Gradient Boosting, linear SVM, RBF SVM."""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from .config import SEED


def build_hgb(seed: int = SEED) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=15, random_state=seed,
    )


def build_linear_svc(seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("svc", LinearSVC(class_weight="balanced", dual="auto", max_iter=3000, random_state=seed)),
    ])


def build_rbf_svc(seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                    cache_size=500, random_state=seed)),
    ])
