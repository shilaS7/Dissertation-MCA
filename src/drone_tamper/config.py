"""Shared constants and project paths."""

from pathlib import Path


def find_root(start: Path) -> Path:
    """Resolve the project root no matter where the interpreter was launched from."""
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start


ROOT = find_root(Path.cwd())
DATA_RAW = ROOT / "data" / "raw"
DATA_INDEX = ROOT / "data" / "index"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SEED = 42
PROFILES = ("balanced", "strong", "subtle")
REPLICATES = (0, 1, 2, 3)
RBF_SAMPLE = 20000
RBF_LC_SIZES = (5000, 10000, 20000, 40000)

TAMPER_TYPES = (
    "altitude_spike", "combined", "coordinate_jump", "deletion_gap",
    "heading_inconsistency", "injection", "precision_rounding",
    "speed_inconsistency", "timestamp_drift",
)

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


def ensure_dirs() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
