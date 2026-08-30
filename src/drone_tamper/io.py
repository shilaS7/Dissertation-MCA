"""CSV loading for one profile/replicate."""

from pathlib import Path

import pandas as pd

from .config import DTYPES


def load_profile(raw_dir: Path, profile: str) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / f"{profile}.csv", dtype=DTYPES)
    # pandas 3.0 needs an explicit ISO8601 format for these 6-digit-microsecond strings
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", errors="coerce")
    return df
