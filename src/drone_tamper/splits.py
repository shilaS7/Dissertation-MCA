"""Case-level 80/20 split, stratified by whether a case contains any tampered row."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SEED


def case_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    case_label = df.groupby("case_id")["label"].max().rename("case_label").reset_index()
    tr_cases, va_cases = train_test_split(
        case_label, test_size=test_size, stratify=case_label["case_label"], random_state=seed,
    )
    tr_mask = df["case_id"].isin(tr_cases["case_id"]).to_numpy()
    va_mask = df["case_id"].isin(va_cases["case_id"]).to_numpy()
    return tr_mask, va_mask, tr_cases, va_cases
