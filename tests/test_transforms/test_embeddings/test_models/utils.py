import numpy as np
import pandas as pd


def cut_nan_timestamps(df: pd.DataFrame) -> np.ndarray:
    """Cut NaN timestamps. Is used when encoding full series."""
    last_timestamp = max(np.where(~df.isna().all(axis=1))[0])
    df = df[: last_timestamp + 1]
    return df
