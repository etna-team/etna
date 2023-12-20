import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df


@pytest.fixture
def ts_with_external_timestamp() -> TSDataset:
    df = generate_ar_df(periods=100, start_time=10, n_segments=2, freq=None)
    df_wide = TSDataset.to_dataset(df)
    df_exog = generate_ar_df(periods=100, start_time=10, n_segments=2, freq=None)
    df_exog["target"] = pd.date_range(start="2020-01-01", periods=100).tolist() * 2
    df_exog_wide = TSDataset.to_dataset(df_exog)
    df_exog_wide.rename(columns={"target": "external_timestamp"}, level="feature", inplace=True)
    ts = TSDataset(df=df_wide.iloc[:-10], df_exog=df_exog_wide, known_future="all", freq=None)
    return ts
