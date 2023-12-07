import pandas as pd
import pytest

from etna.models.utils import select_observations


@pytest.fixture()
def df_without_timestamp():
    df = pd.DataFrame({"target": list(range(5))})
    return df


@pytest.mark.parametrize(
    "timestamps",
    (
        pd.to_datetime(pd.Series(["2020-02-01", "2020-02-03"])),
        pd.to_datetime(pd.Series(["2020-02-01"])),
    ),
)
def test_select_observations_without_timestamp(df_without_timestamp, timestamps):
    selected_df = select_observations(
        df=df_without_timestamp, timestamps=timestamps, freq="D", start="2020-02-01", periods=5
    )
    assert len(selected_df) == len(timestamps)
