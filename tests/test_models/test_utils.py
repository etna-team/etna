import pandas as pd
import pytest

from etna.models.utils import select_observations


@pytest.fixture()
def df_without_timestamp():
    df = pd.DataFrame({"target": list(range(5))})
    return df


@pytest.mark.parametrize(
    "timestamps, freq, start, end, periods",
    [
        (pd.to_datetime(pd.Series(["2020-02-01", "2020-02-03"])), "D", "2020-02-01", None, 5),
        (pd.to_datetime(pd.Series(["2020-02-01", "2020-02-03"])), "D", "2020-02-01", "2020-02-05", None),
        (pd.to_datetime(pd.Series(["2020-02-01", "2020-02-03"])), "D", None, "2020-02-05", 5),
        (pd.to_datetime(pd.Series(["2020-02-01"])), "D", "2020-02-01", None, 5),
        (pd.Series([6, 7]), None, 5, None, 5),
        (pd.Series([6, 7]), None, 5, 9, None),
        (pd.Series([6, 7]), None, None, 9, 5),
        (pd.Series([6]), None, 5, None, 5),
    ],
)
def test_select_observations(timestamps, freq, start, end, periods, df_without_timestamp):
    selected_df = select_observations(
        df=df_without_timestamp, timestamps=timestamps, freq=freq, start=start, end=end, periods=periods
    )
    assert len(selected_df) == len(timestamps)


@pytest.mark.parametrize(
    "timestamps, freq, start, end, periods",
    [
        (pd.to_datetime(pd.Series(["2020-02-01", "2020-02-03"])), "D", None, None, None),
        (pd.to_datetime(pd.Series(["2020-02-01", "2020-02-03"])), "D", "2020-02-01", None, None),
        (pd.to_datetime(pd.Series(["2020-02-01", "2020-02-03"])), "D", None, "2020-02-05", None),
        (pd.to_datetime(pd.Series(["2020-02-01", "2020-02-03"])), "D", None, None, 5),
        (pd.to_datetime(pd.Series(["2020-02-01", "2020-02-03"])), "D", "2020-02-01", "2020-02-05", 5),
        (pd.Series([6, 7]), None, None, None, None),
        (pd.Series([6, 7]), None, 5, None, None),
        (pd.Series([6, 7]), None, None, 9, None),
        (pd.Series([6, 7]), None, None, None, 5),
        (pd.Series([6, 7]), None, 5, 9, 5),
    ],
)
def test_select_observations_fail(timestamps, freq, start, end, periods, df_without_timestamp):
    with pytest.raises(ValueError, match="Of the three parameters: start, end, periods, exactly two must be specified"):
        _ = select_observations(
            df=df_without_timestamp, timestamps=timestamps, freq=freq, start=start, end=end, periods=periods
        )
