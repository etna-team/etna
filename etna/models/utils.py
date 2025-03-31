from typing import Optional
from typing import Union

import pandas as pd

from etna.datasets.utils import determine_freq  # noqa: F401
from etna.datasets.utils import determine_num_steps  # noqa: F401
from etna.datasets.utils import timestamp_range


def select_observations(
    df: pd.DataFrame,
    timestamps: pd.Series,
    freq: Union[pd.offsets.BaseOffset, str, None] = None,
    start: Optional[Union[pd.Timestamp, int, str]] = None,
    end: Optional[Union[pd.Timestamp, int, str]] = None,
    periods: Optional[int] = None,
) -> pd.DataFrame:
    """Select observations from dataframe with known timeline.

    Parameters
    ----------
    df:
        dataframe with known timeline
    timestamps:
        series of timestamps to select
    freq:
        frequency of timestamp in df, possible values:

        - :py:class:`pandas.offsets.BaseOffset` object for datetime timestamp

        - `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
          for datetime timestamp

        - None for integer timestamp

    start:
        start of the timeline
    end:
        end of the timeline (included)
    periods:
        number of periods in the timeline

    Returns
    -------
    :
        dataframe with selected observations

    Raises
    ------
    ValueError:
        Of the three parameters: start, end, periods, exactly two must be specified
    """
    df["timestamp"] = timestamp_range(start=start, end=end, periods=periods, freq=freq)

    if not (set(timestamps) <= set(df["timestamp"])):
        raise ValueError("Some timestamps do not lie inside the timeline of the provided dataframe.")

    observations = df.set_index("timestamp")
    observations = observations.loc[timestamps]
    observations.reset_index(drop=True, inplace=True)
    return observations
