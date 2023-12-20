from typing import Optional
from typing import Union
from typing import cast

import numpy as np
import pandas as pd

from etna.datasets.utils import determine_freq  # noqa: F401
from etna.datasets.utils import determine_num_steps  # noqa: F401


def select_observations(
    df: pd.DataFrame,
    timestamps: pd.Series,
    freq: Optional[str],
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
    if freq is None:
        start = cast(int, start)
        end = cast(int, end)
        two_nans = (
            (start is None and end is None) or (start is None and periods is None) or (end is None and periods is None)
        )
        zero_nans = (start is not None) and (end is not None) and (periods is not None)
        if two_nans or zero_nans:
            raise ValueError("Of the three parameters: start, end, periods, exactly two must be specified")
        elif start is None:
            start = end - periods + 1
        elif end is None:
            end = start + periods - 1
        df["timestamp"] = np.arange(start, end + 1)
    else:
        df["timestamp"] = pd.date_range(start=start, end=end, periods=periods, freq=freq)

    if not (set(timestamps) <= set(df["timestamp"])):
        raise ValueError("Some timestamps do not lie inside the timeline of the provided dataframe.")

    observations = df.set_index("timestamp")
    observations = observations.loc[timestamps]
    observations.reset_index(drop=True, inplace=True)
    return observations
