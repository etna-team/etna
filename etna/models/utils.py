from typing import Optional
from typing import Union

import pandas as pd

from etna.datasets.utils import determine_freq  # noqa: F401
from etna.datasets.utils import determine_num_steps  # noqa: F401


def select_observations(
    df: pd.DataFrame,
    timestamps: pd.Series,
    freq: str,
    start: Optional[Union[pd.Timestamp, str]] = None,
    end: Optional[Union[pd.Timestamp, str]] = None,
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
        pandas frequency string
    start:
        start of the timeline
    end:
        end of the timeline
    periods:
        number of periods in the timeline

    Returns
    -------
    :
        dataframe with selected observations
    """
    df["timestamp"] = pd.date_range(start=start, end=end, periods=periods, freq=freq)

    if not (set(timestamps) <= set(df["timestamp"])):
        raise ValueError("Some timestamps do not lie inside the timeline of the provided dataframe.")

    observations = df.set_index("timestamp")
    observations = observations.loc[timestamps]
    observations.reset_index(drop=True, inplace=True)
    return observations
