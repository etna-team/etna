import math
from typing import TYPE_CHECKING
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from etna.datasets import TSDataset


def _prepare_axes(
    num_plots: int, columns_num: int, figsize: Tuple[int, int], set_grid: bool = True
) -> Tuple[matplotlib.figure.Figure, Sequence[matplotlib.axes.Axes]]:
    """Prepare axes according to segments, figure size and number of columns."""
    columns_num = min(columns_num, num_plots)
    rows_num = math.ceil(num_plots / columns_num)

    figsize = (figsize[0] * columns_num, figsize[1] * rows_num)
    fig, ax = plt.subplots(rows_num, columns_num, figsize=figsize, constrained_layout=True)
    ax = np.array([ax]).ravel()

    if set_grid:
        for cur_ax in ax:
            cur_ax.grid()

    return fig, ax


def _get_borders_ts(
    ts: "TSDataset", start: Optional[Union[pd.Timestamp, int, str]], end: Optional[Union[pd.Timestamp, int, str]]
) -> Tuple[str, str]:
    """Get start and end parameters according to given TSDataset."""
    from etna.datasets.utils import _check_timestamp_param

    start = _check_timestamp_param(param=start, param_name="start", freq=ts.freq)
    end = _check_timestamp_param(param=end, param_name="end", freq=ts.freq)

    if start is not None:
        start_idx = ts.timestamps.get_loc(start)
    else:
        start_idx = 0

    if end is not None:
        end_idx = ts.timestamps.get_loc(end)
    else:
        end_idx = ts.size()[0] - 1

    if start_idx >= end_idx:
        raise ValueError("Parameter 'end' must be greater than 'start'!")

    return ts.timestamps[start_idx], ts.timestamps[end_idx]
