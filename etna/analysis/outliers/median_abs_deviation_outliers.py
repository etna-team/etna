import math
import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional

import numpy as np
import pandas as pd

if typing.TYPE_CHECKING:
    from etna.datasets import TSDataset


def get_anomalies_mad(
    ts: TSDataset,
    in_column: str = "target",
    window_size: int = 10,
    stride: int = 1,
    mad_scale: float = 3,
    trend: bool = False,
    seasonality: bool = False,
    period: Optional[int] = None,
    stl_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[List[pd.Timestamp], List[int], pd.Series]]:
    """

    Parameters
    ----------

    Returns
    -------
    result: Dict
       dict of outliers in format {segment: [outliers_timestamps]}
    """
    pass