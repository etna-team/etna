from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

from etna.core import BaseMixin

TTimestampInterval = Tuple[Union[pd.Timestamp, int], Union[pd.Timestamp, int]]
TDetrendModel = Type[RegressorMixin]


class BaseChangePointsModelAdapter(BaseMixin, ABC):
    """BaseChangePointsModelAdapter is the base class for change point models adapters."""

    @abstractmethod
    def get_change_points(self, df: pd.DataFrame, in_column: str) -> List[pd.Timestamp]:
        """Find change points within one segment.

        Parameters
        ----------
        df:
            dataframe indexed with timestamp
        in_column:
            name of column to get change points

        Returns
        -------
        change points:
            change point timestamps
        """
        pass

    @staticmethod
    def _build_intervals(change_points: List[Union[pd.Timestamp, int]], dtype: Any) -> List[TTimestampInterval]:
        """Create list of stable intervals from list of change points."""
        if pd.api.types.is_integer_dtype(dtype):
            change_points.extend([np.iinfo(dtype).min, np.iinfo(dtype).max])
        else:
            change_points.extend([pd.Timestamp.min, pd.Timestamp.max])

        change_points = sorted(change_points)
        intervals = list(zip(change_points[:-1], change_points[1:]))
        return intervals

    def get_change_points_intervals(self, df: pd.DataFrame, in_column: str) -> List[TTimestampInterval]:
        """Find change point intervals in given dataframe and column.

        Parameters
        ----------
        df:
            dataframe indexed with timestamp
        in_column:
            name of column to get change points

        Returns
        -------
        :
            change points intervals
        """
        change_points = self.get_change_points(df=df, in_column=in_column)
        intervals = self._build_intervals(change_points=change_points, dtype=df.index.dtype)
        return intervals
