from enum import Enum
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.distributions import IntDistribution
from etna.transforms.base import IrreversibleTransform


class ImputerMode(str, Enum):
    """Enum for different imputation strategy."""

    binary = "binary"
    distance = "distance"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Supported modes: {', '.join([repr(m.value) for m in cls])}"
        )


class EventTransform(IrreversibleTransform):
    """EventTransform marks days before and after event depending on ``mode``.
     It creates two columns for future and past.

    * In `'binary'` mode shows whether there will be or were events regarding current date.

    * In `'distance'` mode shows distance to the previous and future events regarding current date. Computed as :math:`1 / x`, where x is a distance to the nearest event.
    """

    def __init__(
        self, in_column: str, out_column: str, n_pre: int = 1, n_post: int = 1, mode: str = ImputerMode.binary
    ):
        """
        Init EventTransform.

        Parameters
        ----------
        in_column:
            binary column with event indicator.
        out_column:
            base for creating out columns names for future and past - '{out_column}_pre' and '{out_column}_post'
        n_pre:
            number of days before the event to react.
        n_post:
            number of days after the event to react.
        mode:
            mode of marking events:
            - `'binary'`: whether there will be or were events regarding current date in binary type;
            - `'distance'`: distance to the previous and future events regarding current date;

        Raises
        ------
        ValueError:
            Some ``in_column`` features are not binary.
        ValueError:
            ``n_pre`` or ``n_post`` values are less than one.
        NotImplementedError:
            Given ``mode`` value is not supported.
        """
        super().__init__(required_features=[in_column])
        self.in_column = in_column
        self.out_column = out_column
        self.n_pre = n_pre
        self.n_post = n_post
        self.mode = mode
        self.in_column_regressor: Optional[bool] = None
        self._mode = ImputerMode(mode)

    def fit(self, ts: TSDataset) -> "EventTransform":
        """Fit the transform."""
        self.in_column_regressor = self.in_column in ts.regressors
        super().fit(ts)
        return self

    def _fit(self, df: pd.DataFrame):
        """Fit method does nothing and is kept for compatibility.

        Parameters
        ----------
        df:
            dataframe with data.
        """
        pass

    def _compute_event_column(self, df: pd.DataFrame, column: str, max_distance: int) -> pd.DataFrame:
        """Compute event column."""
        indexes = df.copy()
        indexes[:] = np.repeat((np.arange(len(indexes)) + 1).reshape(-1, 1), len(indexes.columns), axis=1)

        col = indexes.copy()
        col.mask(df != 1, None, inplace=True)
        if column == "prev":
            col = col.bfill().fillna(indexes)
            col = col - indexes
        else:
            col = col.ffill().fillna(indexes)
            col = indexes - col
        distance = 1 if self.mode == "binary" else 1 / col
        col.mask(col > max_distance, 0, inplace=True)
        col.mask((col >= 1) & (col <= max_distance), distance, inplace=True)

        col.rename(columns={self.in_column: f"{self.out_column}_{column}"}, inplace=True, level="feature")
        return col

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add marked days before and after event to dataset.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        :
            transformed dataframe

        """
        if set(df.values.reshape(-1)) != {0, 1}:
            raise ValueError("Input columns must be binary")
        if self.n_pre < 1 or self.n_post < 1:
            raise ValueError(f"`n_pre` and `n_post` must be greater than zero, given {self.n_pre} and {self.n_post}")

        prev = self._compute_event_column(df, column="prev", max_distance=self.n_pre)
        post = self._compute_event_column(df, column="post", max_distance=self.n_post)

        df = pd.concat([df, prev, post], axis=1)

        return df

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        if self.in_column_regressor is None:
            raise ValueError("Fit the transform to get the correct regressors info!")
        return [self.out_column + "_pre", self.out_column + "_post"] if self.in_column_regressor else []

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``n_pre``, ``n_post``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "n_pre": IntDistribution(low=1, high=self.n_pre),
            "n_post": IntDistribution(low=1, high=self.n_post),
            "mode": CategoricalDistribution(["binary", "distance"]),
        }


__all__ = ["EventTransform"]
