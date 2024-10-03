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

    Examples
    --------
    >>> from copy import deepcopy
    >>> import numpy as np
    >>> import pandas as pd
    >>> from etna.datasets import generate_const_df
    >>> from etna.datasets import TSDataset
    >>> from etna.transforms import EventTransform
    >>>
    >>> df = generate_const_df(start_time="2020-01-01", periods=5, freq="D", scale=1, n_segments=1)
    >>> df_exog = generate_const_df(start_time="2020-01-01", periods=10, freq="D", scale=1, n_segments=1)
    >>> df_exog.rename(columns={"target": "holiday"}, inplace=True)
    >>> df_exog["holiday"] = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 0])
    >>> df = TSDataset.to_dataset(df)
    >>> df_exog = TSDataset.to_dataset(df_exog)
    >>> ts = TSDataset(df, freq="D", df_exog=df_exog, known_future="all")
    >>> transform = EventTransform(in_column='holiday', out_column='holiday', n_pre=1, n_post=1)
    >>> transform.fit_transform(deepcopy(ts))
    segment    segment_0
    feature      holiday holiday_post holiday_pre target
    timestamp
    2020-01-01         0          0.0          0.0    1.0
    2020-01-02         0          0.0          1.0    1.0
    2020-01-03         1          0.0          0.0    1.0
    2020-01-04         0          1.0          0.0    1.0
    2020-01-05         0          0.0          0.0    1.0

    >>> transform = EventTransform(in_column='holiday', out_column='holiday', n_pre=2, n_post=2, mode='distance')
    >>> transform.fit_transform(deepcopy(ts))
    segment    segment_0
    feature      holiday holiday_post holiday_pre target
    timestamp
    2020-01-01         0          0.0          0.5    1.0
    2020-01-02         0          0.0          1.0    1.0
    2020-01-03         1          0.0          0.0    1.0
    2020-01-04         0          1.0          0.0    1.0
    2020-01-05         0          0.5          0.0    1.0
    """

    def __init__(self, in_column: str, out_column: str, n_pre: int, n_post: int, mode: str = ImputerMode.binary):
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
        if n_pre < 1 or n_post < 1:
            raise ValueError(f"`n_pre` and `n_post` must be greater than zero, given {n_pre} and {n_post}")
        super().__init__(required_features=[in_column])
        self.in_column = in_column
        self.out_column = out_column
        self.n_pre = n_pre
        self.n_post = n_post
        self.mode = ImputerMode(mode)
        self.in_column_regressor: Optional[bool] = None

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
        col = (col.bfill() if column == "pre" else col.ffill()).fillna(indexes)
        col = (col - indexes).abs()
        distance = 1 if self.mode == "binary" else 1 / col
        col.mask(col > max_distance, 0, inplace=True)
        col = col.mask((col >= 1) & (col <= max_distance), distance).astype(float)

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
        if not set(df.values.reshape(-1)).issubset({0, 1}):
            raise ValueError("Input columns must be binary")

        pre = self._compute_event_column(df, column="pre", max_distance=self.n_pre)
        post = self._compute_event_column(df, column="post", max_distance=self.n_post)

        df = pd.concat([df, pre, post], axis=1)

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
