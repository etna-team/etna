from typing import List
from typing import Optional
from typing import Dict
import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.transforms.base import IrreversibleTransform
from etna.distributions import BaseDistribution, IntDistribution, CategoricalDistribution


class EventTransform(IrreversibleTransform):
    """EventTransform marks days before and after event

    In ``binary`` mode shows whether there will be or were events regarding current date.

    In ``distance`` mode shows distance to the previous and future events regarding current date.
    Computed as :math:`1 / x`, where x is a distance to the nearest event.
    """

    def __init__(
        self,
        in_column: str,
        out_column: str,
        n_pre: int = 1,
        n_post: int = 1,
        mode: str = 'binary'
    ):
        """
        Init EventTransform.

        Parameters
        ----------
        in_column:
            binary column with event indicator.
        out_column:
            base for creating out columns names.
        n_pre:
            number of days before the event to react.
        n_post:
            number of days after the event to react.
        mode: {'binary', 'distance'}, default='binary'
            Specify mode of marking events:

            - `'binary'`: whether there will be or were events regarding current date in binary type;
            - `'distance'`: distance to the previous and future events regarding current date;

        Raises
        ------
        TypeError:
            Type of `n_pre` or `n_post` is different from `int`.
        TypeError:
            Value of `mode` is not in ['binary', 'distance'].
        """
        if not isinstance(n_pre, int) or not isinstance(n_post, int):
            raise TypeError('`n_pre` and `n_post` must have type `int`')
        if mode not in ['binary', 'distance']:
            raise TypeError(f'{type(self).__name__} supports only modes in [\'binary\', \'distance\'], got {mode}.')

        super().__init__(required_features=[in_column])
        self.in_column = in_column
        self.out_column = out_column
        self.n_pre = n_pre
        self.n_post = n_post
        self.mode = mode
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

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mark days before and after event.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        :
            transformed dataframe

        """
        indexes = df.copy()
        indexes[:] = np.repeat((np.arange(len(indexes)) + 1).reshape(-1, 1), len(indexes.columns), axis=1)

        prev = df.copy()
        prev.mask(prev != 1, None, inplace=True)
        prev = prev * indexes
        prev = prev.bfill().fillna(indexes)
        prev = prev - indexes
        if self.mode == 'binary':
            prev.mask((prev >= 1) & (prev <= self.n_pre), 1, inplace=True)
            prev.mask(prev > self.n_pre, 0, inplace=True)
        else:
            prev.mask(prev > self.n_pre, 0, inplace=True)
            prev.mask((prev >= 1) & (prev <= self.n_pre), 1 / prev, inplace=True)
        prev.rename(columns={self.in_column: f'{self.out_column}_prev'}, inplace=True, level="feature")

        post = df.copy()
        post.mask(post != 1, None, inplace=True)
        post = post * indexes
        post = post.ffill().fillna(indexes)
        post = indexes - post
        if self.mode == 'binary':
            post.mask((post >= 1) & (post <= self.n_post), 1, inplace=True)
            post.mask(post > self.n_post, 0, inplace=True)
        else:
            post.mask(post > self.n_post, 0, inplace=True)
            post.mask((post >= 1) & (post <= self.n_post), 1 / post, inplace=True)
        post.rename(columns={self.in_column: f'{self.out_column}_post'}, inplace=True, level="feature")

        df = pd.concat([df, prev, post], axis=1)

        return df

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        if self.in_column_regressor is None:
            raise ValueError("Fit the transform to get the correct regressors info!")
        return [self.out_column + '_pre', self.out_column + '_post'] if self.in_column_regressor else []

    def _params_to_tune(self) -> Dict[str, BaseDistribution]:
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
            "mode": CategoricalDistribution(['binary', 'distance'])
        }


__all__ = ["EventTransform"]
