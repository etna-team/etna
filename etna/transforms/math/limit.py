from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from etna.transforms.base import ReversibleTransform


class LimitTransform(ReversibleTransform):
    """LimitTransform limits values of some feature between the borders (``lower_bound`` - ``tol``, ``upper_bound`` + ``tol``).

    * If both ``lower_bound`` and ``upper_bound`` are not set there is no transformation

    * If both ``lower_bound`` and ``upper_bound`` are set apply

    .. math::
        y = \\log(\\frac{x-(a-tol)}{(b+tol)-x}),

    * If ``lower_bound`` is set and ``upper_bound`` is not set apply

    .. math::
        y = \\log (x-(a-tol))

    * If ``lower_bound`` is not set and ``upper_bound`` is set apply

    .. math::
        y = \\log ((b+tol)-x)

    where :math:`x` is feature, :math:`a` is lower bound, :math:`b` is upper bound, :math:`tol` is offset.

    For more details visit https://datasciencestunt.com/time-series-forecasting-within-limits/ .

    Applying transform to ``in_column`` of dtype int could lead to unexpected behaviour
    in different ``pandas`` versions. Try converting ``in_column`` to float dtype.
    """

    def __init__(
        self,
        in_column: str,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        tol: float = 1e-10,
    ):
        """
        Init LimitTransform.

        Parameters
        ----------
        in_column:
            column to apply transform.
        lower_bound:
            lower bound for the value of the column, inclusive.
        upper_bound:
            upper bound for the value of the column, inclusive.
        tol:
            offset from the bounds used to calculate transformed values.

        Raises
        ------
        ValueError:
            Some ``in_column`` features are less than ``lower_bound`` or greater than ``upper_bound``.
        """
        super().__init__(required_features=[in_column])
        self.in_column = in_column
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.tol = tol

        self._lower_bound: Optional[float] = self.lower_bound - self.tol if self.lower_bound is not None else None
        self._upper_bound: Optional[float] = self.upper_bound + self.tol if self.upper_bound is not None else None

    def _fit(self, df: pd.DataFrame):
        """Fit method does nothing and is kept for compatibility.

        Parameters
        ----------
        df:
            dataframe with data.
        """
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaled logit transform to the dataset.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        :
            transformed dataframe

        """
        if (self.lower_bound is not None and (df < self.lower_bound).any().any()) or (
            self.upper_bound is not None and (df > self.upper_bound).any().any()
        ):
            left_border = self.lower_bound
            right_border = self.upper_bound
            if self.lower_bound is None:
                left_border = np.NINF
            if self.upper_bound is None:
                right_border = np.inf
            raise ValueError(f"Detected values out [{left_border}, {right_border}]")

        # TODO: https://github.com/etna-team/etna/issues/66
        if self._lower_bound is None and self._upper_bound is None:
            transformed_features = df
        elif self._lower_bound is not None and self._upper_bound is None:
            transformed_features = np.log(df - self._lower_bound)
        elif self._lower_bound is None and self._upper_bound is not None:
            transformed_features = np.log(self._upper_bound - df)
        else:
            transformed_features = np.log((df - self._lower_bound) / (self._upper_bound - df))

        return transformed_features

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaled logit reverse transform to the dataset.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        :
            transformed series
        """
        # TODO: https://github.com/etna-team/etna/issues/66
        if self._lower_bound is None and self._upper_bound is None:
            transformed_features = df
        elif self._lower_bound is not None and self._upper_bound is None:
            transformed_features = np.exp(df) + self._lower_bound
        elif self._lower_bound is None and self._upper_bound is not None:
            transformed_features = self._upper_bound - np.exp(df)
        else:
            exp_df = np.exp(df)
            transformed_features = ((self._upper_bound - self._lower_bound) * exp_df) / (1 + exp_df) + self._lower_bound  # type: ignore
        return transformed_features

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return []


__all__ = ["LimitTransform"]
