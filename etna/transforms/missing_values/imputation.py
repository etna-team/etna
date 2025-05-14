from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import bottleneck as bn
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.distributions import IntDistribution
from etna.transforms import LagTransform
from etna.transforms import MeanTransform
from etna.transforms.base import ReversibleTransform
from etna.transforms.utils import check_new_segments


class SimpleImputerSubsegment(SimpleImputer):
    def __init__(
        self,
        missing_values=np.nan,
        strategy="mean",
        fill_value=None,
        copy=True,
        add_indicator=False,
    ):
        super().__init__(
            missing_values=missing_values,
            strategy=strategy,
            fill_value=fill_value,
            copy=copy,
            add_indicator=add_indicator,
        )
        self._segment_to_index: Optional[Dict[str, int]] = None

    def fit(self, X, y=None):  # noqa: N803
        X.sort_index(axis=1, inplace=True)
        segments = sorted(X.columns.get_level_values("segment").unique())
        self._segment_to_index = {segment: i for i, segment in enumerate(segments)}
        super().fit(X.values)

    def transform(self, X):  # noqa: N803
        X.sort_index(axis=1, inplace=True)
        segments = X.columns.get_level_values("segment").unique()
        old_statistics = self.statistics_.copy()
        self.n_features_in_ = len(segments)
        self.statistics_ = self.statistics_[[self._segment_to_index[segment] for segment in segments]]
        try:
            super().transform(X.values)  # noqa: N803
        finally:
            self.n_features_in_ = len(old_statistics)
            self.statistics_ = old_statistics


class ImputerMode(str, Enum):
    """Enum for different imputation strategy."""

    mean = "mean"
    running_mean = "running_mean"
    forward_fill = "forward_fill"
    seasonal = "seasonal"
    seasonal_nonautoreg = "seasonal_nonautoreg"
    constant = "constant"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Supported strategies: {', '.join([repr(m.value) for m in cls])}"
        )


class TimeSeriesImputerTransform(ReversibleTransform):
    """Transform to fill NaNs in series of a given dataframe.

    - It is assumed that given series begins with first non NaN value.

    - This transform can't fill NaNs in the future, only on train data.

    - This transform can't fill NaNs if all values are NaNs. In this case exception is raised.

    Imputation with the values that differ from initial ``in_column`` column dtype
    could lead to unexpected behaviour in different ``pandas`` versions.

    Warning
    -------
    This transform can suffer from look-ahead bias in 'mean' mode. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: str = "target",
        strategy: str = ImputerMode.constant,
        window: int = -1,
        seasonality: int = 1,
        default_value: Optional[float] = None,
        constant_value: float = 0,
    ):
        """
        Create instance of TimeSeriesImputerTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        strategy:
            filling value in missing timestamps:

            - If "mean", then replace missing dates using the mean in fit stage.

            - If "running_mean" then replace missing dates using mean of subset of data

            - If "forward_fill" then replace missing dates using last existing value

            - If "seasonal" then replace missing dates using seasonal moving average in autoregressive manner,
             point are imputed one by one in time order, already imputed points are used to impute the next points

            - If "seasonal_nonautoreg" then replace missing dates using seasonal moving average of existing values,
             all points are imputed using only existing points

            - If "constant" then replace missing dates using constant value.

        window:
            In case of moving average and seasonality.

            * If ``window=-1`` all previous dates are taken in account

            * Otherwise only window previous dates

        seasonality:
            the length of the seasonality
        default_value:
            value which will be used to impute the NaNs left after applying the imputer with the chosen strategy
        constant_value:
            value to fill gaps in "constant" strategy

        Raises
        ------
        ValueError:
            if incorrect strategy given
        """
        super().__init__(required_features=[in_column])
        self.in_column = in_column
        self.strategy = strategy
        self.window = window
        self.seasonality = seasonality
        self.default_value = default_value
        self.constant_value = constant_value
        self._strategy = ImputerMode(strategy)
        self._fit_segments: Optional[Sequence[str]] = None
        self._nans_to_impute_mask: Optional[pd.DataFrame] = None
        self._mean_imputer: Optional[SimpleImputerSubsegment] = None

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return []

    def _fit(self, df: pd.DataFrame):
        """Fit the transform.

        Parameters
        ----------
        df:
            Dataframe in etna wide format.
        """
        if df.isna().all().any():
            raise ValueError("Series hasn't non NaN values which means it is empty and can't be filled.")

        self._fit_segments = sorted(set(df.columns.get_level_values("segment")))

        if self._strategy is ImputerMode.mean:
            self._mean_imputer = SimpleImputerSubsegment(strategy="mean", copy=False)
            self._mean_imputer.fit(df)

        _beginning_nans_mask = df.ffill().isna()
        self._nans_to_impute_mask = df.isna() & (~_beginning_nans_mask)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe.

        Parameters
        ----------
        df:
            Dataframe in etna wide format.

        Returns
        -------
        :
            Transformed Dataframe in etna wide format.
        """
        if self._fit_segments is None or self._nans_to_impute_mask is None:
            raise ValueError("Transform is not fitted!")

        segments = df.columns.get_level_values("segment").unique()
        check_new_segments(transform_segments=segments, fit_segments=self._fit_segments)

        index_intersection = df.index.intersection(self._nans_to_impute_mask.index)
        nans_to_restore_mask = df.isna()
        nans_to_restore_mask.loc[index_intersection] = nans_to_restore_mask.loc[index_intersection].mask(
            self._nans_to_impute_mask.loc[index_intersection], False
        )  # Mask is broadcasted with True values

        result_df = self._fill(df)
        result_df.mask(nans_to_restore_mask, inplace=True)
        return result_df

    def _fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill the NaNs in a given Dataframe.

        Fills missed values for new dates according to ``self.strategy``

        Parameters
        ----------
        df:
            dataframe to fill

        Returns
        -------
        :
            Filled Dataframe.
        """
        if self._nans_to_impute_mask is None or (self.strategy is ImputerMode.mean and self._mean_imputer is None):
            raise ValueError("Transform is not fitted!")

        if self._strategy is ImputerMode.constant:
            df.fillna(value=self.constant_value, inplace=True)
        elif self._strategy is ImputerMode.forward_fill:
            df.ffill(inplace=True)
        elif self._strategy is ImputerMode.mean:
            self._mean_imputer.transform(df)  # type: ignore
        elif self._strategy is ImputerMode.running_mean or self._strategy is ImputerMode.seasonal:
            history = self.seasonality * self.window if self.window != -1 else len(df)
            nan_mask = df.isna().values
            nan_indexes = np.arange(len(df))[nan_mask.any(axis=1)]
            for i in nan_indexes:
                indexes = np.arange(i - self.seasonality, i - self.seasonality - history, -self.seasonality)
                indexes = indexes[indexes >= 0]
                if len(indexes) > 0:
                    impute_values = bn.nanmean(df.iloc[indexes], axis=0)
                    df.iloc[i] = np.where(nan_mask[i], impute_values, df.iloc[i])
        elif self._strategy is ImputerMode.seasonal_nonautoreg:
            lag_transform = LagTransform(in_column=self.in_column, lags=[self.seasonality], out_column="lag")
            sma_transform = MeanTransform(
                in_column=f"lag_{self.seasonality}",
                window=self.window,
                seasonality=self.seasonality,
                fillna=np.NaN,
                out_column="sma",
            )
            df_filled = (
                sma_transform._transform(lag_transform._transform(df))
                .loc[:, pd.IndexSlice[:, "sma"]]
                .rename(columns={"sma": self.in_column})
            )

            index_intersection = df.index.intersection(self._nans_to_impute_mask.index)
            df.loc[index_intersection] = df.loc[index_intersection].mask(
                self._nans_to_impute_mask.loc[index_intersection], df_filled.loc[index_intersection]
            )

        if self.default_value is not None:
            df.fillna(value=self.default_value, inplace=True)
        return df

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform dataframe.

        Parameters
        ----------
        df:
            Dataframe to be inverse transformed.

        Returns
        -------
        :
            Dataframe after applying inverse transformation.
        """
        if self._fit_segments is None or self._nans_to_impute_mask is None:
            raise ValueError("Transform is not fitted!")

        segments = sorted(set(df.columns.get_level_values("segment")))
        check_new_segments(transform_segments=segments, fit_segments=self._fit_segments)

        index_intersection = df.index.intersection(self._nans_to_impute_mask.index)
        df.loc[index_intersection] = df.mask(self._nans_to_impute_mask.loc[index_intersection])
        return df

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``strategy``, ``window``.
        Other parameters are expected to be set by the user.

        Strategy "seasonal" is suggested only if ``self.seasonality`` is set higher than 1.

        Returns
        -------
        :
            Grid to tune.
        """
        if self.seasonality > 1:
            return {
                "strategy": CategoricalDistribution(["constant", "mean", "running_mean", "forward_fill", "seasonal"]),
                "window": IntDistribution(low=1, high=20),
            }
        else:
            return {
                "strategy": CategoricalDistribution(["constant", "mean", "running_mean", "forward_fill"]),
                "window": IntDistribution(low=1, high=20),
            }


__all__ = ["TimeSeriesImputerTransform"]
