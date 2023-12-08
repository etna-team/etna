from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.forecasting.stl import STLForecastResults

from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.transforms.base import OneSegmentTransform
from etna.transforms.base import ReversiblePerSegmentWrapper


class _OneSegmentSTLTransform(OneSegmentTransform):
    def __init__(
        self,
        in_column: str,
        period: int,
        model: Union[str, TimeSeriesModel] = "arima",
        robust: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        stl_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Init _OneSegmentSTLTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        period:
            size of seasonality
        model:
            model to predict trend, default options are:

            1. "arima": ``ARIMA(data, 1, 1, 0)`` (default)

            2. "holt": ``ETSModel(data, trend='add')``

            Custom model should be a subclass of :py:class:`statsmodels.tsa.base.tsa_model.TimeSeriesModel`
            and have method ``get_prediction`` (not just ``predict``)
        robust:
            flag indicating whether to use robust version of STL
        model_kwargs:
            parameters for the model like in :py:class:`statsmodels.tsa.seasonal.STLForecast`
        stl_kwargs:
            additional parameters for :py:class:`statsmodels.tsa.seasonal.STLForecast`
        """
        if model_kwargs is None:
            model_kwargs = {}
        if stl_kwargs is None:
            stl_kwargs = {}

        self.in_column = in_column
        self.period = period

        if isinstance(model, str):
            if model == "arima":
                self.model = ARIMA
                if len(model_kwargs) == 0:
                    model_kwargs = {"order": (1, 1, 0)}
            elif model == "holt":
                self.model = ETSModel
                if len(model_kwargs) == 0:
                    model_kwargs = {"trend": "add"}
            else:
                raise ValueError(f"Not a valid option for model: {model}")
        elif isinstance(model, TimeSeriesModel):
            self.model = model
        else:
            raise ValueError("Model should be a string or TimeSeriesModel")

        self.robust = robust
        self.model_kwargs = model_kwargs
        self.stl_kwargs = stl_kwargs
        self.fit_results: Optional[STLForecastResults] = None
        self._first_int_index: Optional[int] = None

    def fit(self, df: pd.DataFrame) -> "_OneSegmentSTLTransform":
        """
        Perform STL decomposition and fit trend model.

        Parameters
        ----------
        df:
            Features dataframe with time

        Returns
        -------
        :
            instance after processing
        """
        df = df.loc[df[self.in_column].first_valid_index() : df[self.in_column].last_valid_index()]
        if df[self.in_column].isnull().values.any():
            raise ValueError("The input column contains NaNs in the middle of the series! Try to use the imputer.")

        if pd.api.types.is_integer_dtype(df.index):
            self._first_int_index = df.index[0]
            # create daily index, because get_prediction of holt model doesn't work after fitting with numpy data
            fake_index = pd.date_range(start="2020-01-01", periods=len(df), freq="D")
            df.index = pd.Index(fake_index, name=df.index.name)

        model = STLForecast(
            endog=df[self.in_column],
            model=self.model,
            model_kwargs=self.model_kwargs,
            period=self.period,
            robust=self.robust,
            **self.stl_kwargs,
        )
        self.fit_results = model.fit()
        return self

    def _get_season_trend(self, df: pd.DataFrame) -> pd.Series:
        if self.fit_results is None:
            raise ValueError("Transform is not fitted! Fit the Transform before calling transform method.")

        first_valid_index = df[self.in_column].first_valid_index()
        last_valid_index = df[self.in_column].last_valid_index()

        if pd.api.types.is_integer_dtype(df.index):
            start = first_valid_index - self._first_int_index
            end = last_valid_index - self._first_int_index
            if start < 0:
                raise ValueError("Transform can't work on integer timestamp before training data!")

            # call get_prediction by integer indices start, end, it works fine after using fake daily index during fit
            season_trend = self.fit_results.get_prediction(start=start, end=end).predicted_mean
            season_trend.index = np.arange(first_valid_index, last_valid_index + 1)
        else:
            season_trend = self.fit_results.get_prediction(start=first_valid_index, end=last_valid_index).predicted_mean

        return season_trend

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subtract trend and seasonal component.

        Parameters
        ----------
        df:
            Features dataframe with time

        Returns
        -------
        :
            Dataframe with extracted features
        """
        result = df
        season_trend = self._get_season_trend(df=df)
        result[self.in_column] -= season_trend
        return result

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend and seasonal component.

        Parameters
        ----------
        df:
            Features dataframe with time

        Returns
        -------
        :
            Dataframe with extracted features
        """
        result = df
        season_trend = self._get_season_trend(df=df)
        for colum_name in df.columns:
            result.loc[:, colum_name] += season_trend
        return result


class STLTransform(ReversiblePerSegmentWrapper):
    """Transform that uses :py:class:`statsmodels.tsa.seasonal.STL` to subtract season and trend from the data.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: str,
        period: int,
        model: Union[str, TimeSeriesModel] = "arima",
        robust: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        stl_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Init STLTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        period:
            size of seasonality
        model:
            model to predict trend, default options are:

            1. "arima": ``ARIMA(data, 1, 1, 0)`` (default)

            2. "holt": ``ETSModel(data, trend='add')``

            Custom model should be a subclass of :py:class:`statsmodels.tsa.base.tsa_model.TimeSeriesModel`
            and have method ``get_prediction`` (not just ``predict``)
        robust:
            flag indicating whether to use robust version of STL
        model_kwargs:
            parameters for the model like in :py:class:`statsmodels.tsa.forecasting.stl.STLForecast`
        stl_kwargs:
            additional parameters for :py:class:`statsmodels.tsa.forecasting.stl.STLForecast`
        """
        self.in_column = in_column
        self.period = period
        self.model = model
        self.robust = robust
        self.model_kwargs = model_kwargs
        self.stl_kwargs = stl_kwargs
        super().__init__(
            transform=_OneSegmentSTLTransform(
                in_column=self.in_column,
                period=self.period,
                model=self.model,
                robust=self.robust,
                model_kwargs=self.model_kwargs,
                stl_kwargs=self.stl_kwargs,
            ),
            required_features=[self.in_column],
        )

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return []

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``model``, ``robust``. Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "model": CategoricalDistribution(["arima", "holt"]),
            "robust": CategoricalDistribution([False, True]),
        }
