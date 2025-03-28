from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.forecasting.stl import STLForecastResults

from etna.datasets.utils import determine_freq
from etna.datasets.utils import determine_num_steps
from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.transforms.base import OneSegmentTransform
from etna.transforms.base import ReversiblePerSegmentWrapper

_DEFAULT_FREQ = object()


class _OneSegmentSTLTransform(OneSegmentTransform):
    def __init__(
        self,
        in_column: str,
        period: int,
        model: Union[str, Type[TimeSeriesModel]] = "arima",
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
        elif isinstance(model, type) and issubclass(model, TimeSeriesModel):
            self.model = model
        else:
            raise ValueError("Model should be a string or TimeSeriesModel")

        self.robust = robust
        self.model_kwargs = model_kwargs
        self.stl_kwargs = stl_kwargs
        self.fit_results: Optional[STLForecastResults] = None
        self._first_train_timestamp: Union[pd.Timestamp, int, None] = None
        self._freq_offset: Optional[pd.DateOffset] = _DEFAULT_FREQ  # type: ignore

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

        self._first_train_timestamp = df.index.min()
        self._freq_offset = determine_freq(df.index, freq_format="offset")

        endog = df[self.in_column]
        if pd.api.types.is_integer_dtype(df.index):
            # make index start with zero
            endog.index = endog.index - self._first_train_timestamp

        model = STLForecast(
            endog=endog,
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

        start_timestamp = df[self.in_column].first_valid_index()
        end_timestamp = df[self.in_column].last_valid_index()

        # if all values are NaNs
        if start_timestamp is None:
            return pd.Series([], dtype=float)

        start_idx = determine_num_steps(
            start_timestamp=self._first_train_timestamp, end_timestamp=start_timestamp, freq=self._freq_offset
        )
        end_idx = determine_num_steps(
            start_timestamp=self._first_train_timestamp, end_timestamp=end_timestamp, freq=self._freq_offset
        )

        prediction = self.fit_results.get_prediction(start=start_idx, end=end_idx).predicted_mean.values

        index = df.index[df.index.get_loc(start_timestamp) : df.index.get_loc(end_timestamp) + 1]
        season_trend = pd.Series(prediction, index=index)
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
        model: Union[str, Type[TimeSeriesModel]] = "arima",
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
