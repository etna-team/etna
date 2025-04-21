import warnings
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from statsforecast.models import ARIMA
from statsforecast.models import AutoARIMA
from statsforecast.models import AutoCES
from statsforecast.models import AutoETS
from statsforecast.models import AutoTheta

from etna.datasets.utils import determine_freq
from etna.datasets.utils import determine_num_steps
from etna.distributions import BaseDistribution
from etna.distributions import IntDistribution
from etna.models.base import BaseAdapter
from etna.models.base import NonPredictionIntervalContextIgnorantAbstractModel
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.mixins import NonPredictionIntervalContextIgnorantModelMixin
from etna.models.mixins import PerSegmentModelMixin
from etna.models.mixins import PredictionIntervalContextIgnorantModelMixin

StatsForecastModel = Union[AutoCES, AutoARIMA, AutoTheta, AutoETS, ARIMA]
_DEFAULT_FREQ = object()


class _StatsForecastBaseAdapter(BaseAdapter):
    """Base class for adapters for models from statsforecast package."""

    def __init__(self, model: StatsForecastModel, support_prediction_intervals: bool):
        """
        Init model with given parameters.

        Parameters
        ----------
        model:
            Model from statsforecast.
        support_prediction_intervals:
            Should model support prediction intervals.
        """
        self.regressor_columns: Optional[List[str]] = None
        self._freq: Optional[str] = _DEFAULT_FREQ  # type: ignore
        self._first_train_timestamp: Optional[pd.Timestamp] = None
        self._last_train_timestamp: Optional[pd.Timestamp] = None
        self._model = model
        self._support_prediction_intervals = support_prediction_intervals

    def _check_not_used_columns(self, df: pd.DataFrame):
        if self.regressor_columns is None:
            raise ValueError("Something went wrong, regressor_columns is None!")

        columns_not_used = [col for col in df.columns if col not in ["target", "timestamp"] + self.regressor_columns]
        if columns_not_used:
            warnings.warn(
                message=f"This model doesn't work with exogenous features unknown in future. "
                f"Columns {columns_not_used} won't be used."
            )

    def _select_regressors(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Select data with regressors.

        During fit there can't be regressors with NaNs, they are removed at higher level.
        Look at the issue: https://github.com/tinkoff-ai/etna/issues/557

        During prediction without validation NaNs in regressors lead to NaNs in the answer.

        This model requires data to be in float dtype.
        """
        if self.regressor_columns is None:
            raise ValueError("Something went wrong, regressor_columns is None!")

        regressors_with_nans = [regressor for regressor in self.regressor_columns if df[regressor].isna().sum() > 0]
        if regressors_with_nans:
            raise ValueError(
                f"Regressors {regressors_with_nans} contain NaN values. "
                "Try to lower horizon value, or drop these regressors."
            )

        regressor_columns_except_datetime = (
            df[self.regressor_columns].select_dtypes(exclude="datetime").columns.to_list()
        )
        if regressor_columns_except_datetime:
            try:
                result = df[regressor_columns_except_datetime].values.astype(float)
            except ValueError as e:
                raise ValueError(f"Only convertible to float features are allowed! Error: {str(e)}")
        else:
            result = None

        return result

    def fit(self, df: pd.DataFrame, regressors: List[str]) -> "_StatsForecastBaseAdapter":
        """Fit statsforecast adapter.

        Parameters
        ----------
        df:
            Features dataframe
        regressors:
            List of the columns with regressors

        Returns
        -------
        :
            Fitted adapter
        """
        self.regressor_columns = regressors
        self._check_not_used_columns(df)

        endog_data = df["target"].values
        exog_data = self._select_regressors(df)

        self._model.fit(y=endog_data, X=exog_data)

        self._freq = determine_freq(timestamps=df["timestamp"], freq_format="offset")
        self._first_train_timestamp = df["timestamp"].min()
        self._last_train_timestamp = df["timestamp"].max()

        return self

    def _prepare_borders(self, df: pd.DataFrame, in_sample: bool) -> Tuple[int, int]:
        """Determine start and end indexes for timestamps."""
        start_timestamp = df["timestamp"].min()
        end_timestamp = df["timestamp"].max()

        if in_sample:
            if start_timestamp < self._first_train_timestamp:
                raise NotImplementedError(
                    "This model can't make predict on past out-of-sample data! The data before training is given."
                )

            if end_timestamp > self._last_train_timestamp:
                raise NotImplementedError(
                    "This model can't make predict on future out-of-sample data! "
                    "Use forecast method for this type of prediction."
                )

            first_timestamp = self._first_train_timestamp

        else:
            if start_timestamp < self._last_train_timestamp:
                raise NotImplementedError(
                    "This model can't make forecast on history data! Use method predict for in-sample prediction."
                )

            first_timestamp = self._last_train_timestamp

        start_idx = determine_num_steps(start_timestamp=first_timestamp, end_timestamp=start_timestamp, freq=self._freq)
        end_idx = determine_num_steps(start_timestamp=first_timestamp, end_timestamp=end_timestamp, freq=self._freq)

        return start_idx, end_idx

    def forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute predictions on future data from a statsforecast model.

        This method only works on data that goes right after the train.

        Parameters
        ----------
        df:
            Features dataframe

        Returns
        -------
        :
            DataFrame with predictions
        """
        if self._freq is _DEFAULT_FREQ:
            raise ValueError("Model is not fitted! Fit the model before calling predict method!")

        start_idx, end_idx = self._prepare_borders(df=df, in_sample=False)

        if start_idx > 1:
            raise NotImplementedError(
                "This model can't make forecast on out-of-sample data that goes after training data with a gap! "
                "You can only forecast from the next point after the last one in the training dataset."
            )

        exog_data = self._select_regressors(df)

        forecast = self._model.predict(h=end_idx, X=exog_data)
        y_pred = pd.DataFrame({"target": forecast["mean"]})

        return y_pred

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute in-sample predictions from a statsforecast model.

        This method only works on train data.

        Parameters
        ----------
        df:
            Features dataframe

        Returns
        -------
        :
            DataFrame with predictions
        """
        if self._freq is _DEFAULT_FREQ:
            raise ValueError("Model is not fitted! Fit the model before calling predict method!")

        start_idx, end_idx = self._prepare_borders(df=df, in_sample=True)

        forecast = self._model.predict_in_sample()
        y_pred = pd.DataFrame({"target": forecast["fitted"][start_idx : end_idx + 1]})

        return y_pred

    def forecast_intervals(self, df: pd.DataFrame, quantiles: Sequence[float]) -> pd.DataFrame:
        """Compute prediction intervals on future data from a statsforecast model.

        This method only works on data that goes right after the train.

        Parameters
        ----------
        df:
            Features dataframe
        quantiles:
            Levels of prediction distribution

        Returns
        -------
        :
            DataFrame with predictions
        """
        if self._freq is _DEFAULT_FREQ:
            raise ValueError("Model is not fitted! Fit the model before calling predict method!")

        start_idx, end_idx = self._prepare_borders(df=df, in_sample=False)

        if start_idx > 1:
            raise NotImplementedError(
                "This model can't make forecast on out-of-sample data that goes after training data with a gap! "
                "You can only forecast from the next point after the last one in the training dataset."
            )

        levels = []
        for quantile in quantiles:
            width = abs(1 / 2 - quantile) * 2
            level = int(width * 100)
            levels.append(level)

        # get unique levels to prevent strange behavior with stacking interval predictions
        unique_levels = list(set(levels))

        exog_data = self._select_regressors(df)
        forecast = self._model.predict(h=end_idx, X=exog_data, level=unique_levels)

        y_pred = pd.DataFrame()
        for quantile, level in zip(quantiles, levels):
            if quantile < 1 / 2:
                series = forecast[f"lo-{level}"]
            else:
                series = forecast[f"hi-{level}"]
            y_pred[f"target_{quantile:.4g}"] = series

        return y_pred

    def predict_intervals(self, df: pd.DataFrame, quantiles: Sequence[float]) -> pd.DataFrame:
        """Compute in-sample prediction intervals from a statsforecast model.

        This method only works on train data.

        Parameters
        ----------
        df:
            Features dataframe
        quantiles:
            Levels of prediction distribution

        Returns
        -------
        :
            DataFrame with predictions
        """
        if self._freq is _DEFAULT_FREQ:
            raise ValueError("Model is not fitted! Fit the model before calling predict method!")

        start_idx, end_idx = self._prepare_borders(df=df, in_sample=True)

        levels = []
        for quantile in quantiles:
            width = abs(1 / 2 - quantile) * 2
            level = int(width * 100)
            levels.append(level)

        # get unique levels to prevent strange behavior with stacking interval predictions
        unique_levels = list(set(levels))

        forecast = self._model.predict_in_sample(level=unique_levels)  # type: ignore

        y_pred = pd.DataFrame()
        for quantile, level in zip(quantiles, levels):
            if quantile < 1 / 2:
                series = forecast[f"fitted-lo-{level}"]
            else:
                series = forecast[f"fitted-hi-{level}"]
            y_pred[f"target_{quantile:.4g}"] = series[start_idx : end_idx + 1]

        return y_pred

    def forecast_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate forecast components.

        Parameters
        ----------
        df:
            features dataframe

        Returns
        -------
        :
            dataframe with forecast components
        """
        raise NotImplementedError("This mode isn't currently implemented!")

    def predict_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate prediction components.

        Parameters
        ----------
        df:
            features dataframe

        Returns
        -------
        :
            dataframe with prediction components
        """
        raise NotImplementedError("This mode isn't currently implemented!")

    def get_model(self) -> StatsForecastModel:
        """Get statsforecast model that is used inside etna class.

        Returns
        -------
        :
           Internal model
        """
        return self._model


class _AutoARIMAAdapter(_StatsForecastBaseAdapter):
    """Adapter for :py:class:`statsforecast.models.AutoARIMA`."""

    def __init__(
        self,
        d: Optional[int] = None,
        D: Optional[int] = None,  # noqa: N803
        max_p: int = 5,
        max_q: int = 5,
        max_P: int = 2,
        max_Q: int = 2,
        max_order: int = 5,
        max_d: int = 2,
        max_D: int = 1,
        start_p: int = 2,
        start_q: int = 2,
        start_P: int = 1,
        start_Q: int = 1,
        season_length: int = 1,
        **kwargs,
    ):
        """Init model with given params.

        Parameters
        ----------
        d:
            Order of first-differencing.
        D:
            Order of seasonal-differencing.
        max_p:
            Max autorregresives p.
        max_q:
            Max moving averages q.
        max_P:
            Max seasonal autorregresives P.
        max_Q:
            Max seasonal moving averages Q.
        max_order:
            Max p+q+P+Q value if not stepwise selection.
        max_d:
            Max non-seasonal differences.
        max_D:
            Max seasonal differences.
        start_p:
            Starting value of p in stepwise procedure.
        start_q:
            Starting value of q in stepwise procedure.
        start_P:
            Starting value of P in stepwise procedure.
        start_Q:
            Starting value of Q in stepwise procedure.
        season_length:
            Number of observations per unit of time. Ex: 24 Hourly data.
        **kwargs:
            Additional parameters for :py:class:`statsforecast.models.AutoARIMA`.
        """
        self.d = d
        self.D = D
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.max_order = max_order
        self.max_d = max_d
        self.max_D = max_D
        self.start_p = start_p
        self.start_q = start_q
        self.start_P = start_P
        self.start_Q = start_Q
        self.season_length = season_length
        self.kwargs = kwargs
        super().__init__(
            model=AutoARIMA(
                d=self.d,
                D=self.D,
                max_p=self.max_p,
                max_q=self.max_q,
                max_P=self.max_P,
                max_Q=self.max_Q,
                max_order=self.max_order,
                max_d=self.max_d,
                max_D=self.max_D,
                start_p=self.start_P,
                start_q=self.start_q,
                start_P=self.start_P,
                start_Q=self.start_Q,
                season_length=self.season_length,
                **self.kwargs,
            ),
            support_prediction_intervals=True,
        )


class _ARIMAAdapter(_StatsForecastBaseAdapter):
    """Adapter for :py:class:`statsforecast.models.ARIMA`."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (0, 0, 0),
        season_length: int = 1,
        seasonal_order: Tuple[int, int, int] = (0, 0, 0),
        **kwargs,
    ):
        """Init model with given params.

        Parameters
        ----------
        order:
            A specification of the non-seasonal part of the ARIMA model: the three components (p, d, q) are the AR order, the degree of differencing, and the MA order.
        season_length:
            Number of observations per unit of time. Ex: 24 Hourly data.
        seasonal_order:
            A specification of the seasonal part of the ARIMA model.
            (P, D, Q) for the  AR order, the degree of differencing, the MA order.
        **kwargs:
            Additional parameters for :py:class:`statsforecast.models.ARIMA`.
        """
        self.order = order
        self.season_length = season_length
        self.seasonal_order = seasonal_order
        self.kwargs = kwargs
        super().__init__(
            model=ARIMA(
                order=self.order,
                season_length=self.season_length,
                seasonal_order=self.seasonal_order,
                **self.kwargs,
            ),
            support_prediction_intervals=True,
        )


class _AutoThetaAdapter(_StatsForecastBaseAdapter):
    """Adapter for :py:class:`statsforecast.models.AutoTheta`."""

    def __init__(
        self,
        season_length: int = 1,
        decomposition_type: str = "multiplicative",
        model: Optional[str] = None,
    ):
        """Init model with given params.

        Parameters
        ----------
        season_length:
            Number of observations per unit of time. Ex: 24 Hourly data.
        decomposition_type:
            Sesonal decomposition type, 'multiplicative' (default) or 'additive'.
        model:
            Controlling Theta Model. By default searches the best model.
        """
        self.season_length = season_length
        self.decomposition_type = decomposition_type
        self.model = model
        super().__init__(
            model=AutoTheta(
                season_length=self.season_length, decomposition_type=self.decomposition_type, model=self.model
            ),
            support_prediction_intervals=True,
        )


class _AutoCESAdapter(_StatsForecastBaseAdapter):
    """Adapter for :py:class:`statsforecast.models.AutoCES`."""

    def __init__(self, season_length: int = 1, model: str = "Z"):
        """Init model with given params.

        Parameters
        ----------
        season_length:
            Number of observations per unit of time. Ex: 24 Hourly data.
        model:
            Controlling state-space-equations.
        """
        self.season_length = season_length
        self.model = model
        super().__init__(
            model=AutoCES(season_length=self.season_length, model=self.model),
            support_prediction_intervals=False,
        )


class _AutoETSAdapter(_StatsForecastBaseAdapter):
    """Adapter for :py:class:`statsforecast.models.AutoETS`."""

    def __init__(self, season_length: int = 1, model: str = "ZZZ", damped: Optional[bool] = None):
        """Init model with given params.

        Parameters
        ----------
        season_length:
            Number of observations per unit of time. Ex: 24 Hourly data.
        model:
            Controlling state-space-equations.
        damped:
            A parameter that 'dampens' the trend.
        """
        self.season_length = season_length
        self.model = model
        self.damped = damped
        super().__init__(
            model=AutoETS(
                season_length=self.season_length,
                model=self.model,
                damped=self.damped,
            ),
            support_prediction_intervals=True,
        )


class StatsForecastAutoARIMAModel(
    PerSegmentModelMixin, PredictionIntervalContextIgnorantModelMixin, PredictionIntervalContextIgnorantAbstractModel
):
    """
    Class for holding :py:class:`statsforecast.models.AutoARIMA`.

    `Documentation for the underlying model <https://nixtla.github.io/statsforecast/src/core/models.html#autoarima>`_.

    Note
    ----
    This model requires ``statsforecast`` extension to be installed.
    Read more about this at :ref:`installation instruction <installation>`.
    """

    def __init__(
        self,
        d: Optional[int] = None,
        D: Optional[int] = None,  # noqa: N803
        max_p: int = 5,
        max_q: int = 5,
        max_P: int = 2,
        max_Q: int = 2,
        max_order: int = 5,
        max_d: int = 2,
        max_D: int = 1,
        start_p: int = 2,
        start_q: int = 2,
        start_P: int = 1,
        start_Q: int = 1,
        season_length: int = 1,
        **kwargs,
    ):
        """Init model with given params.

        Parameters
        ----------
        d:
            Order of first-differencing.
        D:
            Order of seasonal-differencing.
        max_p:
            Max autorregresives p.
        max_q:
            Max moving averages q.
        max_P:
            Max seasonal autorregresives P.
        max_Q:
            Max seasonal moving averages Q.
        max_order:
            Max p+q+P+Q value if not stepwise selection.
        max_d:
            Max non-seasonal differences.
        max_D:
            Max seasonal differences.
        start_p:
            Starting value of p in stepwise procedure.
        start_q:
            Starting value of q in stepwise procedure.
        start_P:
            Starting value of P in stepwise procedure.
        start_Q:
            Starting value of Q in stepwise procedure.
        season_length:
            Number of observations per unit of time. Ex: 24 Hourly data.
        **kwargs:
            Additional parameters for :py:class:`statsforecast.models.AutoARIMA`.
        """
        self.d = d
        self.D = D
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.max_order = max_order
        self.max_d = max_d
        self.max_D = max_D
        self.start_p = start_p
        self.start_q = start_q
        self.start_P = start_P
        self.start_Q = start_Q
        self.season_length = season_length
        self.kwargs = kwargs
        super().__init__(
            base_model=_AutoARIMAAdapter(
                d=self.d,
                D=self.D,
                max_p=self.max_p,
                max_q=self.max_q,
                max_P=self.max_P,
                max_Q=self.max_Q,
                max_order=self.max_order,
                max_d=self.max_d,
                max_D=self.max_D,
                start_p=self.start_P,
                start_q=self.start_q,
                start_P=self.start_P,
                start_Q=self.start_Q,
                season_length=self.season_length,
                **self.kwargs,
            )
        )


class StatsForecastARIMAModel(
    PerSegmentModelMixin, PredictionIntervalContextIgnorantModelMixin, PredictionIntervalContextIgnorantAbstractModel
):
    """
    Class for holding :py:class:`statsforecast.models.ARIMA`.

    `Documentation for the underlying model <https://nixtla.github.io/statsforecast/src/core/models.html#arima>`_.

    Note
    ----
    This model requires ``statsforecast`` extension to be installed.
    Read more about this at :ref:`installation instruction <installation>`.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (0, 0, 0),
        season_length: int = 1,
        seasonal_order: Tuple[int, int, int] = (0, 0, 0),
        **kwargs,
    ):
        """Init model with given params.

        Parameters
        ----------
        order:
            A specification of the non-seasonal part of the ARIMA model: the three components (p, d, q) are the AR order, the degree of differencing, and the MA order.
        season_length:
            Number of observations per unit of time. Ex: 24 Hourly data.
        seasonal_order:
            A specification of the seasonal part of the ARIMA model.
            (P, D, Q) for the  AR order, the degree of differencing, the MA order.
        **kwargs:
            Additional parameters for :py:class:`statsforecast.models.ARIMA`.
        """
        self.order = order
        self.season_length = season_length
        self.seasonal_order = seasonal_order
        self.kwargs = kwargs
        super().__init__(
            base_model=_ARIMAAdapter(
                order=self.order,
                season_length=self.season_length,
                seasonal_order=self.seasonal_order,
                **self.kwargs,
            ),
        )

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``order.0``, ``order.1``, ``order.2``.
        If ``self.season_length`` is greater than one, then it also tunes parameters:
        ``seasonal_order.0``, ``seasonal_order.1``, ``seasonal_order.2``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        grid: Dict[str, BaseDistribution] = {
            "order.0": IntDistribution(low=1, high=6),
            "order.1": IntDistribution(low=1, high=2),
            "order.2": IntDistribution(low=1, high=6),
        }

        num_periods = self.season_length
        if num_periods > 1:
            grid.update(
                {
                    "seasonal_order.0": IntDistribution(low=0, high=2),
                    "seasonal_order.1": IntDistribution(low=0, high=1),
                    "seasonal_order.2": IntDistribution(low=0, high=1),
                }
            )

        return grid


class StatsForecastAutoThetaModel(
    PerSegmentModelMixin, PredictionIntervalContextIgnorantModelMixin, PredictionIntervalContextIgnorantAbstractModel
):
    """
    Class for holding :py:class:`statsforecast.models.AutoTheta`.

    `Documentation for the underlying model <https://nixtla.github.io/statsforecast/src/core/models.html#autotheta>`_.

    Note
    ----
    This model requires ``statsforecast`` extension to be installed.
    Read more about this at :ref:`installation instruction <installation>`.
    """

    def __init__(
        self,
        season_length: int = 1,
        decomposition_type: str = "multiplicative",
        model: Optional[str] = None,
    ):
        """Init model with given params.

        Parameters
        ----------
        season_length:
            Number of observations per unit of time. Ex: 24 Hourly data.
        decomposition_type:
            Sesonal decomposition type, 'multiplicative' (default) or 'additive'.
        model:
            Controlling Theta Model. By default searches the best model.
        """
        self.season_length = season_length
        self.decomposition_type = decomposition_type
        self.model = model
        super().__init__(
            base_model=_AutoThetaAdapter(
                season_length=self.season_length, decomposition_type=self.decomposition_type, model=self.model
            ),
        )


class StatsForecastAutoCESModel(
    PerSegmentModelMixin,
    NonPredictionIntervalContextIgnorantModelMixin,
    NonPredictionIntervalContextIgnorantAbstractModel,
):
    """
    Class for holding :py:class:`statsforecast.models.AutoCES`.

    `Documentation for the underlying model <https://nixtla.github.io/statsforecast/src/core/models.html#autoces>`_.

    Note
    ----
    This model requires ``statsforecast`` extension to be installed.
    Read more about this at :ref:`installation instruction <installation>`.
    """

    def __init__(self, season_length: int = 1, model: str = "Z"):
        """Init model with given params.

        Parameters
        ----------
        season_length:
            Number of observations per unit of time. Ex: 24 Hourly data.
        model:
            Controlling state-space-equations.
        """
        self.season_length = season_length
        self.model = model
        super().__init__(
            base_model=_AutoCESAdapter(season_length=self.season_length, model=self.model),
        )


class StatsForecastAutoETSModel(
    PerSegmentModelMixin, PredictionIntervalContextIgnorantModelMixin, PredictionIntervalContextIgnorantAbstractModel
):
    """
    Class for holding :py:class:`statsforecast.models.AutoETS`.

    `Documentation for the underlying model <https://nixtla.github.io/statsforecast/src/core/models.html#autoets>`_.

    Note
    ----
    This model requires ``statsforecast`` extension to be installed.
    Read more about this at :ref:`installation instruction <installation>`.
    """

    def __init__(self, season_length: int = 1, model: str = "ZZZ", damped: Optional[bool] = None):
        """Init model with given params.


        Parameters
        ----------
        season_length:
            Number of observations per unit of time. Ex: 24 Hourly data.
        model:
            Controlling state-space-equations.
        damped:
            A parameter that 'dampens' the trend.
        """
        self.season_length = season_length
        self.model = model
        self.damped = damped
        super().__init__(
            base_model=_AutoETSAdapter(season_length=self.season_length, model=self.model, damped=self.damped),
        )
