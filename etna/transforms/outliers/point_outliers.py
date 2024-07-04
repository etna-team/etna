from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Type
from typing import Union

import pandas as pd
from numpy.random import RandomState
from typing_extensions import Literal

from etna import SETTINGS
from etna.analysis import get_anomalies_density
from etna.analysis import get_anomalies_iqr
from etna.analysis import get_anomalies_isolation_forest
from etna.analysis import get_anomalies_mad
from etna.analysis import get_anomalies_median
from etna.analysis import get_anomalies_prediction_interval
from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.distributions import FloatDistribution
from etna.distributions import IntDistribution
from etna.models import SARIMAXModel
from etna.transforms.outliers.base import OutliersTransform

if SETTINGS.prophet_required:
    from etna.models import ProphetModel


class MedianOutliersTransform(OutliersTransform):
    """Transform that uses :py:func:`~etna.analysis.outliers.median_outliers.get_anomalies_median` to find anomalies in data.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: str,
        window_size: int = 10,
        alpha: float = 3,
        ignore_flag_column: Optional[str] = None,
    ):
        """Create instance of MedianOutliersTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        window_size:
            number of points in the window
        alpha:
            coefficient for determining the threshold
        ignore_flag_column:
            column name for skipping values from outlier check
        """
        self.window_size = window_size
        self.alpha = alpha
        super().__init__(in_column=in_column, ignore_flag_column=ignore_flag_column)

    def detect_outliers(self, ts: TSDataset) -> Dict[str, pd.Series]:
        """Call :py:func:`~etna.analysis.outliers.median_outliers.get_anomalies_median` function with self parameters.

        Parameters
        ----------
        ts:
            dataset to process

        Returns
        -------
        :
            dict of outliers in format {segment: [outliers_timestamps]}
        """
        return get_anomalies_median(
            ts=ts, in_column=self.in_column, window_size=self.window_size, alpha=self.alpha, index_only=False
        )

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``window_size``, ``alpha``. Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "window_size": IntDistribution(low=3, high=30),
            "alpha": FloatDistribution(low=0.5, high=5),
        }


class DensityOutliersTransform(OutliersTransform):
    """Transform that uses :py:func:`~etna.analysis.outliers.density_outliers.get_anomalies_density` to find anomalies in data.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: str,
        window_size: int = 15,
        distance_coef: float = 3,
        n_neighbors: int = 3,
        distance_func: Union[Literal["absolute_difference"], Callable[[float, float], float]] = "absolute_difference",
        ignore_flag_column: Optional[str] = None,
    ):
        """Create instance of DensityOutliersTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        window_size:
            size of windows to build
        distance_coef:
            factor for standard deviation that forms distance threshold to determine points are close to each other
        n_neighbors:
            min number of close neighbors of point not to be outlier
        distance_func:
            distance function. If a string is specified, a corresponding vectorized implementation will be used.
            Custom callable will be used as a scalar function, which will result in worse performance.
        ignore_flag_column:
            column name for skipping values from outlier check
        """
        self.window_size = window_size
        self.distance_coef = distance_coef
        self.n_neighbors = n_neighbors
        self.distance_func = distance_func
        super().__init__(in_column=in_column, ignore_flag_column=ignore_flag_column)

    def detect_outliers(self, ts: TSDataset) -> Dict[str, pd.Series]:
        """Call :py:func:`~etna.analysis.outliers.density_outliers.get_anomalies_density` function with self parameters.

        Parameters
        ----------
        ts:
            dataset to process

        Returns
        -------
        :
            dict of outliers in format {segment: [outliers_timestamps]}
        """
        return get_anomalies_density(
            ts=ts,
            in_column=self.in_column,
            window_size=self.window_size,
            distance_coef=self.distance_coef,
            n_neighbors=self.n_neighbors,
            distance_func=self.distance_func,
            index_only=False,
        )

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``window_size``, ``distance_coef``, ``n_neighbors``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "window_size": IntDistribution(low=3, high=30),
            "distance_coef": FloatDistribution(low=0.5, high=5),
            "n_neighbors": IntDistribution(low=1, high=10),
        }


class PredictionIntervalOutliersTransform(OutliersTransform):
    """Transform that uses :py:func:`~etna.analysis.outliers.prediction_interval_outliers.get_anomalies_prediction_interval` to find anomalies in data."""

    def __init__(
        self,
        in_column: str,
        model: Union[Literal["prophet"], Literal["sarimax"], Type["ProphetModel"], Type["SARIMAXModel"]],
        interval_width: float = 0.95,
        ignore_flag_column: Optional[str] = None,
        **model_kwargs,
    ):
        """Create instance of PredictionIntervalOutliersTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        model:
            model for prediction interval estimation
        interval_width:
            width of the prediction interval
        ignore_flag_column:
            column name for skipping values from outlier check
        Notes
        -----
        For not "target" column only column data will be used for learning.
        """
        self.model = model
        self.interval_width = interval_width
        self.model_kwargs = model_kwargs
        self._model_type = self._get_model_type(model)
        super().__init__(in_column=in_column, ignore_flag_column=ignore_flag_column)

    @staticmethod
    def _get_model_type(
        model: Union[Literal["prophet"], Literal["sarimax"], Type["ProphetModel"], Type["SARIMAXModel"]]
    ) -> Union[Type["ProphetModel"], Type["SARIMAXModel"]]:
        if isinstance(model, str):
            if model == "prophet":
                return ProphetModel
            elif model == "sarimax":
                return SARIMAXModel
        return model

    def detect_outliers(self, ts: TSDataset) -> Dict[str, pd.Series]:
        """Call :py:func:`~etna.analysis.outliers.prediction_interval_outliers.get_anomalies_prediction_interval` function with self parameters.

        Parameters
        ----------
        ts:
            dataset to process

        Returns
        -------
        :
            dict of outliers in format {segment: [outliers_timestamps]}
        """
        return get_anomalies_prediction_interval(
            ts=ts,
            model=self._model_type,
            interval_width=self.interval_width,
            in_column=self.in_column,
            index_only=False,
            **self.model_kwargs,
        )

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``interval_width``, ``model``. Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "interval_width": FloatDistribution(low=0.8, high=1.0),
            "model": CategoricalDistribution(["prophet", "sarimax"]),
        }


class IForestOutlierTransform(OutliersTransform):
    """Transform that uses :py:func:`~etna.analysis.outliers.isolation_forest_outliers.get_anomalies_isolation_forest` to find anomalies in data."""

    def __init__(
        self,
        in_column: str,
        ignore_flag_column: Optional[str] = None,
        features_to_use: Optional[Sequence[str]] = None,
        features_to_ignore: Optional[Sequence[str]] = None,
        ignore_missing: bool = False,
        n_estimators: int = 100,
        max_samples: Union[int, float, Literal["auto"]] = "auto",
        contamination: Union[float, Literal["auto"]] = "auto",
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, RandomState]] = None,
        verbose: int = 0,
    ):
        """Create instance of PredictionIntervalOutliersTransform.

        Parameters
        ----------
        in_column:
            Name of the column in which the anomaly is searching
        ignore_flag_column:
            Column name for skipping values from outlier check
        features_to_use:
            List of feature column names to use for anomaly detection
        features_to_ignore:
            List of feature column names to exclude from anomaly detection
        ignore_missing:
            Whether to ignore missing values inside a series
        n_estimators:
            The number of base estimators in the ensemble
        max_samples:
            The number of samples to draw from X to train each base estimator
                *  If int, then draw max_samples samples.

                *  If float, then draw max_samples * X.shape[0] samples.

                *  If “auto”, then max_samples=min(256, n_samples).

            If max_samples is larger than the number of samples provided, all samples will be used for all trees (no sampling).
        contamination:
            The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
            Used when fitting to define the threshold on the scores of the samples.
                *  If ‘auto’, the threshold is determined as in the original paper.

                *  If float, the contamination should be in the range (0, 0.5].
        max_features:
            The number of features to draw from X to train each base estimator.
                *  If int, then draw max_features features.

                *  If float, then draw `max(1, int(max_features * n_features_in_))` features.
            Note: using a float number less than 1.0 or integer less than number of features
            will enable feature subsampling and leads to a longer runtime.
        bootstrap:
                *  If True, individual trees are fit on random subsets of the training data sampled with replacement.
                *  If False, sampling without replacement is performed.
        n_jobs:
            The number of jobs to run in parallel for both fit and predict.
                *  None means 1 unless in a joblib.parallel_backend context.
                *  -1 means using all processors
        random_state:
            Controls the pseudo-randomness of the selection of the feature and split values for
            each branching step and each tree in the forest.
        verbose:
            Controls the verbosity of the tree building process.

        Notes
        -----
        To get more insights on parameters see documentation of Isolation Forest algorithm:

        `Documentation for Isolation Forest <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html>`_.
        """
        self.features_to_use = features_to_use
        self.features_to_ignore = features_to_ignore
        self.ignore_missing = ignore_missing
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        super().__init__(in_column=in_column, ignore_flag_column=ignore_flag_column)

    def detect_outliers(self, ts: TSDataset) -> Dict[str, pd.Series]:
        """Call :py:func:`~etna.analysis.outliers.isolation_forest_outliers.get_anomalies_isolation_forest` function with self parameters.

        Parameters
        ----------
        ts:
            Dataset to process

        Returns
        -------
        :
            Dict of outliers in format {segment: [outliers_timestamps]}
        """
        return get_anomalies_isolation_forest(
            ts=ts,
            in_column=self.in_column,
            features_to_use=self.features_to_use,
            features_to_ignore=self.features_to_ignore,
            ignore_missing=self.ignore_missing,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            index_only=False,
        )

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``n_estimators``, ``max_samples``, ``contamination``, ``max_features``, ``bootstrap``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "n_estimators": IntDistribution(low=10, high=1000),
            "max_samples": FloatDistribution(low=0.1, high=1.0),
            "contamination": FloatDistribution(low=0.1, high=0.5),
            "max_features": FloatDistribution(low=0.1, high=1.0),
            "bootstrap": CategoricalDistribution([True, False]),
        }


class IQROutlierTransform(OutliersTransform):
    """Transform that uses :py:func:`~etna.analysis.outliers.rolling_statistics.get_anomalies_iqr` to find anomalies in data."""

    def __init__(
        self,
        in_column: str = "target",
        ignore_flag_column: Optional[str] = None,
        window_size: int = 10,
        stride: int = 1,
        iqr_scale: float = 1.5,
        trend: bool = False,
        seasonality: bool = False,
        period: Optional[int] = None,
        stl_params: Optional[Dict[str, Any]] = None,
    ):
        """Create instance of ``PredictionIntervalOutliersTransform``.

        Parameters
        ----------
        in_column:
            Name of the column in which the anomaly is searching
        ignore_flag_column:
            Column name for skipping values from outlier check
        window_size:
            Number of points in the window
        stride:
            Offset between neighboring windows
        iqr_scale:
            Scaling parameter of the estimated interval
        trend:
            Whether to remove trend from the series
        seasonality:
            Whether to remove seasonality from the series
        period:
            Periodicity of the sequence for STL
        stl_params:
            Other parameters for STL. See :py:class:`statsmodels.tsa.seasonal.STL`
        """
        self.window_size = window_size
        self.stride = stride
        self.iqr_scale = iqr_scale
        self.trend = trend
        self.seasonality = seasonality
        self.period = period
        self.stl_params = stl_params
        super().__init__(in_column=in_column, ignore_flag_column=ignore_flag_column)

    def detect_outliers(self, ts: TSDataset) -> Dict[str, pd.Series]:
        """Call :py:func:`~etna.analysis.outliers.rolling_statistics.get_anomalies_iqr` function with self parameters.

        Parameters
        ----------
        ts:
            Dataset to process

        Returns
        -------
        :
            Dict of outliers in format {segment: [outliers_timestamps]}
        """
        return get_anomalies_iqr(
            ts=ts,
            in_column=self.in_column,
            window_size=self.window_size,
            stride=self.stride,
            iqr_scale=self.iqr_scale,
            trend=self.trend,
            seasonality=self.seasonality,
            period=self.period,
            stl_params=self.stl_params,
            index_only=False,
        )

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``iqr_scale``, ``trend``, ``seasonality``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "iqr_scale": FloatDistribution(low=0.5, high=10),
            "trend": CategoricalDistribution([True, False]),
            "seasonality": CategoricalDistribution([True, False]),
        }


class MADOutlierTransform(OutliersTransform):
    """Transform that uses :py:func:`~etna.analysis.outliers.rolling_statistics.get_anomalies_mad` to find anomalies in data."""

    def __init__(
        self,
        in_column: str = "target",
        ignore_flag_column: Optional[str] = None,
        window_size: int = 10,
        stride: int = 1,
        mad_scale: float = 3,
        trend: bool = False,
        seasonality: bool = False,
        period: Optional[int] = None,
        stl_params: Optional[Dict[str, Any]] = None,
    ):
        """Create instance of ``MADOutlierTransform``.

        Parameters
        ----------
        in_column:
            Name of the column in which the anomaly is searching
        ignore_flag_column:
            Column name for skipping values from outlier check
        window_size:
            Number of points in the window
        stride:
            Offset between neighboring windows
        mad_scale:
            Scaling parameter of the estimated interval
        trend:
            Whether to remove trend from the series
        seasonality:
            Whether to remove seasonality from the series
        period:
            Periodicity of the sequence for STL
        stl_params:
            Other parameters for STL. See :py:class:`statsmodels.tsa.seasonal.STL`
        """
        self.window_size = window_size
        self.stride = stride
        self.mad_scale = mad_scale
        self.trend = trend
        self.seasonality = seasonality
        self.period = period
        self.stl_params = stl_params
        super().__init__(in_column=in_column, ignore_flag_column=ignore_flag_column)

    def detect_outliers(self, ts: TSDataset) -> Dict[str, pd.Series]:
        """Call :py:func:`~etna.analysis.outliers.rolling_statistics.get_anomalies_mad` function with self parameters.

        Parameters
        ----------
        ts:
            Dataset to process

        Returns
        -------
        :
            Dict of outliers in format {segment: [outliers_timestamps]}
        """
        return get_anomalies_mad(
            ts=ts,
            in_column=self.in_column,
            window_size=self.window_size,
            stride=self.stride,
            mad_scale=self.mad_scale,
            trend=self.trend,
            seasonality=self.seasonality,
            period=self.period,
            stl_params=self.stl_params,
            index_only=False,
        )

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``mad_scale``, ``trend``, ``seasonality``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "mad_scale": FloatDistribution(low=0.5, high=10),
            "trend": CategoricalDistribution([True, False]),
            "seasonality": CategoricalDistribution([True, False]),
        }


__all__ = [
    "MedianOutliersTransform",
    "DensityOutliersTransform",
    "PredictionIntervalOutliersTransform",
    "IForestOutlierTransform",
    "IQROutlierTransform",
    "MADOutlierTransform",
]
