import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from typing_extensions import Literal

from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.ensembles.mixins import EnsembleMixin
from etna.ensembles.mixins import SaveEnsembleMixin
from etna.pipeline.base import BasePipeline


class StackingEnsemble(EnsembleMixin, SaveEnsembleMixin, BasePipeline):
    """StackingEnsemble is a pipeline that forecast future using the metamodel to combine the forecasts of the base models.

    Examples
    --------
    >>> from etna.datasets import generate_ar_df
    >>> from etna.datasets import TSDataset
    >>> from etna.ensembles import VotingEnsemble
    >>> from etna.models import NaiveModel
    >>> from etna.models import MovingAverageModel
    >>> from etna.pipeline import Pipeline
    >>> import pandas as pd
    >>> pd.options.display.float_format = '{:,.2f}'.format
    >>> df = generate_ar_df(periods=100, start_time="2021-06-01", ar_coef=[0.8], n_segments=3)
    >>> ts = TSDataset(df, "D")
    >>> ma_pipeline = Pipeline(model=MovingAverageModel(window=5), transforms=[], horizon=7)
    >>> naive_pipeline = Pipeline(model=NaiveModel(lag=10), transforms=[], horizon=7)
    >>> ensemble = StackingEnsemble(pipelines=[ma_pipeline, naive_pipeline])
    >>> _ = ensemble.fit(ts=ts)
    >>> forecast = ensemble.forecast()
    >>> forecast[:,:,"target"]
    segment    segment_0 segment_1 segment_2
    feature       target    target    target
    timestamp
    2021-09-09      0.70      1.47      0.20
    2021-09-10      0.62      1.53      0.26
    2021-09-11      0.50      1.78      0.36
    2021-09-12      0.37      1.88      0.21
    2021-09-13      0.46      1.87      0.25
    2021-09-14      0.44      1.49      0.21
    2021-09-15      0.36      1.56      0.30
    """

    def __init__(
        self,
        pipelines: List[BasePipeline],
        final_model: Optional[RegressorMixin] = None,
        n_folds: int = 3,
        features_to_use: Union[None, Literal["all"], List[str]] = None,
        n_jobs: int = 1,
        joblib_params: Optional[Dict[str, Any]] = None,
    ):
        """Init StackingEnsemble.

        Parameters
        ----------
        pipelines:
            List of pipelines that should be used in ensemble.
        final_model:
            Regression model with fit/predict interface which will be used to combine the base estimators.
        n_folds:
            Number of folds to use in the backtest. Backtest is not used for model evaluation but for prediction.
        features_to_use:
            Features except the forecasts of the base models to use in the ``final_model``.
        n_jobs:
            Number of jobs to run in parallel.
        joblib_params:
            Additional parameters for :py:class:`joblib.Parallel`.

        Raises
        ------
        ValueError:
            If the number of the pipelines is less than 2 or pipelines have different horizons.
        """
        self._validate_pipeline_number(pipelines=pipelines)
        self.pipelines = pipelines
        self.final_model = LinearRegression(positive=True) if final_model is None else final_model
        self._validate_backtest_n_folds(n_folds)
        self.n_folds = n_folds
        self.features_to_use = features_to_use
        self.filtered_features_for_final_model: Union[None, Set[str]] = None
        self.n_jobs = n_jobs
        if joblib_params is None:
            self.joblib_params = dict(verbose=11, backend="multiprocessing", mmap_mode="c")
        else:
            self.joblib_params = joblib_params
        super().__init__(horizon=self._get_horizon(pipelines=pipelines))

    def _make_same_level(self, ts: TSDataset, forecasts: List[pd.DataFrame]) -> TSDataset:
        if ts.has_hierarchy():
            current_df_level = ts._get_dataframe_level(df=forecasts[0])
            if ts.current_df_level != current_df_level:
                ts = ts.get_level_dataset(current_df_level)  # type: ignore
        return ts

    def _filter_features_to_use(self, forecasts: List[pd.DataFrame]) -> Union[None, Set[str]]:
        """Return all the features from ``features_to_use`` which can be obtained from base models' forecasts."""
        features_df = pd.concat(forecasts, axis=1)
        available_features = set(features_df.columns.get_level_values("feature"))
        features_to_use = self.features_to_use
        if features_to_use is None:
            return None
        elif features_to_use == "all":
            return available_features - {"target"}
        elif isinstance(features_to_use, list):
            features_to_use_unique = set(features_to_use)
            if len(features_to_use_unique) == 0:
                return None
            elif features_to_use_unique.issubset(available_features):
                return features_to_use_unique
            else:
                unavailable_features = features_to_use_unique - available_features
                warnings.warn(f"Features {unavailable_features} are not found and will be dropped!")
                return features_to_use_unique.intersection(available_features)
        else:
            warnings.warn(
                "Feature list is passed in the wrong format."
                "Only the base models' forecasts will be used for the final forecast."
            )
            return None

    def _backtest_pipeline(self, pipeline: BasePipeline, ts: TSDataset) -> List[TSDataset]:
        """Get forecasts from backtest for given pipeline."""
        forecasts = pipeline.get_historical_forecasts(ts=ts, n_folds=self.n_folds)
        return forecasts

    def fit(self, ts: TSDataset, save_ts: bool = True) -> "StackingEnsemble":
        """Fit the ensemble.

        Method doesn't change the given ``ts``.

        Saved ``ts`` is the link to given ``ts``.

        Parameters
        ----------
        ts:
            TSDataset to fit ensemble.
        save_ts:
            Will ``ts`` be saved in the pipeline during ``fit``.

        Returns
        -------
        self:
            Fitted ensemble.
        """
        # Get forecasts from base models on backtest to fit the final model on
        nested_forecast_ts_list = Parallel(n_jobs=self.n_jobs, **self.joblib_params)(
            delayed(self._backtest_pipeline)(pipeline=pipeline, ts=ts) for pipeline in self.pipelines
        )

        # Fit the final model
        forecasts = [
            pd.concat([forecast_ts._df for forecast_ts in forecast_ts_list], axis=0)
            for forecast_ts_list in nested_forecast_ts_list
        ]
        self.filtered_features_for_final_model = self._filter_features_to_use(forecasts)
        x, y = self._make_features(ts=ts, forecasts=forecasts, train=True)
        self.final_model.fit(x, y)

        # Fit the base models
        self.pipelines = Parallel(n_jobs=self.n_jobs, **self.joblib_params)(
            delayed(self._fit_pipeline)(pipeline=pipeline, ts=ts) for pipeline in self.pipelines
        )

        if save_ts:
            self.ts = ts

        return self

    def _make_features(
        self, ts: TSDataset, forecasts: List[pd.DataFrame], train: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare features for the ``final_model``."""
        ts = self._make_same_level(ts=ts, forecasts=forecasts)

        # Stack targets from the forecasts
        targets = [
            forecast.loc[:, pd.IndexSlice[:, "target"]].rename(
                {"target": f"regressor_target_{i}"}, level="feature", axis=1
            )
            for i, forecast in enumerate(forecasts)
        ]
        targets = pd.concat(targets, axis=1)

        # Get features from filtered_features_for_final_model
        features = pd.DataFrame()
        if self.filtered_features_for_final_model is not None:
            features_in_forecasts = [
                list(
                    set(forecast.columns.get_level_values("feature")).intersection(
                        self.filtered_features_for_final_model
                    )
                )
                for forecast in forecasts
            ]
            features = pd.concat(
                [forecast.loc[:, pd.IndexSlice[:, features_in_forecasts[i]]] for i, forecast in enumerate(forecasts)],
                axis=1,
            )
            features = features.loc[:, ~features.columns.duplicated()]
        features_df = pd.concat([features, targets], axis=1)

        # Flatten the features to fit the sklearn interface
        x = pd.concat([features_df.loc[:, segment] for segment in ts.segments], axis=0)
        if train:
            y = pd.concat(
                [ts[forecasts[0].index.min() : forecasts[0].index.max(), segment, "target"] for segment in ts.segments],
                axis=0,
            )
            return x, y
        else:
            return x, None

    def _process_forecasts(self, ts: TSDataset, forecasts: List[TSDataset]) -> TSDataset:
        forecasts_df: List[pd.DataFrame] = [forecast._df for forecast in forecasts]
        ts = self._make_same_level(ts=ts, forecasts=forecasts_df)

        x, _ = self._make_features(ts=ts, forecasts=forecasts_df, train=False)
        y = self.final_model.predict(x)
        num_segments = len(forecasts_df[0].columns.get_level_values("segment").unique())
        y = y.reshape(num_segments, -1).T
        num_timestamps = y.shape[0]

        # Format the forecast into TSDataset
        segment_col = [segment for segment in ts.segments for _ in range(num_timestamps)]
        x.loc[:, "segment"] = segment_col
        x.loc[:, "timestamp"] = x.index.values
        df_exog = TSDataset.to_dataset(x)

        df = forecasts_df[0].loc[pd.IndexSlice[:], pd.IndexSlice[:, "target"]].copy()
        df.loc[pd.IndexSlice[:], pd.IndexSlice[:, "target"]] = np.NAN

        result = TSDataset(df=df, freq=ts.freq, df_exog=df_exog, hierarchical_structure=ts.hierarchical_structure)
        result._df.loc[pd.IndexSlice[:], pd.IndexSlice[:, "target"]] = y
        return result

    def _forecast(self, ts: TSDataset, return_components: bool) -> TSDataset:
        """Make predictions.

        Compute the combination of pipelines' forecasts using ``final_model``
        """
        if return_components:
            raise NotImplementedError("Adding target components is not currently implemented!")

        forecasts = Parallel(n_jobs=self.n_jobs, **self.joblib_params)(
            delayed(self._forecast_pipeline)(pipeline=pipeline, ts=ts) for pipeline in self.pipelines
        )
        forecast = self._process_forecasts(ts=ts, forecasts=forecasts)
        return forecast

    def _predict(
        self,
        ts: TSDataset,
        start_timestamp: Union[pd.Timestamp, int],
        end_timestamp: Union[pd.Timestamp, int],
        prediction_interval: bool,
        quantiles: Sequence[float],
        return_components: bool,
    ) -> TSDataset:
        if prediction_interval:
            raise NotImplementedError(f"Ensemble {self.__class__.__name__} doesn't support prediction intervals!")
        if return_components:
            raise NotImplementedError("Adding target components is not currently implemented!")

        predictions = Parallel(n_jobs=self.n_jobs, **self.joblib_params)(
            delayed(self._predict_pipeline)(
                ts=ts, pipeline=pipeline, start_timestamp=start_timestamp, end_timestamp=end_timestamp
            )
            for pipeline in self.pipelines
        )
        prediction = self._process_forecasts(ts=ts, forecasts=predictions)
        return prediction

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get hyperparameter grid to tune.

        Parameters for pipelines have prefix "pipelines.idx.", e.g. "pipelines.0.model.alpha".

        Returns
        -------
        :
            Grid with hyperparameters.
        """
        all_params = {}
        for ind, pipeline in enumerate(self.pipelines):
            for key, value in pipeline.params_to_tune().items():
                new_key = f"pipelines.{ind}.{key}"
                all_params[new_key] = value
        return all_params
