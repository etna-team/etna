import warnings
from copy import deepcopy
from typing import Sequence
from typing import cast

import pandas as pd
from typing_extensions import get_args

from etna.datasets import TSDataset
from etna.datasets.utils import timestamp_range
from etna.models.base import ContextIgnorantModelType
from etna.models.base import ContextRequiredModelType
from etna.models.base import ModelType
from etna.pipeline.base import BasePipeline
from etna.pipeline.mixins import ModelPipelineParamsToTuneMixin
from etna.pipeline.mixins import ModelPipelinePredictMixin
from etna.pipeline.mixins import SaveModelPipelineMixin
from etna.transforms import Transform


class AutoRegressivePipeline(
    ModelPipelinePredictMixin, ModelPipelineParamsToTuneMixin, SaveModelPipelineMixin, BasePipeline
):
    """
    Pipeline that make regressive models autoregressive.

    Makes forecast in several iterations, on each of them applies transforms and
    predict ``step`` values by using forecast method of model.

    See Also
    --------
    etna.pipeline.Pipeline:
        Makes forecast in one iteration.
    etna.ensembles.DirectEnsemble:
        Makes forecast by merging the forecasts of base pipelines.

    Examples
    --------
    >>> from etna.datasets import generate_periodic_df
    >>> from etna.datasets import TSDataset
    >>> from etna.models import LinearPerSegmentModel
    >>> from etna.transforms import LagTransform
    >>> df = generate_periodic_df(
    ...     periods=100,
    ...     start_time="2020-01-01",
    ...     n_segments=4,
    ...     period=7,
    ...     sigma=3
    ... )
    >>> ts = TSDataset(df, freq="D")
    >>> horizon = 7
    >>> transforms = [
    ...     LagTransform(in_column="target", lags=list(range(1, horizon+1)))
    ... ]
    >>> model = LinearPerSegmentModel()
    >>> pipeline = AutoRegressivePipeline(model, horizon, transforms, step=1)
    >>> _ = pipeline.fit(ts=ts)
    >>> forecast = pipeline.forecast()
    >>> pd.options.display.float_format = '{:,.2f}'.format
    >>> forecast[:, :, "target"]
    segment    segment_0 segment_1 segment_2 segment_3
    feature       target    target    target    target
    timestamp
    2020-04-10      9.00      9.00      4.00      6.00
    2020-04-11      5.00      2.00      7.00      9.00
    2020-04-12      0.00      4.00      7.00      9.00
    2020-04-13      0.00      5.00      9.00      7.00
    2020-04-14      1.00      2.00      1.00      6.00
    2020-04-15      5.00      7.00      4.00      7.00
    2020-04-16      8.00      6.00      2.00      0.00
    """

    def __init__(self, model: ModelType, horizon: int, transforms: Sequence[Transform] = (), step: int = 1):
        """
        Create instance of AutoRegressivePipeline with given parameters.

        Parameters
        ----------
        model:
            Instance of the etna Model
        horizon:
            Number of timestamps in the future for forecasting
        transforms:
            Sequence of the transforms
        step:
            Size of prediction for one step of forecasting
        """
        self.model = model
        self.transforms = transforms
        self.step = step
        super().__init__(horizon=horizon)

    def fit(self, ts: TSDataset, save_ts: bool = True) -> "AutoRegressivePipeline":
        """Fit the AutoRegressivePipeline.

        Fit and apply given transforms to the data, then fit the model on the transformed data.

        Method doesn't change the given ``ts``.

        Saved ``ts`` is the link to given ``ts``.

        Parameters
        ----------
        ts:
            Dataset with timeseries data.
        save_ts:
            Will ``ts`` be saved in the pipeline during ``fit``.

        Returns
        -------
        :
            Fitted Pipeline instance
        """
        cur_ts = deepcopy(ts)
        cur_ts.fit_transform(self.transforms)
        self.model.fit(cur_ts)

        if save_ts:
            self.ts = ts

        return self

    def _create_predictions_template(self, ts: TSDataset) -> pd.DataFrame:
        """Create dataframe to fill with forecasts."""
        prediction_df = ts.to_pandas(features=["target"])
        last_timestamp = prediction_df.index[-1]
        to_add_index = timestamp_range(start=last_timestamp, periods=self.horizon + 1, freq=ts.freq)[1:]
        new_index = prediction_df.index.append(to_add_index)
        index_name = prediction_df.index.name
        prediction_df = prediction_df.reindex(new_index)
        prediction_df.index.name = index_name
        return prediction_df

    def _forecast(self, ts: TSDataset, return_components: bool) -> TSDataset:
        """Make predictions."""
        prediction_df = self._create_predictions_template(ts)

        target_components_dfs = []
        for idx_start in range(0, self.horizon, self.step):
            current_step = min(self.step, self.horizon - idx_start)
            current_idx_border = ts.timestamps.shape[0] + idx_start
            current_ts = TSDataset(
                df=prediction_df.iloc[:current_idx_border],
                freq=ts.freq,
                df_exog=ts._df_exog,
                known_future=ts.known_future,
                hierarchical_structure=ts.hierarchical_structure,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    message="TSDataset freq can't be inferred",
                    action="ignore",
                )
                warnings.filterwarnings(
                    message="You probably set wrong freq.",
                    action="ignore",
                )

                if isinstance(self.model, get_args(ContextRequiredModelType)):
                    self.model = cast(ContextRequiredModelType, self.model)
                    current_ts_forecast = current_ts.make_future(
                        future_steps=current_step, tail_steps=self.model.context_size, transforms=self.transforms
                    )
                    current_ts_future = self.model.forecast(
                        ts=current_ts_forecast, prediction_size=current_step, return_components=return_components
                    )
                else:
                    self.model = cast(ContextIgnorantModelType, self.model)
                    current_ts_forecast = current_ts.make_future(future_steps=current_step, transforms=self.transforms)
                    current_ts_future = self.model.forecast(ts=current_ts_forecast, return_components=return_components)
            current_ts_future.inverse_transform(self.transforms)

            if return_components:
                target_components_dfs.append(current_ts_future.get_target_components())
                current_ts_future.drop_target_components()

            prediction_df = prediction_df.combine_first(current_ts_future.to_pandas()[prediction_df.columns])

        # construct dataset and add all features
        prediction_ts = TSDataset(
            df=prediction_df,
            freq=ts.freq,
            df_exog=ts._df_exog,
            known_future=ts.known_future,
            hierarchical_structure=ts.hierarchical_structure,
        )
        prediction_ts.transform(self.transforms)
        prediction_ts.inverse_transform(self.transforms)

        # cut only last timestamps from result dataset
        prediction_ts._df = prediction_ts._df.tail(self.horizon)
        prediction_ts._raw_df = prediction_ts._raw_df.tail(self.horizon)

        if return_components:
            target_components_df = pd.concat(target_components_dfs)
            prediction_ts.add_target_components(target_components_df=target_components_df)

        return prediction_ts
