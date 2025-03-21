from typing import List
from typing import Union
from typing import get_args

import pandas as pd

from etna import SETTINGS
from etna.datasets import TSDataset
from etna.datasets.utils import determine_num_steps
from etna.models import BATSModel
from etna.models import DeadlineMovingAverageModel
from etna.models import HoltModel
from etna.models import HoltWintersModel
from etna.models import SARIMAXModel
from etna.models import SeasonalMovingAverageModel
from etna.models import SimpleExpSmoothingModel
from etna.models import TBATSModel
from etna.models.base import ContextRequiredModelType
from etna.models.base import ModelType
from etna.transforms import IrreversibleTransform

_SUPPORTED_MODELS = Union[
    HoltWintersModel,  # full
    HoltModel,  # full
    SimpleExpSmoothingModel,  # full
    SARIMAXModel,  # full
    DeadlineMovingAverageModel,  # need to account context/prediction size
    SeasonalMovingAverageModel,  # need to account context/prediction size
    BATSModel,  # dynamic components, not reliable
    TBATSModel,  # dynamic components, not reliable
]

if SETTINGS.prophet_required:
    from etna.models import ProphetModel

    _SUPPORTED_MODELS = Union[  # type: ignore
        _SUPPORTED_MODELS,
        ProphetModel,  # full
    ]


class ModelDecomposeTransform(IrreversibleTransform):
    """Transform that uses ETNA models to estimate series decomposition.

    Note
    ----
    This transform decomposes only in-sample data. For the future timestamps it produces ``NaN``.
    For the dataset to be transformed, it should contain at least the minimum amount of in-sample timestamps that are required by the model.
    """

    def __init__(self, model: ModelType, in_column: str = "target", residuals: bool = False):
        """Init ``ModelDecomposeTransform``.

        Parameters
        ----------
        model:
            instance of the model to use for the decomposition. Note that not all models are supported. Possible selections are:

            - ``HoltWintersModel``
            - ``ProphetModel``
            - ``SARIMAXModel``
            - ``DeadlineMovingAverageModel``
            - ``SeasonalMovingAverageModel``
            - ``BATSModel``
            - ``TBATSModel``

            Currently, only the specified series itself is used for model fitting. There is no way to add additional features/regressors to the decomposition model.

        in_column:
            name of the processed column.
        residuals:
            whether to add residuals after decomposition. This guarantees that all components, including residuals, sum up to the series.

        Warning
        -------
        Options for parameter ``model`` :py:class:`etna.models.BATSModel` and :py:class:`etna.models.TBATSModel` may result in different components set compared to the initialization parameters.
        In such case, a corresponding warning would be raised.
        """
        if not isinstance(model, get_args(_SUPPORTED_MODELS)):
            raise ValueError(
                f"Model type `{type(model).__name__}` is not supported! Supported models are: {_SUPPORTED_MODELS}"
            )

        self.model = model
        self.in_column = in_column
        self.residuals = residuals

        self._first_timestamp = None
        self._last_timestamp = None

        super().__init__(required_features=[in_column])

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return []

    def _fit(self, df: pd.DataFrame):
        """Fit transform with the dataframe."""
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform provided dataframe."""
        pass

    def _prepare_ts(self, ts: TSDataset) -> TSDataset:
        """Prepare dataset for the decomposition model."""
        if self.in_column not in ts.features:
            raise KeyError(f"Column {self.in_column} is not found in features!")

        df = ts._df.loc[:, pd.IndexSlice[:, self.in_column]]
        df = df.rename(columns={self.in_column: "target"}, level="feature")

        return TSDataset(df=df, freq=ts.freq)

    def fit(self, ts: TSDataset) -> "ModelDecomposeTransform":
        """Fit the transform and the decomposition model.

        Parameters
        ----------
        ts:
            dataset to fit the transform on.

        Returns
        -------
        :
            the fitted transform instance.
        """
        self._first_timestamp = ts.timestamps.min()
        self._last_timestamp = ts.timestamps.max()

        ts = self._prepare_ts(ts=ts)

        self.model.fit(ts)
        return self

    def transform(self, ts: TSDataset) -> TSDataset:
        """Transform ``TSDataset`` inplace.

        Parameters
        ----------
        ts:
            Dataset to transform.

        Returns
        -------
        :
            Transformed ``TSDataset``.
        """
        if self._first_timestamp is None:
            raise ValueError("Transform is not fitted!")

        if ts.timestamps.min() < self._first_timestamp:
            raise ValueError(
                f"First index of the dataset to be transformed must be larger or equal than {self._first_timestamp}!"
            )

        if ts.timestamps.min() > self._last_timestamp:
            raise ValueError(
                f"Dataset to be transformed must contain historical observations in range {self._first_timestamp} - {self._last_timestamp}"
            )

        decompose_ts = self._prepare_ts(ts=ts)

        future_steps = 0
        ts_max_timestamp = decompose_ts.timestamps.max()
        if ts_max_timestamp > self._last_timestamp:
            future_steps = determine_num_steps(self._last_timestamp, ts_max_timestamp, freq=decompose_ts.freq)
            decompose_ts._df = decompose_ts._df.loc[: self._last_timestamp]

        target = decompose_ts[..., "target"].droplevel("feature", axis=1)

        if isinstance(self.model, get_args(ContextRequiredModelType)):
            decompose_ts = self.model.predict(
                decompose_ts, prediction_size=decompose_ts.size()[0] - self.model.context_size, return_components=True
            )

        else:
            decompose_ts = self.model.predict(decompose_ts, return_components=True)

        components_df = decompose_ts[..., decompose_ts.target_components_names]

        components_names = [x.replace("target_component", self.in_column) for x in decompose_ts.target_components_names]

        rename = dict(zip(decompose_ts.target_components_names, components_names))

        if self.residuals:
            components_sum = components_df.T.groupby(level="segment").sum().T
            for segment in ts.segments:
                components_df[segment, f"{self.in_column}_residuals"] = target[segment] - components_sum[segment]

        components_df.rename(columns=rename, level="feature", inplace=True)

        if future_steps > 0:
            components_df = TSDataset._expand_index(
                df=components_df, future_steps=future_steps, freq_offset=decompose_ts.freq_offset
            )

        columns_before = set(ts.features)
        columns_before &= set(components_df.columns.get_level_values("feature"))
        self._update_dataset(ts=ts, columns_before=columns_before, df_transformed=components_df)

        return ts


__all__ = ["ModelDecomposeTransform"]
