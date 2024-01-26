import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.datasets.utils import determine_freq
from etna.datasets.utils import determine_num_steps
from etna.distributions import BaseDistribution
from etna.distributions import IntDistribution
from etna.transforms.base import IrreversibleTransform

_DEFAULT_FREQ = object()


class FourierTransform(IrreversibleTransform):
    """Adds fourier features to the dataset.

    Transform can work with two types of timestamp data: numeric and datetime.

    Transform can accept timestamp data in two forms:

    - As index. In this case the dataset index is used to compute features. The features will be the same for each segment.

    - As external column. In this case for each segment its ``in_column`` will be used to compute features.
      It is expected that for each segment we have the same type of timestamp data and for datetime type there is only one frequency.

    If we are working with external column, there is a difference in handling numeric and datetime data:

    - Numeric data can have missing values at any place.

    - Datetime data could have missing values only at the beginning of each segment.

    Notes
    -----
    To understand how transform works we recommend reading:
    `Fourier series <https://otexts.com/fpp3/useful-predictors.html#fourier-series>`_.

    If we already have a numeric data then for a mode $m$ with a period $p$ we have:

    .. math::
        & k = \\left \\lfloor \\frac{m}{2} \\right \\rfloor
        \\\\
        & f_{m, i} = \\sin \\left( \\frac{2 \\pi k i}{p} + \\frac{\\pi}{2} (m \\mod 2) \\right)

    If we have datetime data, then it first should be transformed into numeric.
    During fitting the transform saves frequency and some datetime timestamp as a reference point.
    During transformation it uses reference point to compute number of frequency units between reference point and each timestamp.
    """

    def __init__(
        self,
        period: float,
        order: Optional[int] = None,
        mods: Optional[Sequence[int]] = None,
        out_column: Optional[str] = None,
        in_column: Optional[str] = None,
    ):
        """Create instance of FourierTransform.

        Parameters
        ----------
        period:
            the period of the seasonality to capture in frequency units of time series;

            ``period`` should be >= 2
        order:
            upper order of Fourier components to include;

            ``order`` should be >= 1 and <= ceil(period/2))
        mods:
            alternative and precise way of defining which harmonics will be used,
            for example, ``order=2`` can be represented as ``mods=[1, 2, 3, 4]`` if ``period`` > 4 and
            as ``mods=[1, 2, 3]`` if 3 <= ``period`` <= 4.

            ``mods`` should be >= 1 and < period
        out_column:

            * if set, name of added column, the final name will be '{out_columnt}_{mod}';

            * if don't set, name will be ``transform.__repr__()``,
              repr will be made for transform that creates exactly this column

        in_column:
            name of column to work with:

            * if ``in_column`` is ``None`` (default) both datetime and integer timestamps are supported;

            * if ``in_column`` isn't ``None`` datetime and numeric columns are supported,
              but for datetime values only regular timestamps with some frequency are supported

        Raises
        ------
        ValueError:
            if period < 2
        ValueError:
            if both or none of order, mods is set
        ValueError:
            if order is < 1 or > ceil(period/2)
        ValueError:
            if at least one mod is < 1 or >= period
        """
        if period < 2:
            raise ValueError("Period should be at least 2")
        self.period = period

        self.order = order
        self.mods = mods
        self._mods: Sequence[int]

        if order is not None and mods is None:
            if order < 1 or order > math.ceil(period / 2):
                raise ValueError("Order should be within [1, ceil(period/2)] range")
            self._mods = [mod for mod in range(1, 2 * order + 1) if mod < period]
        elif mods is not None and order is None:
            if min(mods) < 1 or max(mods) >= period:
                raise ValueError("Every mod should be within [1, int(period)) range")
            self._mods = mods
        else:
            raise ValueError("There should be exactly one option set: order or mods")

        self.out_column = out_column
        self.in_column = in_column

        self._reference_timestamp: Union[pd.Timestamp, int, None] = None
        self._freq: Optional[str] = _DEFAULT_FREQ  # type: ignore

        if self.in_column is None:
            self.in_column_regressor: Optional[bool] = True
        else:
            self.in_column_regressor = None

        if in_column is None:
            required_features = ["target"]
        else:
            required_features = [in_column]
        super().__init__(required_features=required_features)

    def _get_column_name(self, mod: int) -> str:
        if self.out_column is None:
            return f"{FourierTransform(period=self.period, mods=[mod]).__repr__()}"
        else:
            return f"{self.out_column}_{mod}"

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        if self.in_column_regressor is None:
            raise ValueError("Fit the transform to get the correct regressors info!")

        if not self.in_column_regressor:
            return []

        output_columns = [self._get_column_name(mod=mod) for mod in self._mods]
        return output_columns

    def fit(self, ts: TSDataset) -> "FourierTransform":
        """Fit the transform."""
        if self.in_column is None:
            self._freq = ts.freq
            self.in_column_regressor = True
        else:
            self.in_column_regressor = self.in_column in ts.regressors
        super().fit(ts)
        return self

    def _validate_external_timestamp(self, df: pd.DataFrame):
        df = df.droplevel("feature", axis=1)

        # here we are assuming that every segment has the same timestamp dtype
        timestamp_dtype = df.dtypes.iloc[0]
        if not pd.api.types.is_datetime64_dtype(timestamp_dtype):
            return

        segments = df.columns.unique()
        freq_values = set()
        for segment in segments:
            timestamp = df[segment]
            timestamp = timestamp.loc[timestamp.first_valid_index() :]
            if len(timestamp) >= 3:
                cur_freq = pd.infer_freq(timestamp)
                if cur_freq is None:
                    raise ValueError(
                        f"Invalid in_column values! Datetime values should be regular timestamps with some frequency. "
                        f"This doesn't hold for segment {segment}"
                    )
                freq_values.add(cur_freq)

        if len(freq_values) > 1:
            raise ValueError(
                f"Invalid in_column values! Datetime values should have the same frequency for every segment. "
                f"Discovered frequencies: {freq_values}"
            )

    def _infer_external_freq(self, df: pd.DataFrame) -> Optional[str]:
        df = df.droplevel("feature", axis=1)

        # here we are assuming that every segment has the same timestamp dtype
        timestamp_dtype = df.dtypes.iloc[0]
        if not pd.api.types.is_datetime64_dtype(timestamp_dtype):
            return None

        sample_segment = df.columns[0]
        sample_timestamp = df[sample_segment]
        sample_timestamp = sample_timestamp.loc[sample_timestamp.first_valid_index() :]
        result = determine_freq(sample_timestamp)
        return result

    def _infer_external_reference_timestamp(self, df: pd.DataFrame) -> Union[pd.Timestamp, int]:
        # here we are assuming that every segment has the same timestamp dtype
        timestamp_dtype = df.dtypes.iloc[0]
        if not pd.api.types.is_datetime64_dtype(timestamp_dtype):
            return 0

        sample_segment = df.columns[0]
        sample_timestamp = df[sample_segment]
        reference_timestamp = sample_timestamp.loc[sample_timestamp.first_valid_index()]
        return reference_timestamp

    def _fit(self, df: pd.DataFrame) -> "FourierTransform":
        """Fit method does nothing and is kept for compatibility.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        result:
        """
        if self.in_column is None:
            self._reference_timestamp = df.index[0]
        else:
            self._validate_external_timestamp(df)
            self._freq = self._infer_external_freq(df)
            self._reference_timestamp = self._infer_external_reference_timestamp(df)
        return self

    @staticmethod
    def _construct_answer_for_index(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        dataframes = []
        for seg in df.columns.get_level_values("segment").unique():
            tmp = df[seg].join(features)
            _idx = tmp.columns.to_frame()
            _idx.insert(0, "segment", seg)
            tmp.columns = pd.MultiIndex.from_frame(_idx)
            dataframes.append(tmp)

        result = pd.concat(dataframes, axis=1).sort_index(axis=1)
        result.columns.names = ["segment", "feature"]
        return result

    def _compute_features(self, timestamp: pd.Series) -> pd.DataFrame:
        features = pd.DataFrame(index=timestamp.index)
        elapsed = timestamp / self.period

        for mod in self._mods:
            order = (mod + 1) // 2
            is_cos = mod % 2 == 0

            features[self._get_column_name(mod)] = np.sin(2 * np.pi * order * elapsed + np.pi / 2 * is_cos)

        return features

    def _convert_regular_timestamp_datetime_to_numeric(
        self, timestamp: pd.Series, reference_timestamp: pd.Timestamp, freq: Optional[str]
    ) -> pd.Series:
        # we should always align timestamps to some fixed point
        end_timestamp = timestamp[-1]
        if end_timestamp >= reference_timestamp:
            end_idx = determine_num_steps(start_timestamp=reference_timestamp, end_timestamp=end_timestamp, freq=freq)
        else:
            end_idx = -determine_num_steps(start_timestamp=end_timestamp, end_timestamp=reference_timestamp, freq=freq)

        numeric_timestamp = pd.Series(np.arange(end_idx - len(timestamp) + 1, end_idx + 1))

        return numeric_timestamp

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add harmonics to the dataset.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result:
            transformed dataframe
        """
        if self._freq is _DEFAULT_FREQ:
            raise ValueError("The transform isn't fitted!")

        if self.in_column is None:
            if pd.api.types.is_integer_dtype(df.index.dtype):
                timestamp = df.index.to_series()
            else:
                timestamp = self._convert_regular_timestamp_datetime_to_numeric(
                    timestamp=df.index, reference_timestamp=self._reference_timestamp, freq=self._freq
                )
            features = self._compute_features(timestamp=timestamp)
            features.index = df.index
            result = self._construct_answer_for_index(df=df, features=features)
        else:
            # here we are assuming that every segment has the same timestamp dtype
            timestamp_dtype = df.dtypes.iloc[0]
            if pd.api.types.is_numeric_dtype(timestamp_dtype):
                flat_timestamp_df = TSDataset.to_flatten(df=df)
                timestamp = flat_timestamp_df[self.in_column]
            else:
                self._validate_external_timestamp(df=df)
                segments = df.columns.get_level_values("segment").unique()
                int_timestamp_values = []
                for segment in segments:
                    segment_timestamp = df[segment][self.in_column]
                    int_segment_timestamp = self._convert_regular_timestamp_datetime_to_numeric(
                        timestamp=segment_timestamp,
                        reference_timestamp=self._reference_timestamp,
                        freq=self._freq,
                    )
                    int_timestamp_values.append(int_segment_timestamp)

                df_int_timestamp = pd.DataFrame(np.array(int_timestamp_values).T, index=df.index, columns=df.columns)
                flat_timestamp_df = TSDataset.to_flatten(df=df_int_timestamp)
                timestamp = flat_timestamp_df[self.in_column]

            features = self._compute_features(timestamp=timestamp)
            features["timestamp"] = flat_timestamp_df["timestamp"]
            features["segment"] = flat_timestamp_df["segment"]
            wide_df = TSDataset.to_dataset(features)
            result = pd.concat([df, wide_df], axis=1).sort_index(axis=1)
        return result

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        If ``self.order`` is set then this grid tunes ``order`` parameter:
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        if self.mods is not None:
            return {}

        max_value = math.ceil(self.period / 2)
        return {"order": IntDistribution(low=1, high=max_value, log=True)}
