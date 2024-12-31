import math
import warnings
from copy import copy
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from deprecated import deprecated
from matplotlib import pyplot as plt
from typing_extensions import Literal

from etna import SETTINGS
from etna.datasets.hierarchical_structure import HierarchicalStructure
from etna.datasets.utils import DataFrameFormat
from etna.datasets.utils import _check_timestamp_param
from etna.datasets.utils import _TorchDataset
from etna.datasets.utils import apply_alignment
from etna.datasets.utils import get_level_dataframe
from etna.datasets.utils import infer_alignment
from etna.datasets.utils import inverse_transform_target_components
from etna.datasets.utils import make_timestamp_df_from_alignment
from etna.datasets.utils import timestamp_range
from etna.loggers import tslogger

if TYPE_CHECKING:
    from etna.transforms.base import Transform

if SETTINGS.torch_required:
    from torch.utils.data import Dataset


class TSDataset:
    """TSDataset is the main class to handle your time series data.

    It prepares the series for exploration analyzing, implements feature generation with Transforms
    and generation of future points.

    Notes
    -----
    TSDataset supports custom indexing and slicing method.
    It maybe done through these interface: ``TSDataset[timestamp, segment, column]``
    If at the start of the period dataset contains NaN those timestamps will be removed.

    During creation segment is casted to string type.

    Examples
    --------
    >>> from etna.datasets import generate_const_df
    >>> df = generate_const_df(periods=30, start_time="2021-06-01", n_segments=2, scale=1)
    >>> ts = TSDataset(df, "D")
    >>> ts["2021-06-01":"2021-06-07", "segment_0", "target"]
    timestamp
    2021-06-01    1.0
    2021-06-02    1.0
    2021-06-03    1.0
    2021-06-04    1.0
    2021-06-05    1.0
    2021-06-06    1.0
    2021-06-07    1.0
    Freq: D, Name: (segment_0, target), dtype: float64

    >>> from etna.datasets import generate_ar_df
    >>> pd.options.display.float_format = '{:,.2f}'.format
    >>> df_to_forecast = generate_ar_df(100, start_time="2021-01-01", n_segments=1)
    >>> df_regressors = generate_ar_df(120, start_time="2021-01-01", n_segments=5)
    >>> df_regressors = df_regressors.pivot(index="timestamp", columns="segment").reset_index()
    >>> df_regressors.columns = ["timestamp"] + [f"regressor_{i}" for i in range(5)]
    >>> df_regressors["segment"] = "segment_0"
    >>> tsdataset = TSDataset(df=df_to_forecast, freq="D", df_exog=df_regressors, known_future="all")
    >>> tsdataset.df.head(5)
    segment      segment_0
    feature    regressor_0 regressor_1 regressor_2 regressor_3 regressor_4 target
    timestamp
    2021-01-01        1.62       -0.02       -0.50       -0.56        0.52   1.62
    2021-01-02        1.01       -0.80       -0.81        0.38       -0.60   1.01
    2021-01-03        0.48        0.47       -0.81       -1.56       -1.37   0.48
    2021-01-04       -0.59        2.44       -2.21       -1.21       -0.69  -0.59
    2021-01-05        0.28        0.58       -3.07       -1.45        0.77   0.28

    >>> from etna.datasets import generate_hierarchical_df
    >>> pd.options.display.width = 0
    >>> df = generate_hierarchical_df(periods=100, n_segments=[2, 4], start_time="2021-01-01",)
    >>> df, hierarchical_structure = TSDataset.to_hierarchical_dataset(df=df, level_columns=["level_0", "level_1"])
    >>> tsdataset = TSDataset(df=df, freq="D", hierarchical_structure=hierarchical_structure)
    >>> tsdataset.df.head(5)
    segment    l0s0_l1s3 l0s1_l1s0 l0s1_l1s1 l0s1_l1s2
    feature       target    target    target    target
    timestamp
    2021-01-01      2.07      1.62     -0.45     -0.40
    2021-01-02      0.59      1.01      0.78      0.42
    2021-01-03     -0.24      0.48      1.18     -0.14
    2021-01-04     -1.12     -0.59      1.77      1.82
    2021-01-05     -1.40      0.28      0.68      0.48
    """

    #: Shortcut for :py:class:`pd.core.indexing.IndexSlice`
    idx = pd.IndexSlice

    def __init__(
        self,
        df: pd.DataFrame,
        freq: Optional[str],
        df_exog: Optional[pd.DataFrame] = None,
        known_future: Union[Literal["all"], Sequence] = (),
        hierarchical_structure: Optional[HierarchicalStructure] = None,
    ):
        """Init TSDataset.

        Parameters
        ----------
        df:
            dataframe with timeseries in a wide or long format: :py:class:`~etna.datasets.utils.DataFrameFormat`;
            it is expected that ``df`` has feature named "target"
        freq:
            frequency of timestamp in df, possible values:

            - `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
              for datetime timestamp

            - None for integer timestamp

        df_exog:
            dataframe with exogenous data in a wide or long format: :py:class:`~etna.datasets.utils.DataFrameFormat`
        known_future:
            columns in ``df_exog[known_future]`` that are regressors,
            if "all" value is given, all columns are meant to be regressors
        hierarchical_structure:
            Structure of the levels in the hierarchy. If None, there is no hierarchical structure in the dataset.
        """
        self.freq = freq
        self.df_exog = None
        self.raw_df = self._prepare_df(df=df.copy(deep=True), freq=freq)
        self.df = self.raw_df.copy(deep=True)

        self.hierarchical_structure = hierarchical_structure
        self.current_df_level: Optional[str] = self._get_dataframe_level(df=self.df)
        self.current_df_exog_level: Optional[str] = None

        if df_exog is not None:
            self.df_exog = self._prepare_df_exog(df_exog=df_exog.copy(deep=True), freq=freq)

            self.known_future = self._check_known_future(known_future, self.df_exog)
            self._regressors = copy(self.known_future)

            self.current_df_exog_level = self._get_dataframe_level(df=self.df_exog)
            if self.current_df_level == self.current_df_exog_level:
                self.df = self._merge_exog(df=self.df)
        else:
            self.known_future = self._check_known_future(known_future, df_exog)
            self._regressors = copy(self.known_future)

        self._target_components_names: Tuple[str, ...] = tuple()
        self._prediction_intervals_names: Tuple[str, ...] = tuple()

        self.df = self.df.sort_index(axis=1, level=("segment", "feature"))

    @classmethod
    def create_from_misaligned(
        cls,
        df: pd.DataFrame,
        freq: Optional[str],
        df_exog: Optional[pd.DataFrame] = None,
        known_future: Union[Literal["all"], Sequence] = (),
        future_steps: int = 1,
        original_timestamp_name: str = "external_timestamp",
    ) -> "TSDataset":
        """Make TSDataset from misaligned data by realigning it according to inferred alignment in ``df``.

        This method:
        - Infers alignment using :py:func:`~etna.datasets.utils.infer_alignment`;
        - Realigns ``df`` and ``df_exog`` using inferred alignment using :py:func:`~etna.datasets.utils.apply_alignment`;
        - Creates exog feature with original timestamp using :py:func:`~etna.datasets.utils.make_timestamp_df_from_alignment`;
        - Creates TSDataset from these data.

        This method doesn't work with ``hierarchical_structure``, because it doesn't make much sense.

        Parameters
        ----------
        df:
            dataframe with timeseries in a long format: :py:class:`~etna.datasets.utils.DataFrameFormat`;
            it is expected that ``df`` has feature named "target"
        freq:
            frequency of timestamp in df, possible values:

            - `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
              for datetime timestamp

            - None for integer timestamp

        df_exog:
            dataframe with exogenous data in a long format: :py:class:`~etna.datasets.utils.DataFrameFormat`
        known_future:
            columns in ``df_exog[known_future]`` that are regressors,
            if "all" value is given, all columns are meant to be regressors
        future_steps:
            determines on how many steps original timestamp should be extended into the future
            before adding into ``df_exog``; expected to be positive
        original_timestamp_name:
            name for original timestamp column to add it into ``df_exog``

        Returns
        -------
        :
            Created TSDataset.

        Raises
        ------
        ValueError:
            If ``future_steps`` is not positive.
        ValueError:
            If ``original_timestamp_name`` intersects with columns in ``df_exog``.
        ValueError:
            Parameter ``df`` isn't in a long format.
        ValueError:
            Parameter ``df_exog`` isn't in a long format if it set.
        """
        if future_steps <= 0:
            raise ValueError("Parameter future_steps should be positive!")
        if df_exog is not None and original_timestamp_name in df_exog.columns:
            raise ValueError("Parameter original_timestamp_name shouldn't intersect with columns in df_exog!")

        alignment = infer_alignment(df)
        df_realigned = apply_alignment(df=df, alignment=alignment)
        df_realigned = TSDataset.to_dataset(df_realigned)

        timestamp_start = df_realigned.index[0]
        periods = len(df_realigned) + future_steps
        timestamp_df = make_timestamp_df_from_alignment(
            alignment=alignment,
            start=timestamp_start,
            periods=periods,
            freq=freq,
            timestamp_name=original_timestamp_name,
        )
        timestamp_df = TSDataset.to_dataset(timestamp_df)

        if df_exog is not None:
            df_exog_realigned = apply_alignment(df=df_exog, alignment=alignment)
            df_exog_realigned = TSDataset.to_dataset(df_exog_realigned)

            df_exog_realigned = df_exog_realigned.join(timestamp_df, how="outer")
        else:
            df_exog_realigned = timestamp_df

        known_future_realigned: Union[Literal["all"], Sequence]
        if known_future != "all":
            known_future_realigned = list(known_future)
            known_future_realigned.append(original_timestamp_name)
        else:
            known_future_realigned = "all"

        return TSDataset(
            df=df_realigned,
            df_exog=df_exog_realigned,
            freq=None,
            known_future=known_future_realigned,
            hierarchical_structure=None,
        )

    def _get_dataframe_level(self, df: pd.DataFrame) -> Optional[str]:
        """Return the level of the passed dataframe in hierarchical structure."""
        if self.hierarchical_structure is None:
            return None

        df_segments = df.columns.get_level_values("segment").unique()
        segment_levels = {self.hierarchical_structure.get_segment_level(segment=segment) for segment in df_segments}
        if len(segment_levels) != 1:
            raise ValueError("Segments in dataframe are from more than 1 hierarchical levels!")

        df_level = segment_levels.pop()
        level_segments = self.hierarchical_structure.get_level_segments(level_name=df_level)
        if len(df_segments) != len(level_segments):
            raise ValueError("Some segments of hierarchical level are missing in dataframe!")

        return df_level

    def transform(self, transforms: Sequence["Transform"]):
        """Apply given transform to the data."""
        self._check_endings(warning=True)
        for transform in transforms:
            tslogger.log(f"Transform {repr(transform)} is applied to dataset")
            transform.transform(self)

    def fit_transform(self, transforms: Sequence["Transform"]):
        """Fit and apply given transforms to the data."""
        self._check_endings(warning=True)
        for transform in transforms:
            tslogger.log(f"Transform {repr(transform)} is applied to dataset")
            transform.fit_transform(self)

    @staticmethod
    def _cast_segment_to_str(df: pd.DataFrame) -> pd.DataFrame:
        columns_frame = df.columns.to_frame()
        dtype = columns_frame["segment"].dtype
        if not pd.api.types.is_object_dtype(dtype):
            warnings.warn(
                f"Segment values doesn't have string type, given type is {dtype}. "
                f"Segments will be converted to string."
            )
            columns_frame["segment"] = columns_frame["segment"].astype(str)
        df.columns = pd.MultiIndex.from_frame(columns_frame)
        return df

    @staticmethod
    def _cast_index_to_datetime(df: pd.DataFrame, freq: str) -> pd.DataFrame:
        if pd.api.types.is_numeric_dtype(df.index):
            warnings.warn(
                f"Timestamp contains numeric values, and given freq is {freq}. Timestamp will be converted to datetime."
            )
        df.index = pd.to_datetime(df.index)
        return df

    @classmethod
    def _prepare_df(cls, df: pd.DataFrame, freq: Optional[str]) -> pd.DataFrame:
        df_format = DataFrameFormat.determine(df)
        if df_format is DataFrameFormat.long:
            df = cls.to_dataset(df)

        # cast segment to str type
        cls._cast_segment_to_str(df)

        # handle freq
        if freq is None:
            if not pd.api.types.is_integer_dtype(df.index.dtype):
                raise ValueError("You set wrong freq. Data contains datetime index, not integer.")

            new_index = np.arange(df.index.min(), df.index.max() + 1)
            index_name = df.index.name
            df = df.reindex(new_index)
            df.index.name = index_name

        else:
            cls._cast_index_to_datetime(df, freq)
            try:
                inferred_freq = pd.infer_freq(df.index)
            except ValueError:
                warnings.warn("TSDataset freq can't be inferred")
                inferred_freq = None

            if inferred_freq is not None and inferred_freq != freq:
                warnings.warn(
                    f"You probably set wrong freq. Discovered freq in you data is {inferred_freq}, you set {freq}"
                )

            df = df.asfreq(freq)

        return df

    @classmethod
    def _prepare_df_exog(cls, df_exog: pd.DataFrame, freq: Optional[str]) -> pd.DataFrame:
        df_format = DataFrameFormat.determine(df_exog)
        if df_format is DataFrameFormat.long:
            df_exog = cls.to_dataset(df_exog)

        df_exog = cls._cast_segment_to_str(df=df_exog)
        if freq is not None:
            cls._cast_index_to_datetime(df_exog, freq)

        return df_exog

    def __repr__(self):
        return self.df.__repr__()

    def _repr_html_(self):
        return self.df._repr_html_()

    def __getitem__(self, item):
        if isinstance(item, slice) or isinstance(item, str):
            df = self.df.loc[self.idx[item]]
        elif len(item) == 2 and item[0] is Ellipsis:
            df = self.df.loc[self.idx[:], self.idx[:, item[1]]]
        elif len(item) == 2 and item[1] is Ellipsis:
            df = self.df.loc[self.idx[item[0]]]
        else:
            df = self.df.loc[self.idx[item[0]], self.idx[item[1], item[2]]]
        first_valid_idx = df.first_valid_index()
        df = df.loc[first_valid_idx:]
        return df

    @staticmethod
    def _expand_index(df: pd.DataFrame, freq: Optional[str], future_steps: int) -> pd.DataFrame:
        to_add_index = timestamp_range(start=df.index[-1], periods=future_steps + 1, freq=freq)[1:]
        new_index = df.index.append(to_add_index)
        index_name = df.index.name
        df = df.reindex(new_index)
        df.index.name = index_name
        return df

    def make_future(
        self, future_steps: int, transforms: Sequence["Transform"] = (), tail_steps: int = 0
    ) -> "TSDataset":
        """Return new TSDataset with features extended into the future.

        The result dataset doesn't contain quantiles and target components.

        Parameters
        ----------
        future_steps:
            number of steps to extend dataset into the future.
        transforms:
            sequence of transforms to be applied.
        tail_steps:
            number of steps to keep from the tail of the original dataset.

        Returns
        -------
        :
            dataset with features extended into the.

        Examples
        --------
        >>> from etna.datasets import generate_const_df
        >>> df = generate_const_df(
        ...    periods=30, start_time="2021-06-01",
        ...    n_segments=2, scale=1
        ... )
        >>> df_regressors = pd.DataFrame({
        ...     "timestamp": list(pd.date_range("2021-06-01", periods=40))*2,
        ...     "regressor_1": np.arange(80), "regressor_2": np.arange(80) + 5,
        ...     "segment": ["segment_0"]*40 + ["segment_1"]*40
        ... })
        >>> ts = TSDataset(
        ...     df, "D", df_exog=df_regressors, known_future="all"
        ... )
        >>> ts.make_future(4)
        segment      segment_0                      segment_1
        feature    regressor_1 regressor_2 target regressor_1 regressor_2 target
        timestamp
        2021-07-01          30          35    NaN          70          75    NaN
        2021-07-02          31          36    NaN          71          76    NaN
        2021-07-03          32          37    NaN          72          77    NaN
        2021-07-04          33          38    NaN          73          78    NaN
        """
        self._check_endings(warning=True)
        df = self._expand_index(df=self.raw_df, freq=self.freq, future_steps=future_steps)

        if self.df_exog is not None and self.current_df_level == self.current_df_exog_level:
            df = self._merge_exog(df=df)

            # check if we have enough values in regressors
            # TODO: check performance
            if self.regressors:
                future_index = df.index.difference(self.index)
                for segment in self.segments:
                    regressors_index = self.df_exog.loc[:, pd.IndexSlice[segment, self.regressors]].index
                    if not np.all(future_index.isin(regressors_index)):
                        warnings.warn(
                            f"Some regressors don't have enough values in segment {segment}, "
                            f"NaN-s will be used for missing values"
                        )

        # remove components and quantiles
        # it should be done if we have quantiles and components in raw_df
        if len(self.target_components_names) > 0:
            df_components_columns = set(self.target_components_names).intersection(
                df.columns.get_level_values(level="feature")
            )
            if len(df_components_columns) > 0:
                df = df.drop(columns=list(df_components_columns), level="feature")

        if len(self.prediction_intervals_names) > 0:
            df_intervals_columns = set(self.prediction_intervals_names).intersection(
                df.columns.get_level_values(level="feature")
            )
            if len(df_intervals_columns) > 0:
                df = df.drop(columns=list(df_intervals_columns), level="feature")

        # Here only df is required, other metadata is not necessary to build the dataset
        ts = TSDataset(df=df, freq=self.freq)
        for transform in transforms:
            tslogger.log(f"Transform {repr(transform)} is applied to dataset")
            transform.transform(ts)
        df = ts.to_pandas()

        future_dataset = df.tail(future_steps + tail_steps).copy(deep=True)

        future_dataset = future_dataset.sort_index(axis=1, level=(0, 1))
        future_ts = TSDataset(df=future_dataset, freq=self.freq, hierarchical_structure=self.hierarchical_structure)

        # can't put known_future into constructor, _check_known_future fails with df_exog=None
        future_ts.known_future = deepcopy(self.known_future)
        future_ts._regressors = deepcopy(self.regressors)
        if self.df_exog is not None:
            future_ts.df_exog = self.df_exog.copy(deep=True)
        return future_ts

    def tsdataset_idx_slice(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> "TSDataset":
        """Return new TSDataset with integer-location based indexing.

        Parameters
        ----------
        start_idx:
            starting integer index (counting from 0) of the slice.
        end_idx:
            last integer index (counting from 0) of the slice.

        Returns
        -------
        :
            TSDataset based on indexing slice.
        """
        df_slice = self.df.iloc[start_idx:end_idx].copy(deep=True)
        tsdataset_slice = TSDataset(df=df_slice, freq=self.freq)
        # can't put known_future into constructor, _check_known_future fails with df_exog=None
        tsdataset_slice.known_future = deepcopy(self.known_future)
        tsdataset_slice._regressors = deepcopy(self.regressors)
        if self.df_exog is not None:
            tsdataset_slice.df_exog = self.df_exog.copy(deep=True)
        tsdataset_slice._target_components_names = deepcopy(self._target_components_names)
        tsdataset_slice._prediction_intervals_names = deepcopy(self._prediction_intervals_names)
        return tsdataset_slice

    @staticmethod
    def _check_known_future(
        known_future: Union[Literal["all"], Sequence], df_exog: Optional[pd.DataFrame]
    ) -> List[str]:
        """Check that ``known_future`` corresponds to ``df_exog`` and returns initial list of regressors."""
        if df_exog is None:
            exog_columns = set()
        else:
            exog_columns = set(df_exog.columns.get_level_values("feature"))

        if isinstance(known_future, str):
            if known_future == "all":
                return sorted(exog_columns)
            else:
                raise ValueError("The only possible literal is 'all'")
        else:
            known_future_unique = set(known_future)
            if not known_future_unique.issubset(exog_columns):
                raise ValueError(
                    f"Some features in known_future are not present in df_exog: "
                    f"{known_future_unique.difference(exog_columns)}"
                )
            else:
                return sorted(known_future_unique)

    @staticmethod
    def _check_regressors(df: pd.DataFrame, df_regressors: pd.DataFrame):
        """Check that regressors begin not later than in ``df`` and end later than in ``df``."""
        if df_regressors.shape[1] == 0:
            return
        # TODO: check performance
        df_segments = df.columns.get_level_values("segment")
        for segment in df_segments:
            target_min = df[segment]["target"].first_valid_index()
            target_min = pd.NaT if target_min is None else target_min
            target_max = df[segment]["target"].last_valid_index()
            target_max = pd.NaT if target_max is None else target_max

            exog_series_min = df_regressors[segment].first_valid_index()
            exog_series_min = pd.NaT if exog_series_min is None else exog_series_min
            exog_series_max = df_regressors[segment].last_valid_index()
            exog_series_max = pd.NaT if exog_series_max is None else exog_series_max
            if target_min < exog_series_min:
                raise ValueError(
                    f"All the regressor series should start not later than corresponding 'target'."
                    f"Series of segment {segment} have not enough history: "
                    f"{target_min} < {exog_series_min}."
                )
            if target_max >= exog_series_max:
                raise ValueError(
                    f"All the regressor series should finish later than corresponding 'target'."
                    f"Series of segment {segment} have not enough history: "
                    f"{target_max} >= {exog_series_max}."
                )

    def _merge_exog(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.df_exog is None:
            raise ValueError("Something went wrong, Trying to merge df_exog which is None!")

        # TODO: this check could probably be skipped at make_future
        df_regressors = self.df_exog.loc[:, pd.IndexSlice[:, self.known_future]]
        self._check_regressors(df=df, df_regressors=df_regressors)

        df = pd.concat((df, self.df_exog), axis=1).loc[df.index].sort_index(axis=1, level=(0, 1))
        return df

    def _check_endings(self, warning=False):
        """Check that all targets ends at the same timestamp."""
        max_index = self.df.index.max()
        if np.any(pd.isna(self.df.loc[max_index, pd.IndexSlice[:, "target"]])):
            if warning:
                warnings.warn(
                    "Segments contains NaNs in the last timestamps. "
                    "Some of the transforms might work incorrectly or even fail. "
                    "Try to start using integer timestamp and align the segments."
                )
            else:
                raise ValueError("All segments should end at the same timestamp")

    def _inverse_transform_target_components(self, target_components_df: pd.DataFrame, target_df: pd.DataFrame):
        """Inverse transform target components in dataset with inverse transformed target."""
        self.drop_target_components()
        inverse_transformed_target_components_df = inverse_transform_target_components(
            target_components_df=target_components_df,
            target_df=target_df,
            inverse_transformed_target_df=self.to_pandas(features=["target"]),
        )
        self.add_target_components(target_components_df=inverse_transformed_target_components_df)

    def inverse_transform(self, transforms: Sequence["Transform"]):
        """Apply inverse transform method of transforms to the data.

        Applied in reversed order.
        """
        # TODO: return regressors after inverse_transform
        # Logic with target components is here for performance reasons.
        # This way we avoid doing the inverse transformation for components several times.
        target_components_present = len(self.target_components_names) > 0
        target_df, target_components_df = None, None
        if target_components_present:
            target_df = self.to_pandas(features=["target"])
            target_components_df = self.get_target_components()
            self.drop_target_components()

        try:
            for transform in reversed(transforms):
                tslogger.log(f"Inverse transform {repr(transform)} is applied to dataset")
                transform.inverse_transform(self)
        finally:
            if target_components_present:
                self._inverse_transform_target_components(
                    target_components_df=target_components_df, target_df=target_df
                )

    @property
    def segments(self) -> List[str]:
        """Get list of all segments in dataset.

        Examples
        --------
        >>> from etna.datasets import generate_const_df
        >>> df = generate_const_df(
        ...    periods=30, start_time="2021-06-01",
        ...    n_segments=2, scale=1
        ... )
        >>> ts = TSDataset(df, "D")
        >>> ts.segments
        ['segment_0', 'segment_1']
        """
        return self.df.columns.get_level_values("segment").unique().tolist()

    @property
    def regressors(self) -> List[str]:
        """Get list of all regressors across all segments in dataset.

        Examples
        --------
        >>> from etna.datasets import generate_const_df
        >>> df = generate_const_df(
        ...    periods=30, start_time="2021-06-01",
        ...    n_segments=2, scale=1
        ... )
        >>> regressors_timestamp = pd.date_range(start="2021-06-01", periods=50)
        >>> df_regressors_1 = pd.DataFrame(
        ...     {"timestamp": regressors_timestamp, "regressor_1": 1, "segment": "segment_0"}
        ... )
        >>> df_regressors_2 = pd.DataFrame(
        ...     {"timestamp": regressors_timestamp, "regressor_1": 2, "segment": "segment_1"}
        ... )
        >>> df_exog = pd.concat([df_regressors_1, df_regressors_2], ignore_index=True)
        >>> ts = TSDataset(
        ...     df, df_exog=df_exog, freq="D", known_future="all"
        ... )
        >>> ts.regressors
        ['regressor_1']
        """
        return self._regressors

    @property
    def features(self) -> List[str]:
        """Get list of all features across all segments in dataset.

        All features include initial exogenous data, generated features, target, target components, prediction intervals.
        The order of features in returned list isn't specified.

        If different segments have different subset of features, then the union of features is returned.

        Returns
        -------
        :
            List of features.
        """
        return self.df.columns.get_level_values("feature").unique().tolist()

    @property
    def target_components_names(self) -> Tuple[str, ...]:
        """Get tuple with target components names. Components sum up to target. Return the empty tuple in case of components absence."""
        return self._target_components_names

    @property
    @deprecated(
        reason="Usage of this property may mislead while accessing prediction intervals, so it will be removed. Use `prediction_intervals_names` property to access intervals names!",
        version="3.0",
    )
    def target_quantiles_names(self) -> Tuple[str, ...]:
        """Get tuple with target quantiles names. Return the empty tuple in case of quantile absence."""
        return self._prediction_intervals_names

    @property
    def prediction_intervals_names(self) -> Tuple[str, ...]:
        """Get a tuple with prediction intervals names. Return an empty tuple in the case of intervals absence."""
        return self._prediction_intervals_names

    def plot(
        self,
        n_segments: int = 10,
        column: str = "target",
        segments: Optional[Sequence[str]] = None,
        start: Union[pd.Timestamp, int, str, None] = None,
        end: Union[pd.Timestamp, int, str, None] = None,
        seed: int = 1,
        figsize: Tuple[int, int] = (10, 5),
    ):
        """Plot of random or chosen segments.

        Parameters
        ----------
        n_segments:
            number of random segments to plot
        column:
            feature to plot
        segments:
            segments to plot
        seed:
            seed for local random state
        start:
            start plot from this timestamp
        end:
            end plot at this timestamp
        figsize:
            size of the figure per subplot with one segment in inches

        Raises
        ------
        ValueError:
            Incorrect type of ``start`` or ``end`` is used according to ``freq``
        """
        if segments is None:
            segments = self.segments
            k = min(n_segments, len(segments))
        else:
            k = len(segments)
        columns_num = min(2, k)
        rows_num = math.ceil(k / columns_num)

        start = _check_timestamp_param(param=start, param_name="start", freq=self.freq)
        end = _check_timestamp_param(param=end, param_name="end", freq=self.freq)

        start = self.df.index.min() if start is None else start
        end = self.df.index.max() if end is None else end

        figsize = (figsize[0] * columns_num, figsize[1] * rows_num)
        _, ax = plt.subplots(rows_num, columns_num, figsize=figsize, squeeze=False)
        ax = ax.ravel()
        rnd_state = np.random.RandomState(seed)
        for i, segment in enumerate(sorted(rnd_state.choice(segments, size=k, replace=False))):
            df_slice = self[start:end, segment, column]  # type: ignore
            ax[i].plot(df_slice.index, df_slice.values)
            ax[i].set_title(segment)
            ax[i].grid()

    @staticmethod
    def to_flatten(df: pd.DataFrame, features: Union[Literal["all"], Sequence[str]] = "all") -> pd.DataFrame:
        """Return pandas DataFrame in a long format.

        The order of columns is (timestamp, segment, target,
        features in alphabetical order).

        Parameters
        ----------
        df:
            DataFrame in ETNA format.
        features:
            List of features to return.
            If "all", return all the features in the dataset.
            Always return columns with timestamp and segemnt.
        Returns
        -------
        pd.DataFrame:
            dataframe with TSDataset data

        Examples
        --------
        >>> from etna.datasets import generate_const_df
        >>> df = generate_const_df(
        ...    periods=30, start_time="2021-06-01",
        ...    n_segments=2, scale=1
        ... )
        >>> df.head(5)
            timestamp    segment  target
        0  2021-06-01  segment_0    1.00
        1  2021-06-02  segment_0    1.00
        2  2021-06-03  segment_0    1.00
        3  2021-06-04  segment_0    1.00
        4  2021-06-05  segment_0    1.00
        >>> df_wide = TSDataset.to_dataset(df)
        >>> TSDataset.to_flatten(df_wide).head(5)
           timestamp    segment  target
        0 2021-06-01  segment_0    1.0
        1 2021-06-02  segment_0    1.0
        2 2021-06-03  segment_0    1.0
        3 2021-06-04  segment_0    1.0
        4 2021-06-05  segment_0    1.0
        """
        segments = df.columns.get_level_values("segment").unique()
        dtypes = df.dtypes
        category_columns = dtypes[dtypes == "category"].index.get_level_values(1).unique()
        if isinstance(features, str):
            if features != "all":
                raise ValueError("The only possible literal is 'all'")
        else:
            df = df.loc[:, pd.IndexSlice[segments, features]].copy()
        columns = df.columns.get_level_values("feature").unique()

        # flatten dataframe
        df_dict: Dict[str, Any] = {}
        df_dict["timestamp"] = np.tile(df.index, len(segments))
        df_dict["segment"] = np.repeat(segments, len(df.index))
        if "target" in columns:
            # set this value to lock position of key "target" in output dataframe columns
            # None is a placeholder, actual column value will be assigned in the following cycle
            df_dict["target"] = None
        for column in columns:
            df_cur = df.loc[:, pd.IndexSlice[:, column]]
            if column in category_columns:
                df_dict[column] = pd.api.types.union_categoricals([df_cur[col] for col in df_cur.columns])
            else:
                stacked = df_cur.values.T.ravel()
                # creating series is necessary for dtypes like "Int64", "boolean", otherwise they will be objects
                df_dict[column] = pd.Series(stacked, dtype=df_cur.dtypes[0])
        df_flat = pd.DataFrame(df_dict)

        return df_flat

    def to_pandas(self, flatten: bool = False, features: Union[Literal["all"], Sequence[str]] = "all") -> pd.DataFrame:
        """Return pandas DataFrame.

        Parameters
        ----------
        flatten:
            * If False, return dataframe in a wide format

            * If True, return dataframe in a long format,
              its order of columns is (timestamp, segment, target,
              features in alphabetical order).

        features:
            List of features to return.
            If "all", return all the features in the dataset.
        Returns
        -------
        pd.DataFrame
            dataframe with TSDataset data

        Examples
        --------
        >>> from etna.datasets import generate_const_df
        >>> df = generate_const_df(
        ...    periods=30, start_time="2021-06-01",
        ...    n_segments=2, scale=1
        ... )
        >>> df.head(5)
            timestamp    segment  target
        0  2021-06-01  segment_0    1.00
        1  2021-06-02  segment_0    1.00
        2  2021-06-03  segment_0    1.00
        3  2021-06-04  segment_0    1.00
        4  2021-06-05  segment_0    1.00
        >>> ts = TSDataset(df, "D")
        >>> ts.to_pandas(True).head(5)
            timestamp    segment  target
        0  2021-06-01  segment_0    1.00
        1  2021-06-02  segment_0    1.00
        2  2021-06-03  segment_0    1.00
        3  2021-06-04  segment_0    1.00
        4  2021-06-05  segment_0    1.00
        >>> ts.to_pandas(False).head(5)
        segment    segment_0 segment_1
        feature       target    target
        timestamp
        2021-06-01      1.00      1.00
        2021-06-02      1.00      1.00
        2021-06-03      1.00      1.00
        2021-06-04      1.00      1.00
        2021-06-05      1.00      1.00
        """
        if not flatten:
            if isinstance(features, str):
                if features == "all":
                    return self.df.copy()
                raise ValueError("The only possible literal is 'all'")
            segments = self.segments
            return self.df.loc[:, self.idx[segments, features]].copy()
        return self.to_flatten(self.df, features=features)

    @staticmethod
    def to_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """Convert pandas dataframe to wide format.

        Columns "timestamp" and "segment" are required.

        Parameters
        ----------
        df:
            DataFrame with columns ["timestamp", "segment"]. Other columns considered features.
            Columns "timestamp" is expected to be one of two types: integer or timestamp.

        Notes
        -----
        During conversion segment is casted to string type.

        Examples
        --------
        >>> from etna.datasets import generate_const_df
        >>> df = generate_const_df(
        ...    periods=30, start_time="2021-06-01",
        ...    n_segments=2, scale=1
        ... )
        >>> df.head(5)
           timestamp    segment  target
        0 2021-06-01  segment_0    1.00
        1 2021-06-02  segment_0    1.00
        2 2021-06-03  segment_0    1.00
        3 2021-06-04  segment_0    1.00
        4 2021-06-05  segment_0    1.00
        >>> df_wide = TSDataset.to_dataset(df)
        >>> df_wide.head(5)
        segment    segment_0 segment_1
        feature       target    target
        timestamp
        2021-06-01      1.00      1.00
        2021-06-02      1.00      1.00
        2021-06-03      1.00      1.00
        2021-06-04      1.00      1.00
        2021-06-05      1.00      1.00

        >>> df_regressors = pd.DataFrame({
        ...     "timestamp": pd.date_range("2021-01-01", periods=10),
        ...     "regressor_1": np.arange(10), "regressor_2": np.arange(10) + 5,
        ...     "segment": ["segment_0"]*10
        ... })
        >>> TSDataset.to_dataset(df_regressors).head(5)
        segment      segment_0
        feature    regressor_1 regressor_2
        timestamp
        2021-01-01           0           5
        2021-01-02           1           6
        2021-01-03           2           7
        2021-01-04           3           8
        2021-01-05           4           9
        """
        df_copy = df.copy(deep=True)

        if not pd.api.types.is_integer_dtype(df_copy["timestamp"]):
            df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

        df_copy["segment"] = df_copy["segment"].astype(str)

        feature_columns = df_copy.columns.tolist()
        feature_columns.remove("timestamp")
        feature_columns.remove("segment")

        df_copy = df_copy.pivot(index="timestamp", columns="segment")
        df_copy = df_copy.reorder_levels([1, 0], axis=1)
        df_copy.columns.names = ["segment", "feature"]
        df_copy = df_copy.sort_index(axis=1, level=(0, 1))
        return df_copy

    @staticmethod
    def _hierarchical_structure_from_level_columns(
        df: pd.DataFrame, level_columns: List[str], sep: str
    ) -> HierarchicalStructure:
        """Create hierarchical structure from dataframe columns."""
        df_level_columns = df[level_columns].astype("string")

        prev_level_name = level_columns[0]
        for cur_level_name in level_columns[1:]:
            df_level_columns[cur_level_name] = (
                df_level_columns[prev_level_name] + sep + df_level_columns[cur_level_name]
            )
            prev_level_name = cur_level_name

        level_structure = {"total": list(df_level_columns[level_columns[0]].unique())}
        cur_level_name = level_columns[0]
        for next_level_name in level_columns[1:]:
            cur_level_to_next_level_edges = df_level_columns[[cur_level_name, next_level_name]].drop_duplicates()
            cur_level_to_next_level_adjacency_list = cur_level_to_next_level_edges.groupby(cur_level_name).agg(list)

            # support for pandas>=1.4, <1.5
            level_structure.update(cur_level_to_next_level_adjacency_list.itertuples(name=None))
            cur_level_name = next_level_name

        hierarchical_structure = HierarchicalStructure(
            level_structure=level_structure, level_names=["total"] + level_columns
        )
        return hierarchical_structure

    @staticmethod
    def to_hierarchical_dataset(
        df: pd.DataFrame,
        level_columns: List[str],
        keep_level_columns: bool = False,
        sep: str = "_",
        return_hierarchy: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[HierarchicalStructure]]:
        """Convert pandas dataframe from long hierarchical to ETNA Dataset format.

        Parameters
        ----------
        df:
            Dataframe in long hierarchical format with columns [timestamp, target] + [level_columns] + [other_columns]
        level_columns:
            Columns of dataframe defines the levels in the hierarchy in order
            from top to bottom i.e [level_name_1, level_name_2, ...]. Names of the columns will be used as
            names of the levels in hierarchy.
        keep_level_columns:
            If true, leave the level columns in the result dataframe.
            By default level columns are concatenated into "segment" column and dropped
        sep:
            String to concatenated the level names with
        return_hierarchy:
            If true, returns the hierarchical structure

        Returns
        -------
        :
            Dataframe in wide format and optionally hierarchical structure

        Raises
        ------
        ValueError
            If ``level_columns`` is empty
        """
        if len(level_columns) == 0:
            raise ValueError("Value of level_columns shouldn't be empty!")

        df_copy = df.copy(deep=True)
        df_copy["segment"] = df_copy[level_columns].astype("string").add(sep).sum(axis=1).str[:-1]
        if not keep_level_columns:
            df_copy.drop(columns=level_columns, inplace=True)
        df_copy = TSDataset.to_dataset(df_copy)

        hierarchical_structure = None
        if return_hierarchy:
            hierarchical_structure = TSDataset._hierarchical_structure_from_level_columns(
                df=df, level_columns=level_columns, sep=sep
            )

        return df_copy, hierarchical_structure

    def _find_all_borders(
        self,
        train_start: Union[pd.Timestamp, int, str, None],
        train_end: Union[pd.Timestamp, int, str, None],
        test_start: Union[pd.Timestamp, int, str, None],
        test_end: Union[pd.Timestamp, int, str, None],
        test_size: Optional[int],
    ) -> Tuple[Union[pd.Timestamp, int], Union[pd.Timestamp, int], Union[pd.Timestamp, int], Union[pd.Timestamp, int]]:
        """Find borders for train_test_split if some values wasn't specified."""
        # prepare and validate values
        train_start = _check_timestamp_param(param=train_start, param_name="train_start", freq=self.freq)
        train_end = _check_timestamp_param(param=train_end, param_name="train_end", freq=self.freq)
        test_start = _check_timestamp_param(param=test_start, param_name="test_start", freq=self.freq)
        test_end = _check_timestamp_param(param=test_end, param_name="test_end", freq=self.freq)

        if test_end is not None and test_start is not None and test_size is not None:
            warnings.warn(
                "test_size, test_start and test_end cannot be applied at the same time. test_size will be ignored"
            )

        if test_end is None:
            if test_start is not None and test_size is not None:
                test_start_idx = self.df.index.get_loc(test_start)
                if test_start_idx + test_size > len(self.df.index):
                    raise ValueError(
                        f"test_size is {test_size}, but only {len(self.df.index) - test_start_idx} available with your test_start"
                    )
                test_end_defined = self.df.index[test_start_idx + test_size]
            elif test_size is not None and train_end is not None:
                test_start_idx = self.df.index.get_loc(train_end)
                test_start = self.df.index[test_start_idx + 1]
                test_end_defined = self.df.index[test_start_idx + test_size]
            else:
                test_end_defined = self.df.index.max()
        else:
            test_end_defined = test_end

        if train_start is None:
            train_start_defined = self.df.index.min()
        else:
            train_start_defined = train_start

        if train_end is None and test_start is None and test_size is None:
            raise ValueError("At least one of train_end, test_start or test_size should be defined")

        if test_size is None:
            if train_end is None:
                test_start_idx = self.df.index.get_loc(test_start)
                train_end_defined = self.df.index[test_start_idx - 1]
            else:
                train_end_defined = train_end

            if test_start is None:
                train_end_idx = self.df.index.get_loc(train_end)
                test_start_defined = self.df.index[train_end_idx + 1]
            else:
                test_start_defined = test_start
        else:
            if test_start is None:
                test_start_idx = self.df.index.get_loc(test_end_defined)
                test_start_defined = self.df.index[test_start_idx - test_size + 1]
            else:
                test_start_defined = test_start

            if train_end is None:
                test_start_idx = self.df.index.get_loc(test_start_defined)
                train_end_defined = self.df.index[test_start_idx - 1]
            else:
                train_end_defined = train_end

        if test_start_defined < train_end_defined:
            raise ValueError("The beginning of the test goes before the end of the train")

        return train_start_defined, train_end_defined, test_start_defined, test_end_defined

    def train_test_split(
        self,
        train_start: Union[pd.Timestamp, int, str, None] = None,
        train_end: Union[pd.Timestamp, int, str, None] = None,
        test_start: Union[pd.Timestamp, int, str, None] = None,
        test_end: Union[pd.Timestamp, int, str, None] = None,
        test_size: Optional[int] = None,
    ) -> Tuple["TSDataset", "TSDataset"]:
        """Split given df with train-test timestamp indices or size of test set.

        In case of inconsistencies between ``test_size`` and (``test_start``, ``test_end``), ``test_size`` is ignored

        Parameters
        ----------
        train_start:
            start timestamp of new train dataset, if None first timestamp is used
        train_end:
            end timestamp of new train dataset, if None previous to ``test_start`` timestamp is used
        test_start:
            start timestamp of new test dataset, if None next to ``train_end`` timestamp is used
        test_end:
            end timestamp of new test dataset, if None last timestamp is used
        test_size:
            number of timestamps to use in test set

        Returns
        -------
        train, test:
            generated datasets

        Raises
        ------
        ValueError:
            Incorrect type of ``train_start`` or ``train_end`` or ``test_start`` or ``test_end``
            is used according to ``ts.freq``

        Examples
        --------
        >>> from etna.datasets import generate_ar_df
        >>> pd.options.display.float_format = '{:,.2f}'.format
        >>> df = generate_ar_df(100, start_time="2021-01-01", n_segments=3)
        >>> ts = TSDataset(df, "D")
        >>> train_ts, test_ts = ts.train_test_split(
        ...     train_start="2021-01-01", train_end="2021-02-01",
        ...     test_start="2021-02-02", test_end="2021-02-07"
        ... )
        >>> train_ts.df.tail(5)
        segment    segment_0 segment_1 segment_2
        feature       target    target    target
        timestamp
        2021-01-28     -2.06      2.03      1.51
        2021-01-29     -2.33      0.83      0.81
        2021-01-30     -1.80      1.69      0.61
        2021-01-31     -2.49      1.51      0.85
        2021-02-01     -2.89      0.91      1.06
        >>> test_ts.df.head(5)
        segment    segment_0 segment_1 segment_2
        feature       target    target    target
        timestamp
        2021-02-02     -3.57     -0.32      1.72
        2021-02-03     -4.42      0.23      3.51
        2021-02-04     -5.09      1.02      3.39
        2021-02-05     -5.10      0.40      2.15
        2021-02-06     -6.22      0.92      0.97
        """
        train_start_defined, train_end_defined, test_start_defined, test_end_defined = self._find_all_borders(
            train_start, train_end, test_start, test_end, test_size
        )

        if test_end_defined > self.df.index.max():
            warnings.warn(f"Max timestamp in df is {self.df.index.max()}.")
        if train_start_defined < self.df.index.min():
            warnings.warn(f"Min timestamp in df is {self.df.index.min()}.")

        train_df = self.df.loc[train_start_defined:train_end_defined][self.raw_df.columns]  # type: ignore
        train_raw_df = self.raw_df.loc[train_start_defined:train_end_defined]  # type: ignore
        train = TSDataset(
            df=train_df,
            df_exog=self.df_exog,
            freq=self.freq,
            known_future=self.known_future,
            hierarchical_structure=self.hierarchical_structure,
        )
        train.raw_df = train_raw_df
        train._regressors = deepcopy(self.regressors)
        train._target_components_names = deepcopy(self.target_components_names)
        train._prediction_intervals_names = deepcopy(self._prediction_intervals_names)

        test_df = self.df.loc[test_start_defined:test_end_defined][self.raw_df.columns]  # type: ignore
        test_raw_df = self.raw_df.loc[train_start_defined:test_end_defined]  # type: ignore
        test = TSDataset(
            df=test_df,
            df_exog=self.df_exog,
            freq=self.freq,
            known_future=self.known_future,
            hierarchical_structure=self.hierarchical_structure,
        )
        test.raw_df = test_raw_df
        test._regressors = deepcopy(self.regressors)
        test._target_components_names = deepcopy(self.target_components_names)
        test._prediction_intervals_names = deepcopy(self._prediction_intervals_names)
        return train, test

    def update_columns_from_pandas(self, df_update: pd.DataFrame):
        """Update the existing columns in the dataset with the new values from pandas dataframe.

        Before updating columns in ``df``, columns of ``df_update`` will be cropped by the last timestamp in ``df``.
        Columns in ``df_exog`` are not updated. If you wish to update the ``df_exog``, create the new
        instance of TSDataset.

        Parameters
        ----------
        df_update:
            Dataframe with new values in wide ETNA format.
        """
        df = df_update.loc[self.df.index.min() : self.df.index.max()]

        if not df.index.equals(self.df.index):
            raise ValueError("Non matching timestamps detected when attempted to update the dataset!")

        if len(df.columns.difference(self.df.columns)) > 0:
            raise ValueError("Some columns in the dataframe for update are not presented in the dataset!")

        column_idx = self.df.columns.get_indexer(df.columns)
        self.df.iloc[:, column_idx] = df

    def add_columns_from_pandas(
        self, df_update: pd.DataFrame, update_exog: bool = False, regressors: Optional[List[str]] = None
    ):
        """Update the dataset with the new columns from pandas dataframe.

        Before updating columns in df, columns of df_update will be cropped by the last timestamp in df.

        Parameters
        ----------
        df_update:
            Dataframe with the new columns in wide ETNA format.
        update_exog:
             If True, update columns also in df_exog.
             If you wish to add new regressors in the dataset it is recommended to turn on this flag.
        regressors:
            List of regressors in the passed dataframe.
        """
        self.df = pd.concat((self.df, df_update.loc[: self.df.index.max()]), axis=1).sort_index(axis=1)
        if update_exog:
            if self.df_exog is None:
                self.df_exog = df_update
            else:
                self.df_exog = pd.concat((self.df_exog, df_update), axis=1).sort_index(axis=1)
        if regressors is not None:
            self._regressors = list(set(self._regressors) | set(regressors))

    def drop_features(self, features: List[str], drop_from_exog: bool = False):
        """Drop columns with features from the dataset.

        Parameters
        ----------
        features:
            List of features to drop.
        drop_from_exog:
            * If False, drop features only from df. Features will appear again in df after make_future.
            * If True, drop features from df and df_exog. Features won't appear in df after make_future.

        Raises
        ------
        ValueError:
            If ``features`` list contains target or target components
        """
        features_set = set(features)

        features_contain_target_components = len(features_set.intersection(self.target_components_names)) > 0
        if features_contain_target_components:
            raise ValueError(
                "Target components can't be dropped from the dataset using this method! Use `drop_target_components` method!"
            )

        features_contain_prediction_intervals = len(features_set.intersection(self.prediction_intervals_names)) > 0
        if features_contain_prediction_intervals:
            raise ValueError(
                "Prediction intervals can't be dropped from the dataset using this method! Use `drop_prediction_intervals` method!"
            )

        if "target" in features_set:
            raise ValueError(f"Target can't be dropped from the dataset!")

        dfs = [("df", self.df)]
        if drop_from_exog:
            dfs.append(("df_exog", self.df_exog))

        for name, df in dfs:
            columns_in_df = df.columns.get_level_values("feature")
            columns_to_remove = list(set(columns_in_df) & features_set)
            unknown_columns = features_set - set(columns_to_remove)
            if len(unknown_columns) > 0:
                warnings.warn(f"Features {unknown_columns} are not present in {name}!")
            if len(columns_to_remove) > 0:
                df.drop(columns=columns_to_remove, level="feature", inplace=True)
        self._regressors = list(set(self._regressors) - features_set)

    @property
    def index(self) -> pd.Index:
        """Return TSDataset timestamp index.

        Returns
        -------
        :
            timestamp index of TSDataset
        """
        return self.df.index

    def level_names(self) -> Optional[List[str]]:
        """Return names of the levels in the hierarchical structure."""
        if self.hierarchical_structure is None:
            return None
        return self.hierarchical_structure.level_names

    def has_hierarchy(self) -> bool:
        """Check whether dataset has hierarchical structure."""
        return self.hierarchical_structure is not None

    def get_level_dataset(self, target_level: str) -> "TSDataset":
        """Generate new TSDataset on target level.

        Parameters
        ----------
        target_level:
            target level name

        Returns
        -------
        TSDataset
            generated dataset
        """
        if self.hierarchical_structure is None or self.current_df_level is None:
            raise ValueError("Method could be applied only to instances with a hierarchy!")

        current_level_segments = self.hierarchical_structure.get_level_segments(level_name=self.current_df_level)
        target_level_segments = self.hierarchical_structure.get_level_segments(level_name=target_level)

        current_level_index = self.hierarchical_structure.get_level_depth(self.current_df_level)
        target_level_index = self.hierarchical_structure.get_level_depth(target_level)

        if target_level_index > current_level_index:
            raise ValueError("Target level should be higher in the hierarchy than the current level of dataframe!")

        target_names = self.prediction_intervals_names + self.target_components_names + ("target",)

        if target_level_index < current_level_index:
            summing_matrix = self.hierarchical_structure.get_summing_matrix(
                target_level=target_level, source_level=self.current_df_level
            )

            target_level_df = get_level_dataframe(
                df=self.to_pandas(features=target_names),
                mapping_matrix=summing_matrix,
                source_level_segments=current_level_segments,
                target_level_segments=target_level_segments,
            )

        else:
            target_level_df = self.to_pandas(features=target_names)

        target_components_df = target_level_df.loc[:, pd.IndexSlice[:, self.target_components_names]]
        if len(self.target_components_names) > 0:  # for pandas >=1.1, <1.2
            target_level_df = target_level_df.drop(columns=list(self.target_components_names), level="feature")

        prediction_intervals_df = target_level_df.loc[:, pd.IndexSlice[:, self.prediction_intervals_names]]
        if len(self.prediction_intervals_names) > 0:  # for pandas >=1.1, <1.2
            target_level_df = target_level_df.drop(columns=list(self.prediction_intervals_names), level="feature")

        ts = TSDataset(
            df=target_level_df,
            freq=self.freq,
            df_exog=self.df_exog,
            known_future=self.known_future,
            hierarchical_structure=self.hierarchical_structure,
        )

        if len(self.target_components_names) > 0:
            ts.add_target_components(target_components_df=target_components_df)

        if len(self.prediction_intervals_names) > 0:
            ts.add_prediction_intervals(prediction_intervals_df=prediction_intervals_df)

        return ts

    def add_target_components(self, target_components_df: pd.DataFrame):
        """Add target components into dataset.

        Parameters
        ----------
        target_components_df:
            Dataframe in a wide format with target components

        Raises
        ------
        ValueError:
            If dataset already contains target components
        ValueError:
            If target components names differ between segments
        ValueError:
            If components don't sum up to target
        """
        if len(self.target_components_names) > 0:
            raise ValueError("Dataset already contains target components!")

        components_names = sorted(target_components_df[self.segments[0]].columns.get_level_values("feature"))
        for segment in self.segments:
            components_names_segment = sorted(target_components_df[segment].columns.get_level_values("feature"))
            if components_names != components_names_segment:
                raise ValueError(
                    f"Set of target components differs between segments '{self.segments[0]}' and '{segment}'!"
                )

        components_sum = target_components_df.groupby(axis=1, level="segment").sum()
        if not np.allclose(components_sum.values, self[..., "target"].values):
            raise ValueError("Components don't sum up to target!")

        self._target_components_names = tuple(components_names)
        self.df = (
            pd.concat((self.df, target_components_df), axis=1)
            .loc[self.df.index]
            .sort_index(axis=1, level=("segment", "feature"))
        )

    def get_target_components(self) -> Optional[pd.DataFrame]:
        """Get DataFrame with target components.

        Returns
        -------
        :
            Dataframe with target components
        """
        if len(self.target_components_names) == 0:
            return None
        return self.to_pandas(features=self.target_components_names)

    def drop_target_components(self):
        """Drop target components from dataset."""
        if len(self.target_components_names) > 0:  # for pandas >=1.1, <1.2
            self.df.drop(columns=list(self.target_components_names), level="feature", inplace=True)
            self._target_components_names = ()

    def add_prediction_intervals(self, prediction_intervals_df: pd.DataFrame):
        """Add target components into dataset.

        Parameters
        ----------
        prediction_intervals_df:
            Dataframe in a wide format with prediction intervals

        Raises
        ------
        ValueError:
            If dataset already contains prediction intervals
        ValueError:
            If prediction intervals names differ between segments
        """
        if len(self.prediction_intervals_names) > 0:
            raise ValueError("Dataset already contains prediction intervals!")

        intervals_names = sorted(prediction_intervals_df[self.segments[0]].columns.get_level_values("feature"))
        for segment in self.segments:
            segment_intervals_names = sorted(prediction_intervals_df[segment].columns.get_level_values("feature"))

            if intervals_names != segment_intervals_names:
                raise ValueError(
                    f"Set of prediction intervals differs between segments '{self.segments[0]}' and '{segment}'!"
                )

        self._prediction_intervals_names = tuple(intervals_names)
        self.df = (
            pd.concat((self.df, prediction_intervals_df), axis=1)
            .loc[self.df.index]
            .sort_index(axis=1, level=("segment", "feature"))
        )

    def get_prediction_intervals(self) -> Optional[pd.DataFrame]:
        """Get ``pandas.DataFrame`` with prediction intervals.

        Returns
        -------
        :
            ``pandas.DataFrame`` with prediction intervals for target variable.
        """
        if len(self.prediction_intervals_names) == 0:
            return None

        return self.to_pandas(features=self.prediction_intervals_names)

    def drop_prediction_intervals(self):
        """Drop prediction intervals from dataset."""
        if len(self.prediction_intervals_names) > 0:  # for pandas >=1.1, <1.2
            self.df.drop(columns=list(self.prediction_intervals_names), level="feature", inplace=True)
            self._prediction_intervals_names = tuple()

    @property
    def columns(self) -> pd.core.indexes.multi.MultiIndex:
        """Return columns of ``self.df``.

        Returns
        -------
        pd.core.indexes.multi.MultiIndex
            multiindex of dataframe with target and features.
        """
        return self.df.columns

    @property
    def loc(self) -> pd.core.indexing._LocIndexer:
        """Return self.df.loc method.

        Returns
        -------
        pd.core.indexing._LocIndexer
            dataframe with self.df.loc[...]
        """
        return self.df.loc

    def isnull(self) -> pd.DataFrame:
        """Return dataframe with flag that means if the correspondent object in ``self.df`` is null.

        Returns
        -------
        pd.Dataframe
            is_null dataframe
        """
        return self.df.isnull()

    def head(self, n_rows: int = 5) -> pd.DataFrame:
        """Return the first ``n_rows`` rows.

        Mimics pandas method.

        This function returns the first ``n_rows`` rows for the object based
        on position. It is useful for quickly testing if your object
        has the right type of data in it.

        For negative values of ``n_rows``, this function returns all rows except
        the last ``n_rows`` rows, equivalent to ``df[:-n_rows]``.

        Parameters
        ----------
        n_rows:
            number of rows to select.

        Returns
        -------
        pd.DataFrame
            the first ``n_rows`` rows or 5 by default.
        """
        return self.df.head(n_rows)

    def tail(self, n_rows: int = 5) -> pd.DataFrame:
        """Return the last ``n_rows`` rows.

        Mimics pandas method.

        This function returns last ``n_rows`` rows from the object based on
        position. It is useful for quickly verifying data, for example,
        after sorting or appending rows.

        For negative values of ``n_rows``, this function returns all rows except
        the first `n` rows, equivalent to ``df[n_rows:]``.

        Parameters
        ----------
        n_rows:
            number of rows to select.

        Returns
        -------
        pd.DataFrame
            the last ``n_rows`` rows or 5 by default.

        """
        return self.df.tail(n_rows)

    def _gather_common_data(self) -> Dict[str, Any]:
        """Gather information about dataset in general."""
        features = set(self.features)
        exogs = (
            features.difference({"target"})
            .difference(self.prediction_intervals_names)
            .difference(self.target_components_names)
        )
        common_dict: Dict[str, Any] = {
            "num_segments": len(self.segments),
            "num_exogs": len(exogs),
            "num_regressors": len(self.regressors),
            "num_known_future": len(self.known_future),
            "freq": self.freq,
        }

        return common_dict

    def _gather_segments_data(self, segments: Optional[Sequence[str]]) -> Dict[str, pd.Series]:
        """Gather information about each segment."""
        segments_index: Union[slice, Sequence[str]]
        if segments is None:
            segments_index = slice(None)
            segments = self.segments
        else:
            segments_index = segments
            segments = segments

        df = self.df.loc[:, (segments_index, "target")]

        num_timestamps = df.shape[0]
        not_na = ~np.isnan(df.values)
        min_idx = np.argmax(not_na, axis=0)
        max_idx = num_timestamps - np.argmax(not_na[::-1, :], axis=0) - 1

        segments_dict = {}
        segments_dict["start_timestamp"] = df.index[min_idx].to_series(index=segments)
        segments_dict["end_timestamp"] = df.index[max_idx].to_series(index=segments)
        segments_dict["length"] = pd.Series(max_idx - min_idx + 1, dtype="Int64", index=segments)
        segments_dict["num_missing"] = pd.Series(
            segments_dict["length"] - np.sum(not_na, axis=0), dtype="Int64", index=segments
        )

        # handle all-nans series
        all_nans_mask = np.all(~not_na, axis=0)
        segments_dict["start_timestamp"][all_nans_mask] = None
        segments_dict["end_timestamp"][all_nans_mask] = None
        segments_dict["length"][all_nans_mask] = None
        segments_dict["num_missing"][all_nans_mask] = None

        return segments_dict

    def describe(self, segments: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Overview of the dataset that returns a DataFrame.

        Method describes dataset in segment-wise fashion. Description columns:

        * start_timestamp: beginning of the segment, missing values in the beginning are ignored

        * end_timestamp: ending of the segment, missing values in the ending are ignored

        * length: length according to ``start_timestamp`` and ``end_timestamp``

        * num_missing: number of missing variables between ``start_timestamp`` and ``end_timestamp``

        * num_segments: total number of segments, common for all segments

        * num_exogs: number of exogenous features, common for all segments

        * num_regressors: number of exogenous factors, that are regressors, common for all segments

        * num_known_future: number of regressors, that are known since creation, common for all segments

        * freq: frequency of the series, common for all segments

        Parameters
        ----------
        segments:
            segments to show in overview, if None all segments are shown.

        Returns
        -------
        result_table: pd.DataFrame
            table with results of the overview

        Examples
        --------
        >>> from etna.datasets import generate_const_df
        >>> pd.options.display.expand_frame_repr = False
        >>> df = generate_const_df(
        ...    periods=30, start_time="2021-06-01",
        ...    n_segments=2, scale=1
        ... )
        >>> regressors_timestamp = pd.date_range(start="2021-06-01", periods=50)
        >>> df_regressors_1 = pd.DataFrame(
        ...     {"timestamp": regressors_timestamp, "regressor_1": 1, "segment": "segment_0"}
        ... )
        >>> df_regressors_2 = pd.DataFrame(
        ...     {"timestamp": regressors_timestamp, "regressor_1": 2, "segment": "segment_1"}
        ... )
        >>> df_exog = pd.concat([df_regressors_1, df_regressors_2], ignore_index=True)
        >>> ts = TSDataset(df, df_exog=df_exog, freq="D", known_future="all")
        >>> ts.describe()
                  start_timestamp end_timestamp  length  num_missing  num_segments  num_exogs  num_regressors  num_known_future freq
        segments
        segment_0      2021-06-01    2021-06-30      30            0             2          1               1                 1    D
        segment_1      2021-06-01    2021-06-30      30            0             2          1               1                 1    D
        """
        # gather common information
        common_dict = self._gather_common_data()

        # gather segment information
        segments_dict = self._gather_segments_data(segments)

        if segments is None:
            segments = self.segments

        # combine information
        segments_dict["num_segments"] = [common_dict["num_segments"]] * len(segments)
        segments_dict["num_exogs"] = [common_dict["num_exogs"]] * len(segments)
        segments_dict["num_regressors"] = [common_dict["num_regressors"]] * len(segments)
        segments_dict["num_known_future"] = [common_dict["num_known_future"]] * len(segments)
        segments_dict["freq"] = [common_dict["freq"]] * len(segments)

        result_df = pd.DataFrame(segments_dict, index=segments)
        columns_order = [
            "start_timestamp",
            "end_timestamp",
            "length",
            "num_missing",
            "num_segments",
            "num_exogs",
            "num_regressors",
            "num_known_future",
            "freq",
        ]
        result_df = result_df[columns_order]
        result_df.index.name = "segments"
        return result_df

    def info(self, segments: Optional[Sequence[str]] = None) -> None:
        """Overview of the dataset that prints the result.

        Method describes dataset in segment-wise fashion.

        Information about dataset in general:

        * num_segments: total number of segments

        * num_exogs: number of exogenous features

        * num_regressors: number of exogenous factors, that are regressors

        * num_known_future: number of regressors, that are known since creation

        * freq: frequency of the dataset

        Information about individual segments:

        * start_timestamp: beginning of the segment, missing values in the beginning are ignored

        * end_timestamp: ending of the segment, missing values in the ending are ignored

        * length: length according to ``start_timestamp`` and ``end_timestamp``

        * num_missing: number of missing variables between ``start_timestamp`` and ``end_timestamp``

        Parameters
        ----------
        segments:
            segments to show in overview, if None all segments are shown.

        Examples
        --------
        >>> from etna.datasets import generate_const_df
        >>> df = generate_const_df(
        ...    periods=30, start_time="2021-06-01",
        ...    n_segments=2, scale=1
        ... )
        >>> regressors_timestamp = pd.date_range(start="2021-06-01", periods=50)
        >>> df_regressors_1 = pd.DataFrame(
        ...     {"timestamp": regressors_timestamp, "regressor_1": 1, "segment": "segment_0"}
        ... )
        >>> df_regressors_2 = pd.DataFrame(
        ...     {"timestamp": regressors_timestamp, "regressor_1": 2, "segment": "segment_1"}
        ... )
        >>> df_exog = pd.concat([df_regressors_1, df_regressors_2], ignore_index=True)
        >>> ts = TSDataset(df, df_exog=df_exog, freq="D", known_future="all")
        >>> ts.info()
        <class 'etna.datasets.TSDataset'>
        num_segments: 2
        num_exogs: 1
        num_regressors: 1
        num_known_future: 1
        freq: D
                  start_timestamp end_timestamp  length  num_missing
        segments
        segment_0      2021-06-01    2021-06-30      30            0
        segment_1      2021-06-01    2021-06-30      30            0
        """
        if segments is None:
            segments = self.segments
        lines = []

        # add header
        lines.append("<class 'etna.datasets.TSDataset'>")

        # add common information
        common_dict = self._gather_common_data()

        for key, value in common_dict.items():
            lines.append(f"{key}: {value}")

        # add segment information
        segments_dict = self._gather_segments_data(segments)
        segment_df = pd.DataFrame(segments_dict, index=segments)
        segment_df.index.name = "segments"

        with pd.option_context("display.width", None):
            lines += segment_df.to_string().split("\n")

        # print the results
        result_string = "\n".join(lines)
        print(result_string)

    def to_torch_dataset(
        self, make_samples: Callable[[pd.DataFrame], Union[Iterator[dict], Iterable[dict]]], dropna: bool = True
    ) -> "Dataset":
        """Convert the TSDataset to a :py:class:`torch.Dataset`.

        Parameters
        ----------
        make_samples:
            function that takes per segment DataFrame and returns iterabale of samples
        dropna:
            if ``True``, missing rows are dropped

        Returns
        -------
        :
            :py:class:`torch.Dataset` with with train or test samples to infer on
        """
        df = self.to_pandas(flatten=True)
        if dropna:
            df = df.dropna()  # TODO: Fix this

        ts_segments = [df_segment for _, df_segment in df.groupby("segment")]
        ts_samples = [samples for df_segment in ts_segments for samples in make_samples(df_segment)]

        return _TorchDataset(ts_samples=ts_samples)

    def size(self) -> Tuple[int, int, Optional[int]]:
        """Return size of TSDataset.

        The order of sizes is (number of time series, number of segments,
        and number of features (if their amounts are equal in each segment; otherwise, returns None)).

        Returns
        -------
        :
            Tuple of TSDataset sizes
        """
        current_number_of_features = 0
        for segment in self.segments:
            cur_seg_features = self.df[segment].columns.get_level_values("feature").unique()
            if current_number_of_features != 0 and current_number_of_features != len(cur_seg_features):
                return len(self.index), len(self.segments), None
            current_number_of_features = len(cur_seg_features)
        return len(self.index), len(self.segments), current_number_of_features
