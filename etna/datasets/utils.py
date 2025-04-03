from collections import Counter
from collections import defaultdict
from enum import Enum
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
from typing import cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing_extensions import assert_never

from etna import SETTINGS

if SETTINGS.torch_required:
    from torch.utils.data import Dataset
else:
    from unittest.mock import Mock

    Dataset = Mock  # type: ignore


class DataFrameFormat(str, Enum):
    """Enum for different kinds of ``pd.DataFrame`` which can be used.

    This dataframe stores:

    - Timestamps;
    - Segments;
    - Features. In this context, 'target' is also a feature.

    Currently, there are formats:

    - Wide

      - Has index to store timestamps.
      - Columns has two levels with names 'segment', 'feature'.
        Each column stores values for a given feature in a given segment.
      - List of columns isn't empty.
      - There are all combinations for (segment, feature) in the columns.

    - Long

      - Has column 'timestamp' to store timestamps.
      - Has column 'segment' to store segments.
      - Has at least one more column except for 'timestamp' and 'segment'.

    Currently, we don't check the types of columns to save compatibility, but it is expected that:

    - Timestamps have type ``int`` or ``pd.Timestamp``. If it isn't, :py:class:`~etna.datasets.tsdataset.TSDataset`
      makes conversion for you.
    - Segments have type ``str``. If it isn't, :py:class:`~etna.datasets.tsdataset.TSDataset` makes conversion for you.
    """

    #: Wide format.
    wide = "wide"

    #: Long format.
    long = "long"

    @classmethod
    def determine(cls, df: pd.DataFrame) -> "DataFrameFormat":
        """Determine format of the given dataframe.

        Parameters
        ----------
        df:
            Dataframe to infer format.

        Returns
        -------
        :
            Format of the given dataframe.

        Raises
        ------
        ValueError:
            Given long dataframe doesn't have required column 'timestamp'
        ValueError:
            Given long dataframe doesn't have required column 'segment'
        ValueError:
            Given long dataframe doesn't have any columns except for 'timestamp` and 'segment'
        ValueError:
            Given wide dataframe doesn't have levels of columns ['segment', 'feature']
        ValueError:
            Given wide dataframe doesn't have any features
        ValueError:
            Given wide dataframe doesn't have all combinations of pairs (segment, feature)
        """
        columns = df.columns
        is_multiindex = isinstance(columns, pd.MultiIndex)

        if not is_multiindex:
            if "timestamp" not in columns:
                raise ValueError("Given long dataframe doesn't have required column 'timestamp'!")
            if "segment" not in columns:
                raise ValueError("Given long dataframe doesn't have required column 'segment'!")
            if set(columns) == {"timestamp", "segment"}:
                raise ValueError("Given long dataframe doesn't have any columns except for 'timestamp` and 'segment'!")
            return DataFrameFormat.long
        else:
            expected_level_names = ["segment", "feature"]
            if columns.names != expected_level_names:
                raise ValueError("Given wide dataframe doesn't have levels of columns ['segment', 'feature']!")

            if len(columns) == 0:
                raise ValueError("Given wide dataframe doesn't have any features!")

            segments = columns.get_level_values("segment").unique()
            features = columns.get_level_values("feature").unique()
            expected_columns = pd.MultiIndex.from_product(
                [segments, features], names=["segment", "feature"]
            ).sort_values()
            if not columns.sort_values().equals(expected_columns):
                raise ValueError("Given wide dataframe doesn't have all combinations of pairs (segment, feature)!")

            return DataFrameFormat.wide


def duplicate_data(df: pd.DataFrame, segments: Sequence[str], format: str = DataFrameFormat.wide) -> pd.DataFrame:
    """Duplicate dataframe for all the segments.

    Parameters
    ----------
    df:
        dataframe to duplicate, there should be column "timestamp"
    segments:
        list of segments for making duplication
    format:
        represent the result in TSDataset inner format (wide) or in flatten format (long)

    Returns
    -------
    result:
        result of duplication for all the segments

    Raises
    ------
    ValueError:
        if segments list is empty
    ValueError:
        if incorrect format is given
    ValueError:
        if dataframe doesn't contain "timestamp" column

    Examples
    --------
    >>> from etna.datasets import generate_const_df
    >>> from etna.datasets import duplicate_data
    >>> from etna.datasets import TSDataset
    >>> df = generate_const_df(
    ...    periods=50, start_time="2020-03-10",
    ...    n_segments=2, scale=1
    ... )
    >>> timestamp = pd.date_range("2020-03-10", periods=100, freq="D")
    >>> is_friday_13 = (timestamp.weekday == 4) & (timestamp.day == 13)
    >>> df_exog_raw = pd.DataFrame({"timestamp": timestamp, "is_friday_13": is_friday_13})
    >>> df_exog = duplicate_data(df_exog_raw, segments=["segment_0", "segment_1"], format="wide")
    >>> ts = TSDataset(df=df, df_exog=df_exog, freq="D", known_future="all")
    >>> ts.head()
    segment       segment_0           segment_1
    feature    is_friday_13 target is_friday_13 target
    timestamp
    2020-03-10        False   1.00        False   1.00
    2020-03-11        False   1.00        False   1.00
    2020-03-12        False   1.00        False   1.00
    2020-03-13         True   1.00         True   1.00
    2020-03-14        False   1.00        False   1.00
    """
    from etna.datasets.tsdataset import TSDataset

    # check segments length
    if len(segments) == 0:
        raise ValueError("Parameter segments shouldn't be empty")

    # check format
    format_enum = DataFrameFormat(format)

    # check the columns
    if "timestamp" not in df.columns:
        raise ValueError("There should be 'timestamp' column")

    # construct long version
    n_segments, n_timestamps = len(segments), df.shape[0]
    df_long = df.iloc[np.tile(np.arange(n_timestamps), n_segments)]
    df_long["segment"] = np.repeat(a=segments, repeats=n_timestamps)

    # construct wide version if necessary
    if format_enum == DataFrameFormat.wide:
        df_wide = TSDataset.to_dataset(df_long)
        return df_wide

    return df_long


class _TorchDataset(Dataset):
    """In memory dataset for torch dataloader."""

    def __init__(self, ts_samples: List[dict]):
        """Init torch dataset.

        Parameters
        ----------
        ts_samples:
            time series samples for training or inference
        """
        self.ts_samples = ts_samples

    def __getitem__(self, index):
        return self.ts_samples[index]

    def __len__(self):
        return len(self.ts_samples)


def set_columns_wide(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    timestamps_left: Optional[Sequence[Union[pd.Timestamp, int]]] = None,
    timestamps_right: Optional[Sequence[Union[pd.Timestamp, int]]] = None,
    segments_left: Optional[Sequence[str]] = None,
    features_right: Optional[Sequence[str]] = None,
    features_left: Optional[Sequence[str]] = None,
    segments_right: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Set columns in a left dataframe with values from the right dataframe.

    Parameters
    ----------
    df_left:
        dataframe to set columns in
    df_right:
        dataframe to set columns from
    timestamps_left:
        timestamps to select in ``df_left``
    timestamps_right:
        timestamps to select in ``df_right``
    segments_left:
        segments to select in ``df_left``
    segments_right:
        segments to select in ``df_right``
    features_left:
        features to select in ``df_left``
    features_right:
        features to select in ``df_right``

    Returns
    -------
    :
        a new dataframe with changed columns
    """
    # sort columns
    df_left = df_left.sort_index(axis=1)
    df_right = df_right.sort_index(axis=1)

    # prepare indexing
    timestamps_left_index = slice(None) if timestamps_left is None else timestamps_left
    timestamps_right_index = slice(None) if timestamps_right is None else timestamps_right
    segments_left_index = slice(None) if segments_left is None else segments_left
    segments_right_index = slice(None) if segments_right is None else segments_right
    features_left_index = slice(None) if features_left is None else features_left
    features_right_index = slice(None) if features_right is None else features_right

    right_value = df_right.loc[timestamps_right_index, (segments_right_index, features_right_index)]
    df_left.loc[timestamps_left_index, (segments_left_index, features_left_index)] = right_value.values

    return df_left


def get_level_dataframe(
    df: pd.DataFrame,
    mapping_matrix: csr_matrix,
    source_level_segments: List[str],
    target_level_segments: List[str],
):
    """Perform mapping to dataframe at the target level.

    Parameters
    ----------
    df:
        dataframe at the source level
    mapping_matrix:
        mapping matrix between levels
    source_level_segments:
        list of segments at the source level, set the order of segments matching the mapping matrix
    target_level_segments:
        list of segments at the target level

    Returns
    -------
    :
       dataframe at the target level
    """
    column_names = sorted(set(df.columns.get_level_values("feature")))
    num_columns = len(column_names)
    num_source_level_segments = len(source_level_segments)
    num_target_level_segments = len(target_level_segments)

    if set(df.columns.get_level_values(level="segment")) != set(source_level_segments):
        raise ValueError("Segments mismatch for provided dataframe and `source_level_segments`!")

    if num_source_level_segments != mapping_matrix.shape[1]:
        raise ValueError("Number of source level segments do not match mapping matrix number of columns!")

    if num_target_level_segments != mapping_matrix.shape[0]:
        raise ValueError("Number of target level segments do not match mapping matrix number of columns!")

    # Slice should be done by source_level_segments -- to fix the order of segments for mapping matrix,
    # by num_columns -- to fix the order of columns to create correct index in the end
    source_level_data = df.loc[
        pd.IndexSlice[:], pd.IndexSlice[source_level_segments, column_names]
    ].values  # shape: (t, num_source_level_segments * num_columns)

    source_level_data = source_level_data.reshape(
        (-1, num_source_level_segments, num_columns)
    )  # shape: (t, num_source_level_segments, num_columns)
    source_level_data = np.swapaxes(source_level_data, 1, 2)  # shape: (t, num_columns, num_source_level_segments)
    source_level_data = source_level_data.reshape(
        (-1, num_source_level_segments)
    )  # shape: (t * num_columns, num_source_level_segments)

    target_level_data = source_level_data @ mapping_matrix.T

    target_level_data = target_level_data.reshape(
        (-1, num_columns, num_target_level_segments)
    )  # shape: (t, num_columns, num_target_level_segments)
    target_level_data = np.swapaxes(target_level_data, 1, 2)  # shape: (t, num_target_level_segments, num_columns)
    target_level_data = target_level_data.reshape(
        (-1, num_columns * num_target_level_segments)
    )  # shape: (t, num_target_level_segments * num_columns)

    target_level_segments = pd.MultiIndex.from_product(
        [target_level_segments, column_names], names=["segment", "feature"]
    )
    target_level_df = pd.DataFrame(data=target_level_data, index=df.index, columns=target_level_segments)

    return target_level_df


def inverse_transform_target_components(
    target_components_df: pd.DataFrame, target_df: pd.DataFrame, inverse_transformed_target_df: pd.DataFrame
) -> pd.DataFrame:
    """Inverse transform target components.

    Parameters
    ----------
    target_components_df:
        Dataframe with target components
    target_df:
        Dataframe with transformed target
    inverse_transformed_target_df:
        Dataframe with inverse_transformed target

    Returns
    -------
    :
       Dataframe with inverse transformed target components
    """
    components_number = len(set(target_components_df.columns.get_level_values("feature")))
    scale_coef = np.repeat((inverse_transformed_target_df / target_df).values, repeats=components_number, axis=1)
    inverse_transformed_target_components_df = target_components_df * scale_coef
    return inverse_transformed_target_components_df


def _check_timestamp_param(
    param: Union[pd.Timestamp, int, str, None], param_name: str, freq: Union[pd.DateOffset, str, None]
) -> Union[pd.Timestamp, int, None]:
    if param is None:
        return param

    if freq is None:
        if not (isinstance(param, int) or isinstance(param, np.integer)):
            raise ValueError(
                f"Parameter {param_name} has incorrect type! For integer timestamp only integer parameter type is allowed."
            )

        return param
    else:
        if not isinstance(param, str) and not isinstance(param, pd.Timestamp):
            raise ValueError(
                f"Parameter {param_name} has incorrect type! For datetime timestamp only pd.Timestamp or str parameter type is allowed."
            )

        new_param = pd.Timestamp(param)
        return new_param


def determine_num_steps(
    start_timestamp: Union[pd.Timestamp, int],
    end_timestamp: Union[pd.Timestamp, int],
    freq: Union[pd.DateOffset, str, None],
) -> int:
    """Determine how many steps of ``freq`` should we make from ``start_timestamp`` to reach ``end_timestamp``.

    Parameters
    ----------
    start_timestamp:
        timestamp to start counting from
    end_timestamp:
        timestamp to end counting, should be not earlier than ``start_timestamp``
    freq:
        frequency of timestamps, possible values:

        - :py:class:`pandas.DateOffset` object for datetime timestamp

        - `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for datetime timestamp

        - None for integer timestamp

    Returns
    -------
    :
        number of steps

    Raises
    ------
    ValueError:
        Value of end timestamp is less than start timestamp
    ValueError:
        Start timestamp isn't correct according to a given frequency
    ValueError:
        End timestamp isn't correct according to a given frequency
    ValueError:
        End timestamp isn't reachable with a given frequency
    """
    freq_offset: Optional[pd.DateOffset] = pd.tseries.frequencies.to_offset(freq)
    if start_timestamp > end_timestamp:
        raise ValueError("Start timestamp should be less or equal than end timestamp!")

    if freq is None:
        if int(start_timestamp) != start_timestamp:
            raise ValueError(f"Start timestamp isn't correct according to given frequency: {freq}")
        if int(end_timestamp) != end_timestamp:
            raise ValueError(f"End timestamp isn't correct according to given frequency: {freq}")

        return end_timestamp - start_timestamp
    else:
        # check if start_timestamp is normalized
        normalized_start_timestamp = pd.date_range(start=start_timestamp, periods=1, freq=freq_offset)
        if normalized_start_timestamp != start_timestamp:
            raise ValueError(f"Start timestamp isn't correct according to given frequency: {freq}")

        # check a simple case
        if start_timestamp == end_timestamp:
            return 0

        # make linear probing, because for complex offsets there is a cycle in `pd.date_range`
        cur_value = 1
        cur_timestamp = start_timestamp
        while True:
            timestamps = pd.date_range(start=cur_timestamp, periods=2, freq=freq_offset)
            if timestamps[-1] == end_timestamp:
                return cur_value
            elif timestamps[-1] > end_timestamp:
                raise ValueError(f"End timestamp isn't reachable with freq: {freq}")
            cur_value += 1
            cur_timestamp = timestamps[-1]


class FreqFormat(str, Enum):
    """Enum for freq format for ``determine_freq``."""

    str = "str"
    offset = "offset"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} values allowed"
        )


def determine_freq(
    timestamps: Union[pd.Series, pd.Index], freq_format: str = FreqFormat.str
) -> Union[pd.DateOffset, str, None]:
    """Determine data frequency using provided timestamps.

    For integer timestamp the value ``None`` is returned.

    Parameters
    ----------
    timestamps:
        timeline to determine frequency
    freq_format:
        type of result, possible values:

        - "str" - frequency string result or ``None``

        - "offset" - :py:class:`pandas.DateOffset` result or ``None``

    Returns
    -------
    :
        pandas frequency string or offset depending on ``freq_format`` for datetime timestamp
        and ``None`` for int timestamp

    Raises
    ------
    ValueError:
        if incorrect freq_format is given
    ValueError:
        unable do determine frequency of data
    ValueError:
        integer timestamp isn't ordered and doesn't contain all the values from min to max
    """
    # check format
    freq_format_enum = FreqFormat(freq_format)

    # check integer timestamp
    if pd.api.types.is_integer_dtype(timestamps):
        diffs = np.diff(timestamps)[1:]
        if not np.all(diffs == 1):
            raise ValueError("Integer timestamp isn't ordered and doesn't contain all the values from min to max")

        return None

    # check datetime timestamp
    else:
        try:
            freq = pd.infer_freq(timestamps)
        except ValueError:
            freq = None

        if freq is None:
            raise ValueError("Can't determine frequency of a given dataframe")

        if freq_format_enum is FreqFormat.str:
            return freq
        elif freq_format_enum is FreqFormat.offset:
            return pd.tseries.frequencies.to_offset(freq)  # type: ignore
        else:
            assert_never(freq_format_enum)


def timestamp_range(
    start: Union[pd.Timestamp, int, str, None] = None,
    end: Union[pd.Timestamp, int, str, None] = None,
    periods: Optional[int] = None,
    freq: Union[pd.DateOffset, str, None] = None,
) -> pd.Index:
    """Create index with timestamps.

    Parameters
    ----------
    start:
        start of index
    end:
        end of index
    periods:
        length of the index
    freq:
        frequency of timestamps, possible values:

        - :py:class:`pandas.DateOffset` object for datetime timestamp

        - `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for datetime timestamp

        - None for integer timestamp

    Returns
    -------
    :
        Created index

    Raises
    ------
    ValueError:
        Incorrect type of ``start`` or ``end`` is used according to ``freq``
    ValueError:
        Of the three parameters: start, end, periods, exactly two must be specified
    """
    freq_offset: Optional[pd.DateOffset] = pd.tseries.frequencies.to_offset(freq)
    start = _check_timestamp_param(param=start, param_name="start", freq=freq_offset)
    end = _check_timestamp_param(param=end, param_name="end", freq=freq_offset)

    num_set = 0
    if start is not None:
        num_set += 1
    if end is not None:
        num_set += 1
    if periods is not None:
        num_set += 1
    if num_set != 2:
        raise ValueError("Of the three parameters: start, end, periods, exactly two must be specified!")

    if freq is None:
        if start is None:
            start = end - periods + 1  # type: ignore
        if periods is None:
            periods = end - start + 1  # type: ignore
        return pd.Index(np.arange(start, start + periods))
    else:
        return pd.date_range(start=start, end=end, periods=periods, freq=freq_offset)


def infer_alignment(df: pd.DataFrame) -> Union[Dict[str, pd.Timestamp], Dict[str, int]]:
    """Inference alignment of a given dataframe.

    Alignment tells us which timestamps for each segment should be considered to have the same integer timestamp after
    alignment transformation.

    For long dataframe the alignment is determined by the last timestamp for each segment.
    Last timestamp is taken without checking is 'target' value missing or not.

    Parameters
    ----------
    df:
        Dataframe in a long format.

    Returns
    -------
    :
        Dictionary with mapping segment -> timestamp.

    Raises
    ------
    ValueError:
        Parameter ``df`` isn't in a long format.
    """
    df_format = DataFrameFormat.determine(df=df)
    if df_format is not DataFrameFormat.long:
        raise ValueError("Parameter df should be in a long format!")

    return df.groupby(by=["segment"]).agg({"timestamp": "max"})["timestamp"].to_dict()


def apply_alignment(
    df: pd.DataFrame,
    alignment: Union[Dict[str, pd.Timestamp], Dict[str, int]],
    original_timestamp_name: Optional[str] = None,
):
    """Apply given alignment to a dataframe.

    Applying alignment creates a new dataframe in which we have a new 'timestamp' column
    with sequential integer timestamps.

    For each segment we sort timestamps and assign them sequential integer values (with step 1)
    in a way that timestamp in ``alignment`` gets value 0.

    Parameters
    ----------
    df:
        Dataframe in a long format.
    alignment:
        Alignment to apply.
    original_timestamp_name:
        Name for a column to save the original timestamps. If ``None`` original timestamps won't be saved.

    Returns
    -------
    :
        Aligned dataframe in a long format.

    Raises
    ------
    ValueError:
        Parameter ``df`` isn't in a long format.
    ValueError:
        There is a segment in ``df`` which isn't present in ``alignment``.
    ValueError:
        There is a segment which doesn't have a timestamp that is present in ``alignment``.
    """
    df_format = DataFrameFormat.determine(df=df)
    if df_format is not DataFrameFormat.long:
        raise ValueError("Parameter df should be in a long format!")

    df_list = []
    for segment in df["segment"].unique():
        if segment not in alignment:
            raise ValueError(f"The segment '{segment}' isn't present in alignment!")

        cur_df = df[df["segment"] == segment].sort_values(by="timestamp")
        reference_timestamp = alignment[segment]

        if reference_timestamp not in cur_df["timestamp"].values:
            raise ValueError(
                f"The segment '{segment}' doesn't contain timestamp '{reference_timestamp}' from alignment!"
            )

        reference_timestamp_index = pd.Index(cur_df["timestamp"]).get_loc(reference_timestamp)
        new_timestamp = np.arange(len(cur_df)) - reference_timestamp_index

        if original_timestamp_name is not None:
            cur_df.rename(columns={"timestamp": original_timestamp_name}, inplace=True)

        cur_df["timestamp"] = new_timestamp
        df_list.append(cur_df)

    result = pd.concat(df_list)
    return result


def make_timestamp_df_from_alignment(
    alignment: Union[Dict[str, pd.Timestamp], Dict[str, int]],
    start: Optional[int] = None,
    end: Optional[int] = None,
    periods: Optional[int] = None,
    freq: Union[pd.DateOffset, str, None] = None,
    timestamp_name: str = "external_timestamp",
):
    """Create a dataframe with timestamp according to a given alignment.

    This utility could be used after alignment of ``df`` to create ``df_exog`` with external timestamps
    extended into the future.

    For each segment we take ``start``, ``end``, ``periods`` and create sequential integer timestamps.
    After that we map this sequential integer timestamps into external timestamps according to ``alignment`` in a way
    that 0 translates into ``alignment[segment]`` timestamp and any other values are calculated based on ``freq``.

    Parameters
    ----------
    alignment:
        Alignment to use.
    start:
        Start timestamp to generate sequential integer timestamps.
    end:
        End timestamp to generate sequential integer timestamps.
    periods:
        Number of periods to generate sequential integer timestamps.
    freq:
        Frequency of timestamps to generate, possible values:

        - :py:class:`pandas.DateOffset` object for datetime timestamp

        - `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for datetime timestamp

        - None for integer timestamp

    timestamp_name:
        Name of created timestamp column.

    Returns
    -------
    :
        Dataframe with a created timestamp in a long format.
    """
    df_list = []
    timestamp = timestamp_range(start=start, end=end, periods=periods, freq=None)
    start = timestamp[0]
    start = cast(int, start)
    for segment, reference_timestamp in alignment.items():
        if start < 0:
            external_start_timestamp = timestamp_range(end=reference_timestamp, periods=-start + 1, freq=freq)[0]
        else:
            external_start_timestamp = timestamp_range(start=reference_timestamp, periods=start + 1, freq=freq)[-1]

        external_timestamp = timestamp_range(start=external_start_timestamp, periods=len(timestamp), freq=freq)
        cur_df = pd.DataFrame(
            {"segment": [segment] * len(timestamp), "timestamp": timestamp, timestamp_name: external_timestamp}
        )
        df_list.append(cur_df)

    result = pd.concat(df_list)
    return result


def _check_features_in_segments(columns: pd.MultiIndex, segments: Optional[List[str]] = None):
    """Check whether all segments have equal feature sets."""
    features_counts: DefaultDict[str, Counter] = defaultdict(Counter)
    for segment, feature in columns.to_flat_index():
        features_counts[segment].update([feature])

    if segments is not None and set(features_counts.keys()) != set(segments):
        raise ValueError(f"There is a mismatch in segments between provided and expected sets!")

    compare_counter = None
    compare_segment = None
    for segment, counter in features_counts.items():
        if compare_counter is None:
            compare_counter = counter
            compare_segment = segment
            continue

        if compare_counter != counter:
            raise ValueError(
                f"There is a mismatch in feature sets between segments '{compare_segment}' and '{segment}'!"
            )


def _slice_index_wide_dataframe(
    df: pd.DataFrame,
    start: Optional[Union[int, str, pd.Timestamp]] = None,
    stop: Optional[Union[int, str, pd.Timestamp]] = None,
    label_indexing: bool = True,
) -> pd.DataFrame:
    """Slice index of the dataframe in the wide format with copy."""
    indexer = df.loc if label_indexing else df.iloc

    # we want to make sure it makes only one copy
    df = indexer[start:stop]  # type: ignore
    if df._is_view or df._is_copy is not None:
        df = df.copy(deep=True)

    return df
