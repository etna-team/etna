import re
from enum import Enum
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from etna import SETTINGS

if SETTINGS.torch_required:
    from torch.utils.data import Dataset
else:
    from unittest.mock import Mock

    Dataset = Mock  # type: ignore


class DataFrameFormat(str, Enum):
    """Enum for different types of result."""

    wide = "wide"
    long = "long"


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
        if incorrect strategy is given
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
    >>> df_ts_format = TSDataset.to_dataset(df)
    >>> ts = TSDataset(df=df_ts_format, df_exog=df_exog, freq="D", known_future="all")
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
    segments_results = []
    for segment in segments:
        df_segment = df.copy()
        df_segment["segment"] = segment
        segments_results.append(df_segment)

    df_long = pd.concat(segments_results, ignore_index=True)

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


def match_target_quantiles(features: Set[str]) -> Set[str]:
    """Find quantiles in dataframe columns."""
    pattern = re.compile(r"target_\d+\.\d+$")
    return {i for i in list(features) if pattern.match(i) is not None}


def match_target_components(features: Set[str]) -> Set[str]:
    """Find target components in a set of features."""
    return set(filter(lambda f: f.startswith("target_component_"), features))


def get_target_with_quantiles(columns: pd.Index) -> Set[str]:
    """Find "target" column and target quantiles among dataframe columns."""
    column_names = set(columns.get_level_values(level="feature"))
    target_columns = match_target_quantiles(column_names)
    if "target" in column_names:
        target_columns.add("target")
    return target_columns


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
    param: Union[pd.Timestamp, int, str, None], param_name: str, freq: Optional[str]
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
    start_timestamp: Union[pd.Timestamp, int], end_timestamp: Union[pd.Timestamp, int], freq: Optional[str]
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
        normalized_start_timestamp = pd.date_range(start=start_timestamp, periods=1, freq=freq)
        if normalized_start_timestamp != start_timestamp:
            raise ValueError(f"Start timestamp isn't correct according to given frequency: {freq}")

        # check a simple case
        if start_timestamp == end_timestamp:
            return 0

        # make linear probing, because for complex offsets there is a cycle in `pd.date_range`
        cur_value = 1
        cur_timestamp = start_timestamp
        while True:
            timestamps = pd.date_range(start=cur_timestamp, periods=2, freq=freq)
            if timestamps[-1] == end_timestamp:
                return cur_value
            elif timestamps[-1] > end_timestamp:
                raise ValueError(f"End timestamp isn't reachable with freq: {freq}")
            cur_value += 1
            cur_timestamp = timestamps[-1]


def determine_freq(timestamps: Union[pd.Series, pd.Index]) -> Optional[str]:
    """Determine data frequency using provided timestamps.

    Parameters
    ----------
    timestamps:
        timeline to determine frequency

    Returns
    -------
    :
        pandas frequency string

    Raises
    ------
    ValueError:
        unable do determine frequency of data
    ValueError:
        integer timestamp isn't ordered and doesn't contain all the values from min to max
    """
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

        return freq
