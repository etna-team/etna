from typing import Optional

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import duplicate_data
from etna.datasets import generate_ar_df
from etna.datasets.utils import _TorchDataset
from etna.datasets.utils import apply_alignment
from etna.datasets.utils import determine_freq
from etna.datasets.utils import determine_num_steps
from etna.datasets.utils import get_level_dataframe
from etna.datasets.utils import get_target_with_quantiles
from etna.datasets.utils import infer_alignment
from etna.datasets.utils import inverse_transform_target_components
from etna.datasets.utils import make_timestamp_df_from_alignment
from etna.datasets.utils import match_target_components
from etna.datasets.utils import set_columns_wide
from etna.datasets.utils import timestamp_range


@pytest.fixture
def df_exog_no_segments() -> pd.DataFrame:
    timestamp = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({"timestamp": timestamp, "exog_1": 1, "exog_2": 2, "exog_3": 3})
    return df


def test_duplicate_data_fail_empty_segments(df_exog_no_segments):
    """Test that `duplicate_data` fails on empty list of segments."""
    with pytest.raises(ValueError, match="Parameter segments shouldn't be empty"):
        _ = duplicate_data(df=df_exog_no_segments, segments=[])


def test_duplicate_data_fail_wrong_format(df_exog_no_segments):
    """Test that `duplicate_data` fails on wrong given format."""
    with pytest.raises(ValueError, match="'wrong_format' is not a valid DataFrameFormat"):
        _ = duplicate_data(df=df_exog_no_segments, segments=["segment_1", "segment_2"], format="wrong_format")


def test_duplicate_data_fail_wrong_df(df_exog_no_segments):
    """Test that `duplicate_data` fails on wrong df."""
    with pytest.raises(ValueError, match="There should be 'timestamp' column"):
        _ = duplicate_data(df=df_exog_no_segments.drop(columns=["timestamp"]), segments=["segment_1", "segment_2"])


def test_duplicate_data_long_format(df_exog_no_segments):
    """Test that `duplicate_data` makes duplication in long format."""
    segments = ["segment_1", "segment_2"]
    df_duplicated = duplicate_data(df=df_exog_no_segments, segments=segments, format="long")
    expected_columns = set(df_exog_no_segments.columns)
    expected_columns.add("segment")
    assert set(df_duplicated.columns) == expected_columns
    for segment in segments:
        df_temp = df_duplicated[df_duplicated["segment"] == segment].reset_index(drop=True)
        for column in df_exog_no_segments.columns:
            assert np.all(df_temp[column] == df_exog_no_segments[column])


def test_duplicate_data_wide_format(df_exog_no_segments):
    """Test that `duplicate_data` makes duplication in wide format."""
    segments = ["segment_1", "segment_2"]
    df_duplicated = duplicate_data(df=df_exog_no_segments, segments=segments, format="wide")
    expected_columns_segment = set(df_exog_no_segments.columns)
    expected_columns_segment.remove("timestamp")
    for segment in segments:
        df_temp = df_duplicated.loc[:, pd.IndexSlice[segment, :]]
        df_temp.columns = df_temp.columns.droplevel("segment")
        assert set(df_temp.columns) == expected_columns_segment
        assert np.all(df_temp.index == df_exog_no_segments["timestamp"])
        for column in df_exog_no_segments.columns.drop("timestamp"):
            assert np.all(df_temp[column].values == df_exog_no_segments[column].values)


def test_torch_dataset():
    """Unit test for `_TorchDataset` class."""
    ts_samples = [{"decoder_target": np.array([1, 2, 3]), "encoder_target": np.array([1, 2, 3])}]

    torch_dataset = _TorchDataset(ts_samples=ts_samples)

    assert torch_dataset[0] == ts_samples[0]
    assert len(torch_dataset) == 1


def _get_df_wide(freq: Optional[str], random_seed: int) -> pd.DataFrame:
    df = generate_ar_df(periods=5, n_segments=3, freq=freq, random_seed=random_seed)
    df_wide = TSDataset.to_dataset(df)

    df_exog = df.copy()
    df_exog = df_exog.rename(columns={"target": "exog_0"})
    df_exog["exog_0"] = df_exog["exog_0"] + 1
    df_exog["exog_1"] = df_exog["exog_0"] + 1
    df_exog["exog_2"] = df_exog["exog_1"] + 1
    df_exog_wide = TSDataset.to_dataset(df_exog)

    ts = TSDataset(df=df_wide, df_exog=df_exog_wide, freq=freq)
    df = ts.df

    # make some reorderings for checking corner cases
    df = df.loc[:, pd.IndexSlice[["segment_2", "segment_0", "segment_1"], ["target", "exog_2", "exog_1", "exog_0"]]]

    return df


@pytest.fixture
def df_left_datetime() -> pd.DataFrame:
    return _get_df_wide(freq="D", random_seed=0)


@pytest.fixture
def df_left_int() -> pd.DataFrame:
    return _get_df_wide(freq=None, random_seed=0)


@pytest.fixture
def df_right_datetime() -> pd.DataFrame:
    return _get_df_wide(freq="D", random_seed=1)


@pytest.fixture
def df_right_int() -> pd.DataFrame:
    return _get_df_wide(freq=None, random_seed=0)


@pytest.mark.parametrize(
    "features_left, features_right",
    [
        (None, None),
        (["exog_0"], ["exog_0"]),
        (["exog_0", "exog_1"], ["exog_0", "exog_1"]),
        (["exog_0", "exog_1"], ["exog_1", "exog_2"]),
    ],
)
@pytest.mark.parametrize(
    "segments_left, segment_right",
    [
        (None, None),
        (["segment_0"], ["segment_0"]),
        (["segment_0", "segment_1"], ["segment_0", "segment_1"]),
        (["segment_0", "segment_1"], ["segment_1", "segment_2"]),
    ],
)
@pytest.mark.parametrize(
    "timestamps_idx_left, timestamps_idx_right", [(None, None), ([0], [0]), ([1, 2], [1, 2]), ([1, 2], [3, 4])]
)
@pytest.mark.parametrize("dataframes", [("df_left_datetime", "df_right_datetime"), ("df_left_int", "df_right_int")])
def test_set_columns_wide(
    timestamps_idx_left,
    timestamps_idx_right,
    segments_left,
    segment_right,
    features_left,
    features_right,
    dataframes,
    request,
):
    df_left_name, df_right_name = dataframes
    df_left = request.getfixturevalue(df_left_name)
    df_right = request.getfixturevalue(df_right_name)

    timestamps_left = None if timestamps_idx_left is None else df_left.index[timestamps_idx_left]
    timestamps_right = None if timestamps_idx_right is None else df_right.index[timestamps_idx_right]

    df_obtained = set_columns_wide(
        df_left,
        df_right,
        timestamps_left=timestamps_left,
        timestamps_right=timestamps_right,
        segments_left=segments_left,
        segments_right=segment_right,
        features_left=features_left,
        features_right=features_right,
    )

    # get expected result
    df_expected = df_left.copy()

    timestamps_left_full = df_left.index.tolist() if timestamps_left is None else timestamps_left
    timestamps_right_full = df_right.index.tolist() if timestamps_left is None else timestamps_right

    segments_left_full = (
        df_left.columns.get_level_values("segment").unique().tolist() if segments_left is None else segments_left
    )
    segments_right_full = (
        df_left.columns.get_level_values("segment").unique().tolist() if segment_right is None else segment_right
    )

    features_left_full = (
        df_left.columns.get_level_values("feature").unique().tolist() if features_left is None else features_left
    )
    features_right_full = (
        df_left.columns.get_level_values("feature").unique().tolist() if features_right is None else features_right
    )

    right_value = df_right.loc[timestamps_right_full, pd.IndexSlice[segments_right_full, features_right_full]]
    df_expected.loc[timestamps_left_full, pd.IndexSlice[segments_left_full, features_left_full]] = right_value.values

    df_expected = df_expected.sort_index(axis=1)

    # compare values
    pd.testing.assert_frame_equal(df_obtained, df_expected)


@pytest.mark.parametrize("segments", (["s1"], ["s1", "s2"]))
@pytest.mark.parametrize(
    "columns,answer",
    (
        ({"a", "b"}, set()),
        ({"a", "b", "target"}, {"target"}),
        ({"a", "b", "target", "target_0.5"}, {"target", "target_0.5"}),
        ({"a", "b", "target", "target_0.5", "target1"}, {"target", "target_0.5"}),
        ({"target_component_a", "a", "b", "target_component_c", "target", "target_0.95"}, {"target", "target_0.95"}),
    ),
)
def test_get_target_with_quantiles(segments, columns, answer):
    columns = pd.MultiIndex.from_product([segments, columns], names=["segment", "feature"])
    targets_names = get_target_with_quantiles(columns)
    assert targets_names == answer


@pytest.mark.parametrize(
    "target_level, answer_name",
    (
        ("market", "market_level_constant_forecast_with_quantiles"),
        ("total", "total_level_constant_forecast_with_quantiles"),
    ),
)
def test_get_level_dataframe(product_level_constant_forecast_with_quantiles, target_level, answer_name, request):
    ts = product_level_constant_forecast_with_quantiles
    answer = request.getfixturevalue(answer_name).to_pandas()

    mapping_matrix = ts.hierarchical_structure.get_summing_matrix(
        target_level=target_level, source_level=ts.current_df_level
    )

    target_level_df = get_level_dataframe(
        df=ts.to_pandas(),
        mapping_matrix=mapping_matrix,
        source_level_segments=ts.hierarchical_structure.get_level_segments(level_name=ts.current_df_level),
        target_level_segments=ts.hierarchical_structure.get_level_segments(level_name=target_level),
    )

    pd.testing.assert_frame_equal(target_level_df, answer)


@pytest.mark.parametrize(
    "source_level_segments,target_level_segments,message",
    (
        (("ABC", "c1"), ("X", "Y"), "Segments mismatch for provided dataframe and `source_level_segments`!"),
        (("ABC", "a"), ("X", "Y"), "Segments mismatch for provided dataframe and `source_level_segments`!"),
        (
            ("a", "b", "c", "d"),
            ("X",),
            "Number of target level segments do not match mapping matrix number of columns!",
        ),
    ),
)
def test_get_level_dataframe_segm_errors(
    product_level_simple_hierarchical_ts, source_level_segments, target_level_segments, message
):
    ts = product_level_simple_hierarchical_ts

    mapping_matrix = product_level_simple_hierarchical_ts.hierarchical_structure.get_summing_matrix(
        target_level="market", source_level=ts.current_df_level
    )

    with pytest.raises(ValueError, match=message):
        get_level_dataframe(
            df=ts.df,
            mapping_matrix=mapping_matrix,
            source_level_segments=source_level_segments,
            target_level_segments=target_level_segments,
        )


@pytest.mark.parametrize(
    "features,answer",
    (
        (set(), set()),
        ({"a", "b"}, set()),
        (
            {"target_component_a", "a", "b", "target_component_c", "target", "target_0.95"},
            {"target_component_a", "target_component_c"},
        ),
    ),
)
def test_match_target_components(features, answer):
    components = match_target_components(features)
    assert components == answer


@pytest.fixture
def target_components_df():
    timestamp = pd.date_range("2021-01-01", "2021-01-05")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target_component_a": 1, "target_component_b": 2, "segment": 1})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target_component_a": 3, "target_component_b": 4, "segment": 2})
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def inverse_transformed_components_df():
    timestamp = pd.date_range("2021-01-01", "2021-01-05")
    df_1 = pd.DataFrame(
        {
            "timestamp": timestamp,
            "target_component_a": [1 * (i + 10) / i for i in range(1, 6)],
            "target_component_b": [2 * (i + 10) / i for i in range(1, 6)],
            "segment": 1,
        }
    )
    df_2 = pd.DataFrame(
        {
            "timestamp": timestamp,
            "target_component_a": [3 * (i + 10) / i for i in range(6, 11)],
            "target_component_b": [4 * (i + 10) / i for i in range(6, 11)],
            "segment": 2,
        }
    )
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def target_df():
    timestamp = pd.date_range("2021-01-01", "2021-01-05")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": range(1, 6), "segment": 1})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": range(6, 11), "segment": 2})
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def inverse_transformed_target_df():
    timestamp = pd.date_range("2021-01-01", "2021-01-05")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": range(11, 16), "segment": 1})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": range(16, 21), "segment": 2})
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    return df


def test_inverse_transform_target_components(
    target_components_df, target_df, inverse_transformed_target_df, inverse_transformed_components_df
):
    obtained_inverse_transformed_components_df = inverse_transform_target_components(
        target_components_df=target_components_df,
        target_df=target_df,
        inverse_transformed_target_df=inverse_transformed_target_df,
    )
    pd.testing.assert_frame_equal(obtained_inverse_transformed_components_df, inverse_transformed_components_df)


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp, freq, answer",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), "D", 1),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-11"), "D", 10),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01"), "D", 0),
        (pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-19"), "W-SUN", 2),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-15"), pd.offsets.Week(), 2),
        (pd.Timestamp("2020-01-31"), pd.Timestamp("2021-02-28"), "M", 13),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-01"), "MS", 17),
        (0, 0, None, 0),
        (0, 5, None, 5),
        (3, 10, None, 7),
    ],
)
def test_determine_num_steps_ok(start_timestamp, end_timestamp, freq, answer):
    result = determine_num_steps(start_timestamp=start_timestamp, end_timestamp=end_timestamp, freq=freq)
    assert result == answer


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp, freq",
    [
        (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-01"), "D"),
        (5, 2, None),
    ],
)
def test_determine_num_steps_fail_wrong_order(start_timestamp, end_timestamp, freq):
    with pytest.raises(ValueError, match="Start timestamp should be less or equal than end timestamp"):
        _ = determine_num_steps(start_timestamp=start_timestamp, end_timestamp=end_timestamp, freq=freq)


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp, freq",
    [
        (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-06-01"), "M"),
        (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-06-01"), "MS"),
        (2.2, 5, None),
    ],
)
def test_determine_num_steps_fail_wrong_start(start_timestamp, end_timestamp, freq):
    with pytest.raises(ValueError, match="Start timestamp isn't correct according to given frequency"):
        _ = determine_num_steps(start_timestamp=start_timestamp, end_timestamp=end_timestamp, freq=freq)


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp, freq",
    [
        (2, 5.5, None),
    ],
)
def test_determine_num_steps_fail_wrong_start(start_timestamp, end_timestamp, freq):
    with pytest.raises(ValueError, match="End timestamp isn't correct according to given frequency"):
        _ = determine_num_steps(start_timestamp=start_timestamp, end_timestamp=end_timestamp, freq=freq)


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp, freq",
    [
        (pd.Timestamp("2020-01-31"), pd.Timestamp("2020-06-05"), "M"),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-05"), "MS"),
    ],
)
def test_determine_num_steps_fail_wrong_end(start_timestamp, end_timestamp, freq):
    with pytest.raises(ValueError, match="End timestamp isn't reachable with freq"):
        _ = determine_num_steps(start_timestamp=start_timestamp, end_timestamp=end_timestamp, freq=freq)


@pytest.mark.parametrize(
    "timestamps,answer",
    (
        (pd.date_range(start="2020-01-01", periods=3, freq="M"), "M"),
        (pd.date_range(start="2020-01-01", periods=3, freq="W"), "W-SUN"),
        (pd.date_range(start="2020-01-01", periods=3, freq="D"), "D"),
        (pd.Series(np.arange(10)), None),
        (pd.Series(np.arange(5, 15)), None),
        (pd.Series(np.arange(1)), None),
    ),
)
def test_determine_freq(timestamps, answer):
    assert determine_freq(timestamps=timestamps) == answer


@pytest.mark.parametrize(
    "timestamps",
    (
        pd.to_datetime(pd.Series(["2020-02-01", "2020-02-15", "2021-02-15"])),
        pd.to_datetime(pd.Series(["2020-02-15", "2020-01-22", "2020-01-23"])),
    ),
)
def test_determine_freq_fail_cant_determine(timestamps):
    with pytest.raises(ValueError, match="Can't determine frequency of a given dataframe"):
        _ = determine_freq(timestamps=timestamps)


@pytest.mark.parametrize(
    "timestamps",
    (
        pd.Series([5, 4, 3]),
        pd.Series([4, 5, 3]),
        pd.Series([3, 4, 6]),
    ),
)
def test_determine_freq_fail_int_gaps(timestamps):
    with pytest.raises(ValueError, match="Integer timestamp isn't ordered and doesn't contain all the values"):
        _ = determine_freq(timestamps=timestamps)


@pytest.mark.parametrize(
    "start, end, periods, freq, expected_range",
    [
        ("2020-01-01", "2020-01-10", None, "D", pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")),
        ("2020-01-01", None, 10, "D", pd.date_range(start="2020-01-01", periods=10, freq="D")),
        (None, "2020-01-10", 10, "D", pd.date_range(end="2020-01-10", periods=10, freq="D")),
        ("2020-01-01", None, 10, "MS", pd.date_range(start="2020-01-01", periods=10, freq="MS")),
        (10, 19, None, None, np.arange(10, 20)),
        (10, None, 10, None, np.arange(10, 20)),
        (None, 19, 10, None, np.arange(10, 20)),
    ],
)
def test_timestamp_range(start, end, periods, freq, expected_range):
    result = timestamp_range(start=start, end=end, periods=periods, freq=freq)
    np.testing.assert_array_equal(result, expected_range)


@pytest.mark.parametrize(
    "start, end, periods, freq",
    [
        ("2020-01-01", "2020-01-10", None, None),
        ("2020-01-01", None, 10, None),
        (None, "2020-01-10", 10, None),
        ("2020-01-01", 20, None, "D"),
        (10, "2020-01-10", None, "D"),
        (10, 20, None, "D"),
        (10, None, 10, "D"),
        (None, 20, 10, "D"),
    ],
)
def test_timestamp_range_fail_type(start, end, periods, freq):
    with pytest.raises(ValueError, match="Parameter .* has incorrect type"):
        _ = timestamp_range(start=start, end=end, periods=periods, freq=freq)


@pytest.mark.parametrize(
    "start, end, periods, freq",
    [
        ("2020-01-01", "2020-01-10", 10, "D"),
        ("2020-01-01", None, None, "D"),
        (None, "2020-01-10", None, "D"),
        (None, None, 10, "D"),
        (None, None, None, "D"),
        (10, 19, 10, None),
        (10, None, None, None),
        (None, 19, None, None),
        (None, None, 10, None),
        (None, None, None, None),
    ],
)
def test_timestamp_range_fail_num_parameters(start, end, periods, freq):
    with pytest.raises(ValueError, match="Of the three parameters: .* must be specified"):
        _ = timestamp_range(start=start, end=end, periods=periods, freq=freq)


@pytest.fixture
def df_aligned_datetime() -> pd.DataFrame:
    df = generate_ar_df(start_time="2020-01-01", periods=10, n_segments=2, freq="D")
    return df


@pytest.fixture
def df_aligned_int() -> pd.DataFrame:
    df = generate_ar_df(start_time=10, periods=10, n_segments=2, freq=None)
    return df


@pytest.fixture
def df_misaligned_datetime() -> pd.DataFrame:
    df = generate_ar_df(start_time="2020-01-01", periods=10, n_segments=2, freq="D")
    df = df.iloc[:-3]
    return df


@pytest.fixture
def df_misaligned_int() -> pd.DataFrame:
    df = generate_ar_df(start_time=10, periods=10, n_segments=2, freq=None)
    df = df.iloc[:-3]
    return df


@pytest.fixture
def df_misaligned_datetime_with_missing_values() -> pd.DataFrame:
    df = generate_ar_df(start_time="2020-01-01", periods=10, n_segments=2, freq="D")
    df.loc[df.index[-3:], "target"] = np.NaN
    return df


@pytest.fixture
def df_misaligned_int_with_missing_values() -> pd.DataFrame:
    df = generate_ar_df(start_time=10, periods=10, n_segments=2, freq=None)
    df.loc[df.index[-3:], "target"] = np.NaN
    return df


@pytest.fixture
def df_aligned_datetime_with_additional_columns() -> pd.DataFrame:
    df = generate_ar_df(start_time="2020-01-01", periods=10, n_segments=2, freq="D")
    df["feature_1"] = df["timestamp"].dt.weekday
    return df


@pytest.mark.parametrize(
    "df_name, expected_alignment",
    [
        ("df_aligned_datetime", {"segment_0": pd.Timestamp("2020-01-10"), "segment_1": pd.Timestamp("2020-01-10")}),
        ("df_aligned_int", {"segment_0": 19, "segment_1": 19}),
        ("df_misaligned_datetime", {"segment_0": pd.Timestamp("2020-01-10"), "segment_1": pd.Timestamp("2020-01-07")}),
        ("df_misaligned_int", {"segment_0": 19, "segment_1": 16}),
        (
            "df_misaligned_datetime_with_missing_values",
            {"segment_0": pd.Timestamp("2020-01-10"), "segment_1": pd.Timestamp("2020-01-10")},
        ),
        ("df_misaligned_int_with_missing_values", {"segment_0": 19, "segment_1": 19}),
    ],
)
def test_infer_alignment(df_name, expected_alignment, request):
    df = request.getfixturevalue(df_name)
    alignment = infer_alignment(df)
    assert alignment == expected_alignment


@pytest.mark.parametrize(
    "df_name, alignment",
    [
        ("df_aligned_datetime", {}),
        ("df_aligned_datetime", {"segment_0": pd.Timestamp("2020-01-10")}),
        ("df_aligned_datetime", {"segment_1": pd.Timestamp("2020-01-10")}),
    ],
)
def test_apply_alignment_fail_no_segment(df_name, alignment, request):
    df = request.getfixturevalue(df_name)
    with pytest.raises(ValueError, match="The segment .* isn't present in alignment"):
        _ = apply_alignment(df=df, alignment=alignment)


@pytest.mark.parametrize(
    "df_name, alignment",
    [
        ("df_aligned_datetime", {"segment_0": pd.Timestamp("2020-01-20"), "segment_1": pd.Timestamp("2020-01-10")}),
        ("df_aligned_datetime", {"segment_0": pd.Timestamp("2020-01-10"), "segment_1": pd.Timestamp("2020-01-20")}),
    ],
)
def test_apply_alignment_fail_no_timestamp(df_name, alignment, request):
    df = request.getfixturevalue(df_name)
    with pytest.raises(ValueError, match="The segment .* doesn't contain timestamp .* from alignment"):
        _ = apply_alignment(df=df, alignment=alignment)


@pytest.mark.parametrize(
    "df_name, alignment, original_timestamp_name, expected_columns",
    [
        (
            "df_aligned_datetime",
            {"segment_0": pd.Timestamp("2020-01-10"), "segment_1": pd.Timestamp("2020-01-10")},
            None,
            {"timestamp", "segment", "target"},
        ),
        (
            "df_aligned_datetime",
            {"segment_0": pd.Timestamp("2020-01-10"), "segment_1": pd.Timestamp("2020-01-05")},
            None,
            {"timestamp", "segment", "target"},
        ),
        (
            "df_aligned_datetime",
            {"segment_0": pd.Timestamp("2020-01-10"), "segment_1": pd.Timestamp("2020-01-10")},
            "original_timestamp",
            {"timestamp", "segment", "target", "original_timestamp"},
        ),
        (
            "df_aligned_datetime_with_additional_columns",
            {"segment_0": pd.Timestamp("2020-01-10"), "segment_1": pd.Timestamp("2020-01-10")},
            None,
            {"timestamp", "segment", "target", "feature_1"},
        ),
        (
            "df_aligned_datetime_with_additional_columns",
            {"segment_0": pd.Timestamp("2020-01-10"), "segment_1": pd.Timestamp("2020-01-10")},
            "original_timestamp",
            {"timestamp", "segment", "target", "feature_1", "original_timestamp"},
        ),
        ("df_aligned_int", {"segment_0": 19, "segment_1": 19}, None, {"timestamp", "segment", "target"}),
        (
            "df_aligned_int",
            {"segment_0": 19, "segment_1": 19},
            "original_timestamp",
            {"timestamp", "segment", "target", "original_timestamp"},
        ),
    ],
)
def test_apply_alignment_format(df_name, alignment, original_timestamp_name, expected_columns, request):
    df = request.getfixturevalue(df_name)
    result_df = apply_alignment(df=df, alignment=alignment, original_timestamp_name=original_timestamp_name)

    assert len(result_df) == len(df)
    assert set(result_df.columns) == expected_columns


@pytest.mark.parametrize(
    "df_name, alignment",
    [
        (
            "df_aligned_datetime",
            {"segment_0": pd.Timestamp("2020-01-10"), "segment_1": pd.Timestamp("2020-01-10")},
        ),
        (
            "df_aligned_datetime",
            {"segment_0": pd.Timestamp("2020-01-10"), "segment_1": pd.Timestamp("2020-01-05")},
        ),
        (
            "df_aligned_datetime",
            {"segment_0": pd.Timestamp("2020-01-05"), "segment_1": pd.Timestamp("2020-01-10")},
        ),
        (
            "df_misaligned_datetime",
            {"segment_0": pd.Timestamp("2020-01-10"), "segment_1": pd.Timestamp("2020-01-07")},
        ),
        ("df_aligned_int", {"segment_0": 19, "segment_1": 19}),
        ("df_aligned_int", {"segment_0": 19, "segment_1": 14}),
        ("df_aligned_int", {"segment_0": 14, "segment_1": 19}),
        ("df_misaligned_int", {"segment_0": 19, "segment_1": 16}),
    ],
)
def test_apply_alignment_doesnt_change_original(df_name, alignment, request):
    df = request.getfixturevalue(df_name)
    result_df = apply_alignment(df=df[::-1], alignment=alignment, original_timestamp_name="original_timestamp")

    check_df = result_df.drop(columns=["timestamp"])
    check_df = check_df.rename(columns={"original_timestamp": "timestamp"})
    pd.testing.assert_frame_equal(check_df.loc[df.index], df)


@pytest.mark.parametrize(
    "df_name, alignment, expected_timestamps",
    [
        (
            "df_aligned_datetime",
            {"segment_0": pd.Timestamp("2020-01-10"), "segment_1": pd.Timestamp("2020-01-10")},
            list(range(-9, 1)) + list(range(-9, 1)),
        ),
        (
            "df_aligned_datetime",
            {"segment_0": pd.Timestamp("2020-01-10"), "segment_1": pd.Timestamp("2020-01-05")},
            list(range(-9, 1)) + list(range(-4, 6)),
        ),
        (
            "df_aligned_datetime",
            {"segment_0": pd.Timestamp("2020-01-05"), "segment_1": pd.Timestamp("2020-01-10")},
            list(range(-4, 6)) + list(range(-9, 1)),
        ),
        (
            "df_misaligned_datetime",
            {"segment_0": pd.Timestamp("2020-01-10"), "segment_1": pd.Timestamp("2020-01-07")},
            list(range(-9, 1)) + list(range(-6, 1)),
        ),
        ("df_aligned_int", {"segment_0": 19, "segment_1": 19}, list(range(-9, 1)) + list(range(-9, 1))),
        ("df_aligned_int", {"segment_0": 19, "segment_1": 14}, list(range(-9, 1)) + list(range(-4, 6))),
        ("df_aligned_int", {"segment_0": 14, "segment_1": 19}, list(range(-4, 6)) + list(range(-9, 1))),
        ("df_misaligned_int", {"segment_0": 19, "segment_1": 16}, list(range(-9, 1)) + list(range(-6, 1))),
    ],
)
def test_apply_alignment_new_timestamps(df_name, alignment, expected_timestamps, request):
    df = request.getfixturevalue(df_name)
    result_df = apply_alignment(df=df, alignment=alignment)

    np.testing.assert_array_equal(result_df["timestamp"], expected_timestamps)


@pytest.mark.parametrize(
    "alignment, start, end, periods, freq, timestamp_name, expected_timestamp",
    [
        (
            {"segment_0": pd.Timestamp("2020-01-01")},
            0,
            9,
            None,
            "D",
            "external_timestamp",
            timestamp_range(start="2020-01-01", periods=10, freq="D"),
        ),
        (
            {"segment_0": pd.Timestamp("2020-01-01")},
            2,
            11,
            None,
            "D",
            "external_timestamp",
            timestamp_range(start="2020-01-03", periods=10, freq="D"),
        ),
        (
            {"segment_0": pd.Timestamp("2020-01-01")},
            -2,
            7,
            None,
            "D",
            "external_timestamp",
            timestamp_range(start="2019-12-30", periods=10, freq="D"),
        ),
        (
            {"segment_0": pd.Timestamp("2020-01-01")},
            0,
            None,
            10,
            "D",
            "external_timestamp",
            timestamp_range(start="2020-01-01", periods=10, freq="D"),
        ),
        (
            {"segment_0": pd.Timestamp("2020-01-01")},
            None,
            9,
            10,
            "D",
            "external_timestamp",
            timestamp_range(start="2020-01-01", periods=10, freq="D"),
        ),
        (
            {"segment_0": pd.Timestamp("2020-01-01"), "segment_1": pd.Timestamp("2020-01-03")},
            0,
            9,
            None,
            "D",
            "external_timestamp",
            pd.concat(
                [
                    timestamp_range(start="2020-01-01", periods=10, freq="D").to_series(),
                    timestamp_range(start="2020-01-03", periods=10, freq="D").to_series(),
                ]
            ),
        ),
        ({"segment_0": 10}, 0, 9, None, None, "external_timestamp", timestamp_range(start=10, periods=10, freq=None)),
        ({"segment_0": 10}, 2, 11, None, None, "external_timestamp", timestamp_range(start=12, periods=10, freq=None)),
        ({"segment_0": 10}, -2, 7, None, None, "external_timestamp", timestamp_range(start=8, periods=10, freq=None)),
        ({"segment_0": 10}, 0, None, 10, None, "external_timestamp", timestamp_range(start=10, periods=10, freq=None)),
        ({"segment_0": 10}, None, 9, 10, None, "external_timestamp", timestamp_range(start=10, periods=10, freq=None)),
        (
            {"segment_0": 10, "segment_1": 12},
            0,
            9,
            None,
            None,
            "external_timestamp",
            pd.concat(
                [
                    timestamp_range(start=10, periods=10, freq=None).to_series(),
                    timestamp_range(start=12, periods=10, freq=None).to_series(),
                ]
            ),
        ),
    ],
)
def test_make_timestamp_df_from_alignment_format(
    alignment, start, end, periods, freq, timestamp_name, expected_timestamp
):
    df = make_timestamp_df_from_alignment(
        alignment=alignment, start=start, end=end, periods=periods, freq=freq, timestamp_name=timestamp_name
    )

    assert set(df.columns) == {"timestamp", "segment", timestamp_name}
    np.testing.assert_array_equal(df[timestamp_name], expected_timestamp)
