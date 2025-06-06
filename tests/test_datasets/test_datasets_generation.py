import numpy as np
import pandas as pd
import pytest

from etna.datasets.datasets_generation import generate_ar_df
from etna.datasets.datasets_generation import generate_const_df
from etna.datasets.datasets_generation import generate_from_patterns_df
from etna.datasets.datasets_generation import generate_hierarchical_df
from etna.datasets.datasets_generation import generate_periodic_df
from etna.datasets.tsdataset import TSDataset


def check_equals(generated_value, expected_value, **kwargs):
    """Check that generated_value is equal to expected_value."""
    return generated_value == expected_value


def check_not_equal_within_3_sigma(generated_value, expected_value, sigma, **kwargs):
    """Check that generated_value is not equal to expected_value, but within 3 sigma range."""
    if generated_value == expected_value:
        return False
    return abs(generated_value - expected_value) <= 3 * sigma


@pytest.mark.parametrize("generate_method", [generate_ar_df, generate_periodic_df, generate_const_df])
@pytest.mark.parametrize("freq", [None, pd.offsets.Day().freqstr, pd.offsets.Day()])
@pytest.mark.parametrize("periods, n_segments", [(5, 1), (6, 2)])
def test_generate_method_columns_set(generate_method, freq, periods, n_segments):
    expected_columns = {"timestamp", "segment", "target"}
    df = generate_method(periods=periods, n_segments=n_segments, freq=freq)
    assert set(df.columns) == expected_columns


@pytest.mark.parametrize("freq", [None, pd.offsets.Day().freqstr, pd.offsets.Day()])
@pytest.mark.parametrize("periods, patterns", [(5, [[0, 1], [0, 2, 1]])])
def test_generate_from_patterns_df_columns_set(freq, periods, patterns):
    expected_columns = {"timestamp", "segment", "target"}
    df = generate_from_patterns_df(periods=periods, patterns=patterns, freq=freq, start_time=None)
    assert set(df.columns) == expected_columns


@pytest.mark.parametrize("generate_method", [generate_ar_df, generate_periodic_df, generate_const_df])
@pytest.mark.parametrize("freq", [None, pd.offsets.Day().freqstr, pd.offsets.Day()])
@pytest.mark.parametrize("periods, n_segments", [(5, 1), (6, 2)])
def test_generate_method_periods(generate_method, freq, periods, n_segments):
    df = generate_method(periods=periods, n_segments=n_segments, freq=freq)
    assert df["timestamp"].nunique() == periods


@pytest.mark.parametrize("freq", [None, pd.offsets.Day().freqstr, pd.offsets.Day()])
@pytest.mark.parametrize("periods, patterns", [(5, [[0, 1], [0, 2, 1]])])
def test_generate_from_patterns_df_periods(freq, periods, patterns):
    df = generate_from_patterns_df(periods=periods, patterns=patterns, freq=freq, start_time=None)
    assert df["timestamp"].nunique() == periods


@pytest.mark.parametrize("generate_method", [generate_ar_df, generate_periodic_df, generate_const_df])
@pytest.mark.parametrize("freq", [None, pd.offsets.Day().freqstr, pd.offsets.Day()])
@pytest.mark.parametrize("periods, n_segments", [(5, 1), (6, 2)])
def test_generate_method_segments(generate_method, freq, periods, n_segments):
    df = generate_method(periods=periods, n_segments=n_segments, freq=freq)
    assert df["segment"].nunique() == n_segments


@pytest.mark.parametrize("freq", [None, pd.offsets.Day().freqstr, pd.offsets.Day()])
@pytest.mark.parametrize("periods, patterns", [(5, [[0, 1], [0, 2, 1]])])
def test_generate_from_patterns_df_segments(freq, periods, patterns):
    df = generate_from_patterns_df(periods=periods, patterns=patterns, freq=freq, start_time=None)
    assert df["segment"].nunique() == len(patterns)


@pytest.mark.parametrize("generate_method", [generate_ar_df, generate_periodic_df, generate_const_df])
@pytest.mark.parametrize(
    "start_time, expected_start_time, freq",
    [
        (None, 0, None),
        (1, 1, None),
        (None, pd.Timestamp("2000-01-01"), pd.offsets.Day().freqstr),
        ("2020-01-01", pd.Timestamp("2020-01-01"), pd.offsets.Day().freqstr),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01"), pd.offsets.Day().freqstr),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01"), pd.offsets.Day()),
    ],
)
@pytest.mark.parametrize("periods, n_segments", [(5, 1), (6, 2)])
def test_generate_method_start_time(generate_method, start_time, expected_start_time, freq, periods, n_segments):
    df = generate_method(periods=periods, n_segments=n_segments, freq=freq, start_time=start_time)
    assert df["timestamp"].min() == expected_start_time


@pytest.mark.parametrize(
    "start_time, expected_start_time, freq",
    [
        (None, 0, None),
        (1, 1, None),
        (None, pd.Timestamp("2000-01-01"), pd.offsets.Day().freqstr),
        ("2020-01-01", pd.Timestamp("2020-01-01"), pd.offsets.Day().freqstr),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01"), pd.offsets.Day().freqstr),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01"), pd.offsets.Day()),
    ],
)
@pytest.mark.parametrize("periods, patterns", [(5, [[0, 1], [0, 2, 1]])])
def test_generate_from_patterns_df_start_time(start_time, expected_start_time, freq, periods, patterns):
    df = generate_from_patterns_df(periods=periods, patterns=patterns, freq=freq, start_time=start_time)
    assert df["timestamp"].min() == expected_start_time


@pytest.mark.parametrize("generate_method", [generate_ar_df, generate_periodic_df, generate_const_df])
@pytest.mark.parametrize(
    "start_time, freq",
    [("2020-01-01", None), (pd.Timestamp("2020-01-01"), None), (10, pd.offsets.Day().freqstr), (10, pd.offsets.Day())],
)
@pytest.mark.parametrize("periods, n_segments", [(5, 1), (6, 2)])
def test_generate_method_timestamp_start_time_fail(generate_method, start_time, freq, periods, n_segments):
    with pytest.raises(ValueError, match="Parameter start_time has incorrect type"):
        _ = generate_method(periods=periods, n_segments=n_segments, freq=freq, start_time=start_time)


@pytest.mark.parametrize(
    "start_time, freq",
    [("2020-01-01", None), (pd.Timestamp("2020-01-01"), None), (10, pd.offsets.Day().freqstr), (10, pd.offsets.Day())],
)
@pytest.mark.parametrize("periods, patterns", [(5, [[0, 1], [0, 2, 1]])])
def test_generate_from_patterns_df_start_time_fail(start_time, freq, periods, patterns):
    with pytest.raises(ValueError, match="Parameter start_time has incorrect type"):
        _ = generate_from_patterns_df(periods=periods, patterns=patterns, freq=freq, start_time=start_time)


def test_generate_ar_df_values():
    ar_coef = [10, 11]
    random_seed = 1
    periods = 10
    random_numbers = np.random.RandomState(seed=random_seed).normal(size=(2, periods))
    ar_df = generate_ar_df(
        periods=periods, start_time="2020-01-01", n_segments=2, ar_coef=ar_coef, random_seed=random_seed
    )

    assert len(ar_df) == 2 * periods
    np.testing.assert_allclose(ar_df.iat[0, 2], random_numbers[0, 0])
    np.testing.assert_allclose(ar_df.iat[1, 2], ar_coef[0] * ar_df.iat[0, 2] + random_numbers[0, 1])
    np.testing.assert_allclose(
        ar_df.iat[2, 2], ar_coef[1] * ar_df.iat[0, 2] + ar_coef[0] * ar_df.iat[1, 2] + random_numbers[0, 2]
    )


@pytest.mark.parametrize("add_noise, checker", [(False, check_equals), (True, check_not_equal_within_3_sigma)])
def test_generate_periodic_df_values(add_noise, checker):
    period = 3
    periods = 11
    sigma = 0.1
    periodic_df = generate_periodic_df(
        periods=periods,
        start_time="2020-01-01",
        n_segments=2,
        period=period,
        add_noise=add_noise,
        sigma=sigma,
        random_seed=1,
    )
    assert len(periodic_df) == 2 * periods
    diff_sigma = np.sqrt(2) * sigma
    assert checker(periodic_df.iat[0, 2], periodic_df.iat[0 + period, 2], sigma=diff_sigma)
    assert checker(periodic_df.iat[1, 2], periodic_df.iat[1 + period, 2], sigma=diff_sigma)
    assert checker(periodic_df.iat[3, 2], periodic_df.iat[3 + period, 2], sigma=diff_sigma)


@pytest.mark.parametrize("add_noise, checker", [(False, check_equals), (True, check_not_equal_within_3_sigma)])
def test_generate_const_df_values(add_noise, checker):
    const = 1
    periods = 3
    sigma = 0.1
    const_df = generate_const_df(
        start_time="2020-01-01",
        n_segments=2,
        periods=periods,
        scale=const,
        add_noise=add_noise,
        sigma=sigma,
        random_seed=1,
    )
    assert len(const_df) == 2 * periods
    assert checker(const_df.iat[0, 2], const, sigma=sigma)
    assert checker(const_df.iat[1, 2], const, sigma=sigma)
    assert checker(const_df.iat[3, 2], const, sigma=sigma)


@pytest.mark.parametrize("add_noise, checker", [(False, check_equals), (True, check_not_equal_within_3_sigma)])
def test_generate_from_patterns_df_values(add_noise, checker):
    patterns = [[0, 1], [0, 2, 1]]
    periods = 10
    sigma = 0.1
    from_patterns_df = generate_from_patterns_df(
        start_time="2020-01-01", patterns=patterns, periods=periods, add_noise=add_noise, sigma=sigma, random_seed=1
    )
    assert len(from_patterns_df) == len(patterns) * periods
    assert checker(from_patterns_df[from_patterns_df.segment == "segment_0"].iat[0, 2], patterns[0][0], sigma=sigma)
    assert checker(from_patterns_df[from_patterns_df.segment == "segment_0"].iat[1, 2], patterns[0][1], sigma=sigma)
    assert checker(from_patterns_df[from_patterns_df.segment == "segment_0"].iat[2, 2], patterns[0][0], sigma=sigma)
    assert checker(from_patterns_df[from_patterns_df.segment == "segment_1"].iat[0, 2], patterns[1][0], sigma=sigma)
    assert checker(from_patterns_df[from_patterns_df.segment == "segment_1"].iat[3, 2], patterns[1][0], sigma=sigma)
    assert checker(from_patterns_df[from_patterns_df.segment == "segment_1"].iat[4, 2], patterns[1][1], sigma=sigma)


def test_generate_hierarchical_df_empty_n_segments_error():
    with pytest.raises(ValueError, match="`n_segments` should contain at least one positive integer!"):
        generate_hierarchical_df(periods=2, n_segments=[])


@pytest.mark.parametrize("n_segments", [[-1], [0, 2]])
def test_generate_hierarchical_df_negative_size_error(n_segments):
    with pytest.raises(ValueError, match="All `n_segments` elements should be positive!"):
        generate_hierarchical_df(periods=2, n_segments=n_segments)


@pytest.mark.parametrize(
    "periods,n_segments,expected_columns",
    (
        (2, [1, 2], {"target", "timestamp", "level_0", "level_1"}),
        (2, [2], {"target", "timestamp", "level_0"}),
        (4, [3, 4], {"target", "timestamp", "level_0", "level_1"}),
        (4, [3, 3], {"target", "timestamp", "level_0", "level_1"}),
    ),
)
@pytest.mark.parametrize("freq", [None, pd.offsets.Day().freqstr, pd.offsets.Day()])
def test_generate_hierarchical_df_columns_set(periods, n_segments, expected_columns, freq):
    hierarchical_df = generate_hierarchical_df(periods=periods, n_segments=n_segments, freq=freq)
    assert expected_columns == set(hierarchical_df.columns)


@pytest.mark.parametrize("periods,n_segments", ((2, [1, 2]), (2, [2]), (4, [3, 4]), (4, [3, 3])))
@pytest.mark.parametrize("freq", [None, pd.offsets.Day().freqstr, pd.offsets.Day()])
def test_generate_hierarchical_df_periods(periods, n_segments, freq):
    hierarchical_df = generate_hierarchical_df(periods=periods, n_segments=n_segments, freq=freq)
    assert hierarchical_df["timestamp"].nunique() == periods


@pytest.mark.parametrize("periods,n_segments", ((2, [1, 2]), (2, [2]), (4, [3, 4]), (4, [3, 3])))
@pytest.mark.parametrize("freq", [None, pd.offsets.Day().freqstr, pd.offsets.Day()])
def test_generate_hierarchical_df_segments(periods, n_segments, freq):
    hierarchical_df = generate_hierarchical_df(periods=periods, n_segments=n_segments, freq=freq)

    for level_id, segment_size in enumerate(n_segments):
        assert hierarchical_df[f"level_{level_id}"].nunique() == segment_size


@pytest.mark.parametrize("periods,n_segments", ((2, [1, 2]), (2, [2]), (4, [3, 4]), (4, [3, 3])))
@pytest.mark.parametrize("freq", [None, pd.offsets.Day().freqstr, pd.offsets.Day()])
def test_generate_hierarchical_df_segments_names(periods, n_segments, freq):
    hierarchical_df = generate_hierarchical_df(periods=periods, n_segments=n_segments, freq=freq)

    num_levels = len(n_segments)
    for level_id in range(num_levels):
        assert all(hierarchical_df[f"level_{level_id}"].str.match(rf"l{level_id}s\d+"))


@pytest.mark.parametrize("periods,n_segments", ((2, [1, 2]), (2, [2]), (4, [3, 4]), (4, [3, 3])))
@pytest.mark.parametrize("freq", [None, pd.offsets.Day().freqstr, pd.offsets.Day()])
def test_generate_hierarchical_df_convert_to_wide_format(periods, n_segments, freq):
    hierarchical_df = generate_hierarchical_df(periods=periods, n_segments=n_segments, freq=freq)
    level_names = [f"level_{idx}" for idx in range(len(n_segments))]
    TSDataset.to_hierarchical_dataset(df=hierarchical_df, level_columns=level_names)


@pytest.mark.parametrize("start_time", ["2020-01-01", pd.Timestamp("2020-01-01")])
@pytest.mark.parametrize("periods,n_segments", ((2, [1, 2]), (2, [2]), (4, [3, 4]), (4, [3, 3])))
def test_generate_hierarchical_df_start_time_fail(start_time, periods, n_segments):
    with pytest.raises(ValueError, match="Parameter start_time has incorrect type"):
        _ = generate_hierarchical_df(periods=periods, n_segments=n_segments, freq=None, start_time=start_time)
