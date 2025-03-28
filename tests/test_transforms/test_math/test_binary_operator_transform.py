import operator
from copy import deepcopy

import numpy as np
import numpy.testing
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.transforms.math import binary_operator

ops = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
    "==": operator.eq,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


@pytest.fixture
def ts_one_segment(random_seed) -> TSDataset:
    """Generate dataset with non-positive target."""
    df = generate_ar_df(start_time="2020-01-01", periods=100, freq=pd.offsets.Day(), n_segments=1)
    df["feature"] = np.random.uniform(10, 0, size=100)
    df["target"] = np.random.uniform(10, 0, size=100)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq=pd.offsets.Day())
    return ts


@pytest.fixture
def ts_two_segments(random_seed) -> TSDataset:
    """Generate dataset with non-positive target."""
    df = generate_ar_df(start_time="2020-01-01", periods=100, freq=pd.offsets.Day(), n_segments=2)
    df["feature"] = np.random.uniform(10, 0, size=200)
    df["target"] = np.random.uniform(10, 0, size=200)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq=pd.offsets.Day())
    return ts


@pytest.mark.parametrize(
    "operand, left_column, right_column, out_column",
    [
        ("+", "feature", "target", "target"),
        ("-", "feature", "target", "target"),
        ("*", "feature", "target", "target"),
        ("/", "feature", "target", "target"),
        ("//", "feature", "target", "target"),
        ("%", "feature", "target", "target"),
        ("**", "feature", "target", "target"),
        ("==", "feature", "target", "target"),
        (">=", "feature", "target", "target"),
        ("<=", "feature", "target", "target"),
        (">", "feature", "target", "target"),
        ("<", "feature", "target", "target"),
        ("+", "feature", "target", "new_col"),
        ("-", "feature", "target", "new_col"),
        ("*", "feature", "target", "new_col"),
        ("/", "feature", "target", "new_col"),
        ("//", "feature", "target", "new_col"),
        ("%", "feature", "target", "new_col"),
        ("**", "feature", "target", "new_col"),
        ("==", "feature", "target", "new_col"),
        (">=", "feature", "target", "new_col"),
        ("<=", "feature", "target", "new_col"),
        (">", "feature", "target", "new_col"),
        ("<", "feature", "target", "new_col"),
        ("+", "feature", "target", None),
        ("-", "feature", "target", None),
        ("*", "feature", "target", None),
        ("/", "feature", "target", None),
        ("//", "feature", "target", None),
        ("%", "feature", "target", None),
        ("**", "feature", "target", None),
        ("==", "feature", "target", None),
        (">=", "feature", "target", None),
        ("<=", "feature", "target", None),
        (">", "feature", "target", None),
        ("<", "feature", "target", None),
    ],
)
def test_simple_one_segment(ts_one_segment: TSDataset, operand, left_column, right_column, out_column):
    left_vals = deepcopy(ts_one_segment._df["segment_0"][left_column].values)
    right_vals = deepcopy(ts_one_segment._df["segment_0"][right_column].values)
    checker_vals = deepcopy(ops[operand](left_vals, right_vals))
    transformer = binary_operator.BinaryOperationTransform(
        left_column=left_column, right_column=right_column, operator=operand, out_column=out_column
    )
    new_ts = transformer.fit_transform(ts=ts_one_segment)
    new_ts_vals = new_ts._df["segment_0"][transformer.out_column].to_numpy()
    numpy.testing.assert_array_almost_equal(new_ts_vals, checker_vals)
    if out_column is None:
        assert transformer.out_column == left_column + operand + right_column


@pytest.mark.parametrize(
    "operand, left_column, right_column, out_column",
    [
        ("+", "feature", "target", "target"),
        ("-", "feature", "target", "target"),
        ("*", "feature", "target", "target"),
        ("/", "feature", "target", "target"),
        ("//", "feature", "target", "target"),
        ("%", "feature", "target", "target"),
        ("**", "feature", "target", "target"),
        ("==", "feature", "target", "target"),
        (">=", "feature", "target", "target"),
        ("<=", "feature", "target", "target"),
        (">", "feature", "target", "target"),
        ("<", "feature", "target", "target"),
        ("+", "feature", "target", "new_col"),
        ("-", "feature", "target", "new_col"),
        ("*", "feature", "target", "new_col"),
        ("/", "feature", "target", "new_col"),
        ("//", "feature", "target", "new_col"),
        ("%", "feature", "target", "new_col"),
        ("**", "feature", "target", "new_col"),
        ("==", "feature", "target", "new_col"),
        (">=", "feature", "target", "new_col"),
        ("<=", "feature", "target", "new_col"),
        (">", "feature", "target", "new_col"),
        ("<", "feature", "target", "new_col"),
        ("+", "feature", "target", None),
        ("-", "feature", "target", None),
        ("*", "feature", "target", None),
        ("/", "feature", "target", None),
        ("//", "feature", "target", None),
        ("%", "feature", "target", None),
        ("**", "feature", "target", None),
        ("==", "feature", "target", None),
        (">=", "feature", "target", None),
        ("<=", "feature", "target", None),
        (">", "feature", "target", None),
        ("<", "feature", "target", None),
    ],
)
def test_simple_two_segments(ts_two_segments: TSDataset, operand, left_column, right_column, out_column):
    left_vals1 = deepcopy(ts_two_segments._df["segment_0"][left_column].values)
    right_vals1 = deepcopy(ts_two_segments._df["segment_0"][right_column].values)
    left_vals2 = deepcopy(ts_two_segments._df["segment_1"][left_column].values)
    right_vals2 = deepcopy(ts_two_segments._df["segment_1"][right_column].values)
    checker_vals1 = ops[operand](left_vals1, right_vals1)
    checker_vals2 = ops[operand](left_vals2, right_vals2)
    transformer = binary_operator.BinaryOperationTransform(
        left_column=left_column, right_column=right_column, operator=operand, out_column=out_column
    )
    new_ts = transformer.fit_transform(ts=ts_two_segments)
    new_ts_vals1 = new_ts._df["segment_0"][transformer.out_column].to_numpy()
    new_ts_vals2 = new_ts._df["segment_1"][transformer.out_column].to_numpy()
    numpy.testing.assert_array_almost_equal(new_ts_vals1, checker_vals1)
    numpy.testing.assert_array_almost_equal(new_ts_vals2, checker_vals2)
    if out_column is None:
        assert transformer.out_column == left_column + operand + right_column


@pytest.mark.parametrize(
    "operand, left_column, right_column, out_column",
    [
        ("+", "feature", "target", "target"),
        ("-", "feature", "target", "target"),
        ("*", "feature", "target", "target"),
        ("/", "feature", "target", "target"),
        ("+", "target", "feature", "target"),
        ("-", "target", "feature", "target"),
        ("*", "target", "feature", "target"),
        ("/", "target", "feature", "target"),
        ("+", "feature", "target", "feature"),
        ("-", "feature", "target", "feature"),
        ("*", "feature", "target", "feature"),
        ("/", "feature", "target", "feature"),
        ("+", "target", "feature", "feature"),
        ("-", "target", "feature", "feature"),
        ("*", "target", "feature", "feature"),
        ("/", "target", "feature", "feature"),
    ],
)
def test_inverse_one_segment(ts_one_segment, operand, left_column, right_column, out_column):
    target_vals = deepcopy(ts_one_segment._df["segment_0"][out_column].values)
    transformer = binary_operator.BinaryOperationTransform(
        left_column=left_column, right_column=right_column, operator=operand, out_column=out_column
    )
    new_ts = transformer.fit_transform(ts=ts_one_segment)
    new_ts = transformer.inverse_transform(ts=new_ts)
    new_ts_vals = new_ts._df["segment_0"][out_column].to_numpy()
    numpy.testing.assert_array_almost_equal(new_ts_vals, target_vals)


@pytest.mark.parametrize(
    "operand, left_column, right_column, out_column",
    [
        ("+", "feature", "target", "target"),
        ("-", "feature", "target", "target"),
        ("*", "feature", "target", "target"),
        ("/", "feature", "target", "target"),
        ("+", "target", "feature", "target"),
        ("-", "target", "feature", "target"),
        ("*", "target", "feature", "target"),
        ("/", "target", "feature", "target"),
        ("+", "feature", "target", "feature"),
        ("-", "feature", "target", "feature"),
        ("*", "feature", "target", "feature"),
        ("/", "feature", "target", "feature"),
        ("+", "target", "feature", "feature"),
        ("-", "target", "feature", "feature"),
        ("*", "target", "feature", "feature"),
        ("/", "target", "feature", "feature"),
    ],
)
def test_inverse_two_segments(ts_two_segments, operand, left_column, right_column, out_column):
    target_vals1 = deepcopy(ts_two_segments._df["segment_0"][out_column].values)
    target_vals2 = deepcopy(ts_two_segments._df["segment_1"][out_column].values)
    transformer = binary_operator.BinaryOperationTransform(
        left_column=left_column, right_column=right_column, operator=operand, out_column=out_column
    )
    new_ts = transformer.fit_transform(ts=ts_two_segments)
    new_ts = transformer.inverse_transform(ts=new_ts)
    new_ts_vals1 = new_ts._df["segment_0"][out_column].to_numpy()
    new_ts_vals2 = new_ts._df["segment_1"][out_column].to_numpy()
    numpy.testing.assert_array_almost_equal(new_ts_vals1, target_vals1)
    numpy.testing.assert_array_almost_equal(new_ts_vals2, target_vals2)


@pytest.mark.parametrize(
    "operand",
    [
        "//",
        "%",
        "**",
        "==",
        ">=",
        "<=",
        ">",
        "<",
    ],
)
def test_inverse_failed_unsupported_operator(ts_one_segment, operand):
    transformer = binary_operator.BinaryOperationTransform(
        left_column="feature", right_column="target", operator=operand, out_column="target"
    )
    with pytest.raises(
        ValueError,
        match="We only support inverse transform if the original operation is .+, .-, .*, ./",
    ):
        _ = transformer.inverse_transform(ts=ts_one_segment)


@pytest.mark.parametrize(
    "operand, left_column, right_column, out_column",
    [
        ("+", "feature", "target", "new_col"),
        ("-", "feature", "target", "new_col"),
        ("*", "feature", "target", "new_col"),
        ("/", "feature", "target", "new_col"),
    ],
)
def test_inverse_failed_not_inplace(ts_one_segment, operand, left_column, right_column, out_column):
    left_vals = deepcopy(ts_one_segment._df["segment_0"][left_column].values)
    right_vals = deepcopy(ts_one_segment._df["segment_0"][right_column].values)
    checker_vals = deepcopy(ops[operand](left_vals, right_vals))
    transformer = binary_operator.BinaryOperationTransform(
        left_column=left_column, right_column=right_column, operator=operand, out_column=out_column
    )
    new_ts = transformer.fit_transform(ts=ts_one_segment)
    new_ts = transformer.inverse_transform(ts=new_ts)
    new_ts_vals = new_ts._df["segment_0"][out_column].to_numpy()
    numpy.testing.assert_array_almost_equal(new_ts_vals, checker_vals)
