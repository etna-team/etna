import operator

import numpy as np
import numpy.testing
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.transforms.math import binary_operator

ops = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "%": operator.mod,
    "==": operator.eq,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


@pytest.fixture
def simple_ts_(random_seed) -> TSDataset:
    """Generate dataset with non-positive target."""
    periods = 100
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df["segment"] = "segment"
    df["feature"] = np.random.uniform(-5, 0, size=periods)
    df["target"] = np.random.uniform(-10, 0, size=periods)
    df = TSDataset.to_dataset(df)

    ts = TSDataset(df, freq="D")
    return ts


@pytest.mark.parametrize(
    "operand",
    [
        "+",
        "-",
        "*",
        "/",
        "%",
        "==",
        ">=",
        "<=",
        ">",
        "<",
    ],
)
def test_simple_change_target(simple_ts_: TSDataset, operand):
    target_vals = simple_ts_.df["segment"]["target"].values
    feature_vals = simple_ts_.df["segment"]["feature"].values
    checker_vals = ops[operand](feature_vals, target_vals)
    transformer = binary_operator.BinaryOperationTransform(
        left_operand="feature", right_operand="target", operator=operand, out_column="target"
    )
    new_ts = transformer.fit_transform(ts=simple_ts_)
    new_ts_vals = new_ts.df["segment"]["target"].to_numpy()
    numpy.testing.assert_array_almost_equal(new_ts_vals, checker_vals)


@pytest.mark.parametrize(
    "operand",
    [
        "+",
        "-",
        "*",
        "/",
        "%",
        "==",
        ">=",
        "<=",
        ">",
        "<",
    ],
)
def test_simple_add_column(simple_ts_, operand):
    target_vals = simple_ts_.df["segment"]["target"].values
    feature_vals = simple_ts_.df["segment"]["feature"].values
    checker_vals = ops[operand](feature_vals, target_vals)
    transformer = binary_operator.BinaryOperationTransform(
        left_operand="feature", right_operand="target", operator=operand, out_column="new_col"
    )
    new_ts = transformer.fit_transform(ts=simple_ts_)
    new_ts_vals = new_ts.df["segment"]["new_col"].to_numpy()
    numpy.testing.assert_array_almost_equal(new_ts_vals, checker_vals)


@pytest.mark.parametrize(
    "operand",
    [
        "+",
        "-",
        "*",
        "/",
    ],
)
def test_inverse(simple_ts_, operand):
    target_vals = simple_ts_.df["segment"]["target"].values
    transformer = binary_operator.BinaryOperationTransform(
        left_operand="feature", right_operand="target", operator=operand, out_column="target"
    )
    new_ts = transformer.fit_transform(ts=simple_ts_)
    new_ts = transformer.inverse_transform(ts=new_ts)
    new_ts_vals = new_ts.df["segment"]["target"].to_numpy()
    numpy.testing.assert_array_almost_equal(new_ts_vals, target_vals)


@pytest.mark.parametrize(
    "operand",
    [
        "%",
        "==",
        ">=",
        "<=",
        ">",
        "<",
    ],
)
def test_inverse_failed(simple_ts_, operand):
    target_vals = simple_ts_.df["segment"]["target"].values
    transformer = binary_operator.BinaryOperationTransform(
        left_operand="feature", right_operand="target", operator=operand, out_column="target"
    )
    with pytest.raises(
        ValueError,
    ):
        new_ts = transformer.inverse_transform(ts=simple_ts_)
