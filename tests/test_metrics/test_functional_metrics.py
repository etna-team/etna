import numpy as np
import numpy.testing as npt
import pytest

from etna.metrics import mae
from etna.metrics import mape
from etna.metrics import max_deviation
from etna.metrics import medae
from etna.metrics import mse
from etna.metrics import msle
from etna.metrics import r2_score
from etna.metrics import rmse
from etna.metrics import sign
from etna.metrics import smape
from etna.metrics import wape
from etna.metrics.functional_metrics import count_missing_values


@pytest.fixture()
def right_mae_value():
    pass


@pytest.fixture()
def y_true_1d():
    return [1, 3]


@pytest.fixture()
def y_pred_1d():
    return [2, 4]


@pytest.mark.parametrize(
    "metric, right_metrics_value",
    (
        (mae, 1),
        (mse, 1),
        (rmse, 1),
        (mape, 66 + 2 / 3),
        (smape, 47.6190476),
        (medae, 1),
        (r2_score, 0),
        (sign, -1),
        (max_deviation, 2),
        (wape, 1 / 2),
        (count_missing_values, 0),
    ),
)
def test_all_1d_metrics(metric, right_metrics_value, y_true_1d, y_pred_1d):
    npt.assert_almost_equal(metric(y_true_1d, y_pred_1d), right_metrics_value)


def test_mle_metric_exception(y_true_1d, y_pred_1d):
    y_true_1d[-1] = -1

    with pytest.raises(ValueError):
        msle(y_true_1d, y_pred_1d)


@pytest.mark.parametrize(
    "metric",
    (
        mse,
        mape,
        smape,
        sign,
        max_deviation,
        wape,
        count_missing_values,
    ),
)
def test_all_wrong_mode(metric, y_true_1d, y_pred_1d):
    with pytest.raises(NotImplementedError):
        metric(y_true_1d, y_pred_1d, multioutput="unknown")


@pytest.fixture()
def y_true_2d():
    return [[10, 1], [11, 2]]


@pytest.fixture()
def y_pred_2d():
    return [[11, 2], [10, 1]]


@pytest.mark.parametrize(
    "metric, right_metrics_value",
    (
        (mae, 1),
        (mse, 1),
        (rmse, 1),
        (mape, 42 + 3 / 11),
        (smape, 38.0952380),
        (medae, 1),
        (r2_score, -3),
        (sign, 0),
        (max_deviation, 2),
        (wape, 1 / 6),
        (count_missing_values, 0),
    ),
)
def test_all_2d_metrics_joint(metric, right_metrics_value, y_true_2d, y_pred_2d):
    npt.assert_almost_equal(metric(y_true_2d, y_pred_2d), right_metrics_value)


@pytest.mark.parametrize(
    "metric, params, right_metrics_value",
    (
        (mae, {"multioutput": "raw_values"}, [1, 1]),
        (mse, {"multioutput": "raw_values"}, [1, 1]),
        (rmse, {"multioutput": "raw_values"}, [1, 1]),
        (mape, {"multioutput": "raw_values"}, [9.5454545, 75]),
        (smape, {"multioutput": "raw_values"}, [9.5238095, 66 + 2 / 3]),
        (medae, {"multioutput": "raw_values"}, [1, 1]),
        (r2_score, {"multioutput": "raw_values"}, [-3, -3]),
        (sign, {"multioutput": "raw_values"}, [0, 0]),
        (max_deviation, {"multioutput": "raw_values"}, [1, 1]),
        (wape, {"multioutput": "raw_values"}, [0.0952381, 2 / 3]),
        (count_missing_values, {"multioutput": "raw_values"}, [0, 0]),
    ),
)
def test_all_2d_metrics_per_output(metric, params, right_metrics_value, y_true_2d, y_pred_2d):
    npt.assert_almost_equal(metric(y_true_2d, y_pred_2d, **params), right_metrics_value)


@pytest.mark.parametrize(
    "y_true, y_pred, multioutput, expected",
    [
        # 1d
        (np.array([1.0]), np.array([1.0]), "joint", 0.0),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", 2.0),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", 2.5),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", 2.5),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", 2.5),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, np.NaN]), "joint", 4.0),
        (np.array([1.0, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, 2.0]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([3.0, 1.0, 2.0]), "joint", np.NaN),
        (np.array([1.0, 2.0, 3.0]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        # 2d
        (np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T, np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T, "joint", 2.5),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            4.0,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            2.5,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            np.NaN,
        ),
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "raw_values",
            np.array([2.0, 3.0]),
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([4.0, 4.0]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, 2.5]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, np.NaN]),
        ),
    ],
)
def test_mse_ok(y_true, y_pred, multioutput, expected):
    result = mse(y_true=y_true, y_pred=y_pred, multioutput=multioutput)
    npt.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "y_true, y_pred, multioutput, expected",
    [
        # 1d
        (np.array([1.0]), np.array([1.0]), "joint", 0.0),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", 4 / 3),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", 1.5),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", 1.5),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", 1.5),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, np.NaN]), "joint", 2.0),
        (np.array([1.0, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, 2.0]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([3.0, 1.0, 2.0]), "joint", np.NaN),
        (np.array([1.0, 2.0, 3.0]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        # 2d
        (np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T, np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T, "joint", 1.5),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            2.0,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            1.5,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            np.NaN,
        ),
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "raw_values",
            np.array([4 / 3, 5 / 3]),
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([2.0, 2.0]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, 1.5]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, np.NaN]),
        ),
    ],
)
def test_mae_ok(y_true, y_pred, multioutput, expected):
    result = mae(y_true=y_true, y_pred=y_pred, multioutput=multioutput)
    npt.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "y_true, y_pred, multioutput, expected",
    [
        # 1d
        (np.array([1.0]), np.array([1.0]), "joint", 0.0),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", 0.0),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", 1.0),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", 0.0),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", 1.0),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, np.NaN]), "joint", 1.0),
        (np.array([1.0, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, 2.0]), "joint", 2.0),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([3.0, 1.0, 2.0]), "joint", 3.0),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, np.NaN]), "joint", 3.0),
        # 2d
        (np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T, np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T, "joint", 0.0),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "joint",
            2.0,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "joint",
            3.0,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "joint",
            6.0,
        ),
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "raw_values",
            np.array([0.0, 0.0]),
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "raw_values",
            np.array([1.0, 1.0]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "raw_values",
            np.array([3.0, 0.0]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "raw_values",
            np.array([3.0, 3.0]),
        ),
    ],
)
def test_count_missing_values_ok(y_true, y_pred, multioutput, expected):
    result = count_missing_values(y_true=y_true, y_pred=y_pred, multioutput=multioutput)
    npt.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "y_true, y_pred, multioutput, expected",
    [
        # 1d
        (np.array([1.0]), np.array([1.0]), "joint", 0.0),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", 17.0 / 18.0 * 100),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", 7.0 / 6.0 * 100),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", 7.0 / 6.0 * 100),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", 7.0 / 6.0 * 100),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, np.NaN]), "joint", 2.0 * 100),
        (np.array([1.0, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, 2.0]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([3.0, 1.0, 2.0]), "joint", np.NaN),
        (np.array([1.0, 2.0, 3.0]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        # 2d
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "joint",
            21.0 / 30.0 * 100,
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            8.0 / 6.0 * 100,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            13.0 / 30.0 * 100,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            np.NaN,
        ),
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "raw_values",
            np.array([17.0 / 18.0 * 100, 41.0 / 90.0 * 100]),
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([2.0 * 100, 2.0 / 3.0 * 100]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, 13.0 / 30.0 * 100]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, np.NaN]),
        ),
    ],
)
def test_mape_ok(y_true, y_pred, multioutput, expected):
    result = mape(y_true=y_true, y_pred=y_pred, multioutput=multioutput)
    npt.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "y_true, y_pred, multioutput, expected",
    [
        # 1d
        (np.array([1.0]), np.array([1.0]), "joint", 0.0),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", 31.0 / 45.0 * 100),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", 7.0 / 10.0 * 100),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", 7.0 / 10.0 * 100),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", 7.0 / 10.0 * 100),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, np.NaN]), "joint", 1.0 * 100),
        (np.array([1.0, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, 2.0]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([3.0, 1.0, 2.0]), "joint", np.NaN),
        (np.array([1.0, 2.0, 3.0]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        # 2d
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "joint",
            311.0 / 540.0 * 100,
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            3.0 / 4.0 * 100,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            13.0 / 36.0 * 100,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            np.NaN,
        ),
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "raw_values",
            np.array([31.0 / 45.0 * 100, 25.0 / 54.0 * 100]),
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([1.0 * 100, 1.0 / 2.0 * 100]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, 13.0 / 36.0 * 100]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, np.NaN]),
        ),
    ],
)
def test_smape_ok(y_true, y_pred, multioutput, expected):
    result = smape(y_true=y_true, y_pred=y_pred, multioutput=multioutput)
    npt.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "y_true, y_pred, multioutput, expected",
    [
        # 1d
        (np.array([1.0]), np.array([1.0]), "joint", 0.0),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", 1 / 3),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", 0.0),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", 0.0),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", 0.0),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, np.NaN]), "joint", -1.0),
        (np.array([1.0, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, 2.0]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([3.0, 1.0, 2.0]), "joint", np.NaN),
        (np.array([1.0, 2.0, 3.0]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        # 2d
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "joint",
            1 / 3,
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            -1.0,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            0.0,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            np.NaN,
        ),
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "raw_values",
            np.array([1 / 3, 1 / 3]),
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([-1.0, -1.0]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, 0.0]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, np.NaN]),
        ),
    ],
)
def test_sign_ok(y_true, y_pred, multioutput, expected):
    result = sign(y_true=y_true, y_pred=y_pred, multioutput=multioutput)
    npt.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "y_true, y_pred, multioutput, expected",
    [
        # 1d
        (np.array([1.0]), np.array([1.0]), "joint", 0.0),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", 4 / 6),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", 3 / 4),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", 3 / 4),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", 3 / 4),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, np.NaN]), "joint", 2 / 1),
        (np.array([1.0, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, 2.0]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([3.0, 1.0, 2.0]), "joint", np.NaN),
        (np.array([1.0, 2.0, 3.0]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        # 2d
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "joint",
            9 / 18,
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            4 / 4,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            3 / 8,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            np.NaN,
        ),
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "raw_values",
            np.array([4 / 6, 5 / 12]),
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([2 / 1, 2 / 3]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, 3 / 8]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, np.NaN]),
        ),
    ],
)
def test_wape_ok(y_true, y_pred, multioutput, expected):
    result = wape(y_true=y_true, y_pred=y_pred, multioutput=multioutput)
    npt.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "y_true, y_pred, multioutput, expected",
    [
        # 1d
        (np.array([1.0]), np.array([1.0]), "joint", 0.0),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", 2.0),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", 2.0),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", 2.0),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", 2.0),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, np.NaN]), "joint", 2.0),
        (np.array([1.0, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, 2.0]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([3.0, 1.0, 2.0]), "joint", np.NaN),
        (np.array([1.0, 2.0, 3.0]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        # 2d
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "joint",
            4.0,
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            4.0,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            2.0,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            np.NaN,
        ),
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "raw_values",
            np.array([2.0, 2.0]),
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([2.0, 2.0]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, 2.0]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, np.NaN]),
        ),
    ],
)
def test_max_deviation(y_true, y_pred, multioutput, expected):
    result = max_deviation(y_true=y_true, y_pred=y_pred, multioutput=multioutput)
    npt.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "y_true, y_pred, multioutput, expected",
    [
        # 1d
        (np.array([1.0]), np.array([1.0]), "joint", 0.0),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", np.sqrt(2.0)),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, 2.0]), "joint", np.sqrt(2.5)),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", np.sqrt(2.5)),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, np.NaN, 2.0]), "joint", np.sqrt(2.5)),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, np.NaN]), "joint", 2.0),
        (np.array([1.0, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, 2.0]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([3.0, 1.0, 2.0]), "joint", np.NaN),
        (np.array([1.0, 2.0, 3.0]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        # 2d
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "joint",
            np.sqrt(2.5),
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            2.0,
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            np.sqrt(2.5),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            np.NaN,
        ),
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "raw_values",
            np.sqrt(np.array([2.0, 3.0])),
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([2.0, 2.0]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.sqrt(np.array([np.NaN, 2.5])),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, np.NaN]),
        ),
    ],
)
def test_mse_ok(y_true, y_pred, multioutput, expected):
    result = rmse(y_true=y_true, y_pred=y_pred, multioutput=multioutput)
    npt.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "y_true, y_pred, multioutput, expected",
    [
        # 1d
        (np.array([1.0]), np.array([1.0]), "joint", 0.0),
        (
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 1.0, 2.0]),
            "joint",
            1 / 3 * (np.log(1 / 2) ** 2 + np.log(3 / 2) ** 2 + np.log(4 / 3) ** 2),
        ),
        (
            np.array([1.0, np.NaN, 3.0]),
            np.array([3.0, 1.0, 2.0]),
            "joint",
            0.5 * (np.log(1 / 2) ** 2 + np.log(4 / 3) ** 2),
        ),
        (
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, np.NaN, 2.0]),
            "joint",
            0.5 * (np.log(1 / 2) ** 2 + np.log(4 / 3) ** 2),
        ),
        (
            np.array([1.0, np.NaN, 3.0]),
            np.array([3.0, np.NaN, 2.0]),
            "joint",
            0.5 * (np.log(1 / 2) ** 2 + np.log(4 / 3) ** 2),
        ),
        (np.array([1.0, np.NaN, 3.0]), np.array([3.0, 1.0, np.NaN]), "joint", (np.log1p(1.0) - np.log1p(3.0)) ** 2),
        (np.array([1.0, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, 2.0]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([3.0, 1.0, 2.0]), "joint", np.NaN),
        (np.array([1.0, 2.0, 3.0]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([np.NaN, np.NaN, np.NaN]), "joint", np.NaN),
        # 2d
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "joint",
            1
            / 6
            * (
                np.log(1 / 2) ** 2
                + np.log(3 / 2) ** 2
                + np.log(4 / 3) ** 2
                + np.log(2 / 3) ** 2
                + np.log(5 / 3) ** 2
                + np.log(6 / 5) ** 2
            ),
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            0.5 * (np.log(1 / 2) ** 2 + np.log(2 / 3) ** 2),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            0.5 * (np.log(2 / 3) ** 2 + np.log(6 / 5) ** 2),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "joint",
            np.NaN,
        ),
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, 2.0], [5.0, 2.0, 4.0]]).T,
            "raw_values",
            np.array(
                [
                    1 / 3 * (np.log(1 / 2) ** 2 + np.log(3 / 2) ** 2 + np.log(4 / 3) ** 2),
                    1 / 3 * (np.log(2 / 3) ** 2 + np.log(5 / 3) ** 2 + np.log(6 / 5) ** 2),
                ]
            ),
        ),
        (
            np.array([[1.0, np.NaN, 3.0], [3.0, 4.0, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.log(1 / 2) ** 2, np.log(2 / 3) ** 2]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, 0.5 * (np.log(2 / 3) ** 2 + np.log(6 / 5) ** 2)]),
        ),
        (
            np.array([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]]).T,
            np.array([[3.0, 1.0, np.NaN], [5.0, np.NaN, 4.0]]).T,
            "raw_values",
            np.array([np.NaN, np.NaN]),
        ),
    ],
)
def test_msle_ok(y_true, y_pred, multioutput, expected):
    result = msle(y_true=y_true, y_pred=y_pred, multioutput=multioutput)
    npt.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "y_true, y_pred, multioutput",
    [
        # 1d
        (np.array([1.0, 2.0, -3.0]), np.array([3.0, 2.0, 1.0]), "joint"),
        (np.array([1.0, 2.0, 3.0]), np.array([-3.0, 2.0, 1.0]), "joint"),
        (np.array([1.0, -2.0, 3.0]), np.array([3.0, 2.0, -1.0]), "joint"),
        # 2d
        (np.array([[1.0, 2.0, -3.0], [3.0, 4.0, 5.0]]).T, np.array([[3.0, 2.0, 1.0], [5.0, 4.0, 3.0]]).T, "joint"),
        (np.array([[1.0, 2.0, 3.0], [-3.0, 4.0, 5.0]]).T, np.array([[3.0, 2.0, 1.0], [5.0, 4.0, 3.0]]).T, "joint"),
        (np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T, np.array([[3.0, 2.0, -1.0], [5.0, 4.0, 3.0]]).T, "joint"),
        (np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T, np.array([[3.0, 2.0, 1.0], [-5.0, 4.0, 3.0]]).T, "joint"),
        (np.array([[1.0, 2.0, -3.0], [3.0, 4.0, -5.0]]).T, np.array([[3.0, -2.0, 1.0], [5.0, 4.0, -3.0]]).T, "joint"),
        (
            np.array([[1.0, 2.0, -3.0], [3.0, 4.0, -5.0]]).T,
            np.array([[3.0, 2.0, 1.0], [5.0, 4.0, 3.0]]).T,
            "raw_values",
        ),
        (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]).T,
            np.array([[3.0, -2.0, 1.0], [-5.0, 4.0, 3.0]]).T,
            "raw_values",
        ),
    ],
)
def test_msle_negative(y_true, y_pred, multioutput):
    with pytest.raises(ValueError):
        msle(y_true=y_true, y_pred=y_pred, multioutput=multioutput)
