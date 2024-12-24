import warnings
from enum import Enum
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from typing_extensions import assert_never

ArrayLike = Union[float, Sequence[float], Sequence[Sequence[float]]]


class FunctionalMetricMultioutput(str, Enum):
    """Enum for different functional metric multioutput modes."""

    #: Compute one scalar value taking into account all outputs.
    joint = "joint"

    #: Compute one value per each output.
    raw_values = "raw_values"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} options allowed"
        )


def _get_axis_by_multioutput(multioutput: str) -> Optional[int]:
    multioutput_enum = FunctionalMetricMultioutput(multioutput)
    if multioutput_enum is FunctionalMetricMultioutput.joint:
        return None
    elif multioutput_enum is FunctionalMetricMultioutput.raw_values:
        return 0
    else:
        assert_never(multioutput_enum)


def mse(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "joint") -> ArrayLike:
    """Mean squared error with missing values handling.

    .. math::
        MSE(y\_true, y\_pred) = \\frac{\\sum_{i=1}^{n}{(y\_true_i - y\_pred_i)^2}}{n}

    The nans are ignored during computation. If all values are nans, the result is NaN.

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A non-negative floating point value (the best value is 0.0), or an array of floating point values,
        one for each individual target.

    Raises
    ------
    :
    ValueError:
        If the shapes of the input arrays do not match.
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    axis = _get_axis_by_multioutput(multioutput)
    with warnings.catch_warnings():
        # this helps to prevent warning in case of all nans
        warnings.filterwarnings(
            message="Mean of empty slice",
            action="ignore",
        )
        result = np.nanmean((y_true_array - y_pred_array) ** 2, axis=axis)
    return result


def mae(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "joint") -> ArrayLike:
    """Mean absolute error with missing values handling.

    .. math::
        MAE(y\_true, y\_pred) = \\frac{\\sum_{i=1}^{n}{\\mid y\_true_i - y\_pred_i \\mid}}{n}

    The nans are ignored during computation. If all values are nans, the result is NaN.

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A non-negative floating point value (the best value is 0.0), or an array of floating point values,
        one for each individual target.

    Raises
    ------
    :
    ValueError:
        If the shapes of the input arrays do not match.
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    axis = _get_axis_by_multioutput(multioutput)
    with warnings.catch_warnings():
        # this helps to prevent warning in case of all nans
        warnings.filterwarnings(
            message="Mean of empty slice",
            action="ignore",
        )
        result = np.nanmean(np.abs(y_true_array - y_pred_array), axis=axis)
    return result


def mape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-15, multioutput: str = "joint") -> ArrayLike:
    """Mean absolute percentage error with missing values handling.

    .. math::
        MAPE(y\_true, y\_pred) = \\frac{1}{n} \\cdot \\sum_{i=1}^{n} \\frac{\\mid y\_true_i - y\_pred_i\\mid}{\\mid y\_true_i \\mid + \epsilon}

    `Scale-dependent errors <https://otexts.com/fpp3/accuracy.html#scale-dependent-errors>`_

    The nans are ignored during computation. If all values are nans, the result is NaN.

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    eps:
        MAPE is undefined for ``y_true[i]==0`` for any ``i``, so all zeros ``y_true[i]`` are
        clipped to ``max(eps, abs(y_true))``.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A non-negative floating point value (the best value is 0.0), or an array of floating point values,
        one for each individual target.

    Raises
    ------
    :
    ValueError:
        If the shapes of the input arrays do not match.
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    y_true_array = y_true_array.clip(eps)

    axis = _get_axis_by_multioutput(multioutput)

    with warnings.catch_warnings():
        # this helps to prevent warning in case of all nans
        warnings.filterwarnings(
            message="Mean of empty slice",
            action="ignore",
        )
        result = np.nanmean(np.abs((y_true_array - y_pred_array) / y_true_array), axis=axis) * 100

    return result


def smape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-15, multioutput: str = "joint") -> ArrayLike:
    """Symmetric mean absolute percentage error with missing values handling.

    .. math::
        SMAPE(y\_true, y\_pred) = \\frac{2 \\cdot 100 \\%}{n} \\cdot \\sum_{i=1}^{n} \\frac{\\mid y\_true_i - y\_pred_i\\mid}{\\mid y\_true_i \\mid + \\mid y\_pred_i \\mid}

    The nans are ignored during computation. If all values are nans, the result is NaN.

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    eps: float=1e-15
        SMAPE is undefined for ``y_true[i] + y_pred[i] == 0`` for any ``i``, so all zeros ``y_true[i] + y_pred[i]`` are
        clipped to ``max(eps, abs(y_true) + abs(y_pred))``.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A non-negative floating point value (the best value is 0.0), or an array of floating point values,
        one for each individual target.

    Raises
    ------
    :
    ValueError:
        If the shapes of the input arrays do not match.
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    axis = _get_axis_by_multioutput(multioutput)

    with warnings.catch_warnings():
        # this helps to prevent warning in case of all nans
        warnings.filterwarnings(
            message="Mean of empty slice",
            action="ignore",
        )
        result = 100 * np.nanmean(
            2 * np.abs(y_pred_array - y_true_array) / (np.abs(y_true_array) + np.abs(y_pred_array)).clip(eps), axis=axis
        )

    return result


def r2_score(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "joint") -> ArrayLike:
    """Coefficient of determination metric.

    .. math::
        R^2(y\_true, y\_pred) = 1 - \\frac{\\sum_{i=1}^{n}{(y\_true_i - y\_pred_i)^2}}{\\sum_{i=1}^{n}{(y\_true_i - \\overline{y\_true})^2}}

    The nans are ignored during computation. If all values are nans, the result is NaN.

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A floating point value, or an array of floating point values,
        one for each individual target.

    Raises
    ------
    :
    ValueError:
        If the shapes of the input arrays do not match.
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    axis = _get_axis_by_multioutput(multioutput)
    not_nan = ~np.isnan(y_true_array - y_pred_array)
    with warnings.catch_warnings():
        # this helps to prevent warning in case of all nans
        warnings.filterwarnings(
            message="invalid value encountered in scalar divide",
            action="ignore",
        )
        warnings.filterwarnings(
            message="invalid value encountered in divide",
            action="ignore",
        )
        warnings.filterwarnings(
            message="Degrees of freedom <= 0 for slice",
            action="ignore",
        )

        numerator = np.asarray(mse(y_true=y_true, y_pred=y_pred, multioutput=multioutput))
        y_true_array = y_true_array.astype(float)  # otherwise we can't assign NaN to it
        y_true_array[~not_nan] = np.NaN
        denominator = np.asarray(np.nanvar(y_true_array, axis=axis))
        nonzero_numerator = np.asarray(numerator != 0)
        nonzero_denominator = np.asarray(denominator != 0)

        result = np.ones_like(numerator, dtype=float)
        valid_score = nonzero_denominator & nonzero_numerator
        # if numerator and denominator aren't zero, then just compute r2_score
        result[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
        # if numerator is non-zero, the answer is 0.0, otherwise (getting 0/0) the answer is 1.0
        result[nonzero_numerator & ~nonzero_denominator] = 0.0

        # if there are less than 2 values, result is NaN
        num_not_nans = np.sum(not_nan, axis=axis)
        result = np.where(num_not_nans < 2, np.NaN, result)

        if multioutput is FunctionalMetricMultioutput.joint:
            return result.item()
        else:
            return result  # type: ignore


def medae(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "joint") -> ArrayLike:
    """Median absolute error metric.

    .. math::
       MedAE(y\_true, y\_pred) = median(\\mid y\_true_1 - y\_pred_1 \\mid, \\cdots, \\mid y\_true_n - y\_pred_n \\mid)

    The nans are ignored during computation. If all values are nans, the result is NaN.

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A non-negative floating point value (the best value is 0.0), or an array of floating point values,
        one for each individual target.

    Raises
    ------
    :
    ValueError:
        If the shapes of the input arrays do not match.
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    axis = _get_axis_by_multioutput(multioutput)
    with warnings.catch_warnings():
        # this helps to prevent warning in case of all nans
        warnings.filterwarnings(
            message="All-NaN slice encountered",
            action="ignore",
        )
        result = np.nanmedian(np.abs(y_true_array - y_pred_array), axis=axis)
    return result


def sign(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "joint") -> ArrayLike:
    """Sign error metric.

    .. math::
        Sign(y\_true, y\_pred) = \\frac{1}{n}\\cdot\\sum_{i=1}^{n}{sign(y\_true_i - y\_pred_i)}

    The nans are ignored during computation. If all values are nans, the result is NaN.

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A floating point value, or an array of floating point values,
        one for each individual target.

    Raises
    ------
    :
    ValueError:
        If the shapes of the input arrays do not match.
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    axis = _get_axis_by_multioutput(multioutput)
    with warnings.catch_warnings():
        # this helps to prevent warning in case of all nans
        warnings.filterwarnings(
            message="Mean of empty slice",
            action="ignore",
        )
        result = np.nanmean(np.sign(y_true_array - y_pred_array), axis=axis)

    return result


def max_deviation(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "joint") -> ArrayLike:
    """Max Deviation metric.

    .. math::
        MaxDeviation(y\_true, y\_pred) = \\max_{1 \\le j \\le n} | y_j |, where \\, y_j = \\sum_{i=1}^{j}{y\_pred_i - y\_true_i}

    The nans are ignored during computation. If all values are nans, the result is NaN.

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A non-negative floating point value (the best value is 0.0), or an array of floating point values,
        one for each individual target.

    Raises
    ------
    :
    ValueError:
        If the shapes of the input arrays do not match.
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    axis = _get_axis_by_multioutput(multioutput)
    diff = y_pred_array - y_true_array
    prefix_error_sum = np.nancumsum(diff, axis=axis)
    isnan = np.all(np.isnan(diff), axis=axis)
    result = np.max(np.abs(prefix_error_sum), axis=axis)
    result = np.where(isnan, np.NaN, result)
    if multioutput is FunctionalMetricMultioutput.joint:
        return result.item()
    else:
        return result


def rmse(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "joint") -> ArrayLike:
    """Root mean squared error with missing values handling.

    .. math::
        RMSE(y\_true, y\_pred) = \\sqrt\\frac{\\sum_{i=1}^{n}{(y\_true_i - y\_pred_i)^2}}{n}

    The nans are ignored during computation. If all values are nans, the result is NaN.

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A non-negative floating point value (the best value is 0.0), or an array of floating point values,
        one for each individual target.

    Raises
    ------
    :
    ValueError:
        If the shapes of the input arrays do not match.
    """
    mse_result = mse(y_true=y_true, y_pred=y_pred, multioutput=multioutput)
    result = np.sqrt(mse_result)

    return result  # type: ignore


def msle(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "joint") -> ArrayLike:
    """Mean squared logarithmic error with missing values handling.

    .. math::
        MSLE(y\_true, y\_pred) = \\frac{1}{n}\\cdot\\sum_{i=1}^{n}{(log(1 + y\_true_i) - log(1 + y\_pred_i))^2}

    The nans are ignored during computation. If all values are nans, the result is NaN.

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A non-negative floating point value (the best value is 0.0), or an array of floating point values,
        one for each individual target.

    Raises
    ------
    :
    ValueError:
        If the shapes of the input arrays do not match.
    ValueError:
        If input arrays contain negative values.
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    if (y_true_array < 0).any() or (y_pred_array < 0).any():
        raise ValueError("Mean squared logarithmic error cannot be used when targets contain negative values.")

    axis = _get_axis_by_multioutput(multioutput)

    with warnings.catch_warnings():
        # this helps to prevent warning in case of all nans
        warnings.filterwarnings(
            message="Mean of empty slice",
            action="ignore",
        )
        result = np.nanmean((np.log1p(y_true_array) - np.log1p(y_pred_array)) ** 2, axis=axis)

    return result


def wape(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "joint") -> ArrayLike:
    """Weighted average percentage Error metric.

    .. math::
        WAPE(y\_true, y\_pred) = \\frac{\\sum_{i=1}^{n} |y\_true_i - y\_pred_i|}{\\sum_{i=1}^{n}|y\\_true_i|}

    The nans are ignored during computation. If all values are nans, the result is NaN.

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A non-negative floating point value (the best value is 0.0), or an array of floating point values,
        one for each individual target.

    Raises
    ------
    :
    ValueError:
        If the shapes of the input arrays do not match.
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    axis = _get_axis_by_multioutput(multioutput)
    diff = y_true_array - y_pred_array
    numerator = np.nansum(np.abs(diff), axis=axis)
    isnan = np.isnan(diff)
    denominator = np.nansum(np.abs(y_true_array * (~isnan)), axis=axis)
    with warnings.catch_warnings():
        # this helps to prevent warning in case of all nans
        warnings.filterwarnings(
            message="invalid value encountered in scalar divide",
            action="ignore",
        )
        warnings.filterwarnings(
            message="invalid value encountered in divide",
            action="ignore",
        )
        warnings.filterwarnings(
            message="divide by zero encountered in scalar divide",
            action="ignore",
        )
        warnings.filterwarnings(
            message="divide by zero encountered in divide",
            action="ignore",
        )
        isnan = np.all(isnan, axis=axis)
        result = np.where(denominator == 0, np.NaN, numerator / denominator)
        result = np.where(isnan, np.NaN, result)
        if multioutput is FunctionalMetricMultioutput.joint:
            return result.item()
        else:
            return result  # type: ignore


def count_missing_values(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "joint") -> ArrayLike:
    """Count missing values in ``y_true``.

    .. math::
        MissingCounter(y\_true, y\_pred) = \\sum_{i=1}^{n}{isnan(y\_true_i)}

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A floating point value, or an array of floating point values,
        one for each individual target.

    Raises
    ------
    :
    ValueError:
        If the shapes of the input arrays do not match.
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    axis = _get_axis_by_multioutput(multioutput)

    return np.sum(np.isnan(y_true), axis=axis).astype(float)


__all__ = [
    "mae",
    "mse",
    "msle",
    "medae",
    "r2_score",
    "mape",
    "smape",
    "sign",
    "max_deviation",
    "rmse",
    "wape",
    "count_missing_values",
]
