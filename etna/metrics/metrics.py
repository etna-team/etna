from functools import partial

from etna.metrics.base import MetricWithMissingHandling
from etna.metrics.functional_metrics import count_missing_values
from etna.metrics.functional_metrics import mae
from etna.metrics.functional_metrics import mape
from etna.metrics.functional_metrics import max_deviation
from etna.metrics.functional_metrics import medae
from etna.metrics.functional_metrics import mse
from etna.metrics.functional_metrics import msle
from etna.metrics.functional_metrics import r2_score
from etna.metrics.functional_metrics import rmse
from etna.metrics.functional_metrics import sign
from etna.metrics.functional_metrics import smape
from etna.metrics.functional_metrics import wape


class MAE(MetricWithMissingHandling):
    """Mean absolute error metric with multi-segment computation support.

    .. math::
        MAE(y\_true, y\_pred) = \\frac{\\sum_{i=1}^{n}{\\mid y\_true_i - y\_pred_i \\mid}}{n}

    This metric can handle missing values with parameter ``missing_mode``.
    If there are too many of them in ``ignore`` mode, the result will be ``None``.

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = "per-segment", missing_mode: str = "error", **kwargs):
        """Init metric.

        Parameters
        ----------
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.

        missing_mode:
            mode of handling missing values (see :py:class:`~etna.metrics.base.MetricMissingMode`)
        kwargs:
            metric's computation arguments
        """
        mae_per_output = partial(mae, multioutput="raw_values")
        super().__init__(
            mode=mode,
            metric_fn=mae_per_output,
            metric_fn_signature="matrix_to_array",
            missing_mode=missing_mode,
            **kwargs,
        )

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


class MSE(MetricWithMissingHandling):
    """Mean squared error metric with multi-segment computation support.

    .. math::
        MSE(y\_true, y\_pred) = \\frac{\\sum_{i=1}^{n}{(y\_true_i - y\_pred_i)^2}}{n}

    This metric can handle missing values with parameter ``missing_mode``.
    If there are too many of them in ``ignore`` mode, the result will be ``None``.

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = "per-segment", missing_mode: str = "error", **kwargs):
        """Init metric.

        Parameters
        ----------
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.

        missing_mode:
            mode of handling missing values (see :py:class:`~etna.metrics.base.MetricMissingMode`)
        kwargs:
            metric's computation arguments
        """
        mse_per_output = partial(mse, multioutput="raw_values")
        super().__init__(
            mode=mode,
            metric_fn=mse_per_output,
            metric_fn_signature="matrix_to_array",
            missing_mode=missing_mode,
            **kwargs,
        )

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


class RMSE(MetricWithMissingHandling):
    """Root mean squared error metric with multi-segment computation support.

    .. math::
        RMSE(y\_true, y\_pred) = \\sqrt\\frac{\\sum_{i=1}^{n}{(y\_true_i - y\_pred_i)^2}}{n}

    This metric can handle missing values with parameter ``missing_mode``.
    If there are too many of them in ``ignore`` mode, the result will be ``None``.

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = "per-segment", missing_mode="error", **kwargs):
        """Init metric.

        Parameters
        ----------
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.

        missing_mode:
            mode of handling missing values (see :py:class:`~etna.metrics.base.MetricMissingMode`)
        kwargs:
            metric's computation arguments
        """
        rmse_per_output = partial(rmse, multioutput="raw_values")
        super().__init__(
            mode=mode,
            metric_fn=rmse_per_output,
            metric_fn_signature="matrix_to_array",
            missing_mode=missing_mode,
            **kwargs,
        )

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


class R2(MetricWithMissingHandling):
    """Coefficient of determination metric with multi-segment computation support.

    .. math::
        R^2(y\_true, y\_pred) = 1 - \\frac{\\sum_{i=1}^{n}{(y\_true_i - y\_pred_i)^2}}{\\sum_{i=1}^{n}{(y\_true_i - \\overline{y\_true})^2}}

    This metric can handle missing values with parameter ``missing_mode``.
    If there are too many of them in ``ignore`` mode, the result will be ``None``.

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = "per-segment", missing_mode: str = "error", **kwargs):
        """Init metric.

        Parameters
        ----------
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.
        missing_mode:
            mode of handling missing values (see :py:class:`~etna.metrics.base.MetricMissingMode`)
        kwargs:
            metric's computation arguments
        """
        r2_per_output = partial(r2_score, multioutput="raw_values")
        super().__init__(
            mode=mode,
            metric_fn=r2_per_output,
            metric_fn_signature="matrix_to_array",
            missing_mode=missing_mode,
            **kwargs,
        )

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return True


class MAPE(MetricWithMissingHandling):
    """Mean absolute percentage error metric with multi-segment computation support.

    .. math::
        MAPE(y\_true, y\_pred) = \\frac{1}{n} \\cdot \\sum_{i=1}^{n} \\frac{\\mid y\_true_i - y\_pred_i\\mid}{\\mid y\_true_i \\mid + \epsilon}

    This metric can handle missing values with parameter ``missing_mode``.
    If there are too many of them in ``ignore`` mode, the result will be ``None``.

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = "per-segment", missing_mode: str = "error", **kwargs):
        """Init metric.

        Parameters
        ----------
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.
        missing_mode:
            mode of handling missing values (see :py:class:`~etna.metrics.base.MetricMissingMode`)
        kwargs:
            metric's computation arguments
        """
        mape_per_output = partial(mape, multioutput="raw_values")
        super().__init__(
            mode=mode,
            metric_fn=mape_per_output,
            metric_fn_signature="matrix_to_array",
            missing_mode=missing_mode,
            **kwargs,
        )

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


class SMAPE(MetricWithMissingHandling):
    """Symmetric mean absolute percentage error metric with multi-segment computation support.

    .. math::
        SMAPE(y\_true, y\_pred) = \\frac{2 \\cdot 100 \\%}{n} \\cdot \\sum_{i=1}^{n} \\frac{\\mid y\_true_i - y\_pred_i\\mid}{\\mid y\_true_i \\mid + \\mid y\_pred_i \\mid}

    This metric can handle missing values with parameter ``missing_mode``.
    If there are too many of them in ``ignore`` mode, the result will be ``None``.

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = "per-segment", missing_mode: str = "error", **kwargs):
        """Init metric.

        Parameters
        ----------
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.
        missing_mode:
            mode of handling missing values (see :py:class:`~etna.metrics.base.MetricMissingMode`)
        kwargs:
            metric's computation arguments
        """
        smape_per_output = partial(smape, multioutput="raw_values")
        super().__init__(
            mode=mode,
            metric_fn=smape_per_output,
            metric_fn_signature="matrix_to_array",
            missing_mode=missing_mode,
            **kwargs,
        )

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


class MedAE(MetricWithMissingHandling):
    """Median absolute error metric with multi-segment computation support.

    .. math::
       MedAE(y\_true, y\_pred) = median(\\mid y\_true_1 - y\_pred_1 \\mid, \\cdots, \\mid y\_true_n - y\_pred_n \\mid)

    This metric can handle missing values with parameter ``missing_mode``.
    If there are too many of them in ``ignore`` mode, the result will be ``None``.

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = "per-segment", missing_mode: str = "error", **kwargs):
        """Init metric.

        Parameters
        ----------
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.
        missing_mode:
            mode of handling missing values (see :py:class:`~etna.metrics.base.MetricMissingMode`)
        kwargs:
            metric's computation arguments
        """
        medae_per_output = partial(medae, multioutput="raw_values")
        super().__init__(
            mode=mode,
            metric_fn=medae_per_output,
            metric_fn_signature="matrix_to_array",
            missing_mode=missing_mode,
            **kwargs,
        )

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


class MSLE(MetricWithMissingHandling):
    """Mean squared logarithmic error metric with multi-segment computation support.

    .. math::
        MSLE(y\_true, y\_pred) = \\frac{1}{n}\\cdot\\sum_{i=1}^{n}{(ln(1 + y\_true_i) - ln(1 + y\_pred_i))^2}

    This metric can handle missing values with parameter ``missing_mode``.
    If there are too many of them in ``ignore`` mode, the result will be ``None``.

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = "per-segment", missing_mode="error", **kwargs):
        """Init metric.

        Parameters
        ----------
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.

        missing_mode:
            mode of handling missing values (see :py:class:`~etna.metrics.base.MetricMissingMode`)
        kwargs:
            metric's computation arguments

        """
        msle_per_output = partial(msle, multioutput="raw_values")
        super().__init__(
            mode=mode,
            metric_fn=msle_per_output,
            metric_fn_signature="matrix_to_array",
            missing_mode=missing_mode,
            **kwargs,
        )

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


class Sign(MetricWithMissingHandling):
    """Sign error metric with multi-segment computation support.

    .. math::
        Sign(y\_true, y\_pred) = \\frac{1}{n}\\cdot\\sum_{i=1}^{n}{sign(y\_true_i - y\_pred_i)}

    This metric can handle missing values with parameter ``missing_mode``.
    If there are too many of them in ``ignore`` mode, the result will be ``None``.

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = "per-segment", missing_mode: str = "error", **kwargs):
        """Init metric.

        Parameters
        ----------
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.
        missing_mode:
            mode of handling missing values (see :py:class:`~etna.metrics.base.MetricMissingMode`)
        kwargs:
            metric's computation arguments
        """
        sign_per_output = partial(sign, multioutput="raw_values")
        super().__init__(
            mode=mode,
            metric_fn=sign_per_output,
            metric_fn_signature="matrix_to_array",
            missing_mode=missing_mode,
            **kwargs,
        )

    @property
    def greater_is_better(self) -> None:
        """Whether higher metric value is better."""
        return None


class MaxDeviation(MetricWithMissingHandling):
    """Max Deviation metric with multi-segment computation support (maximum deviation value of cumulative sums).

    .. math::
        MaxDeviation(y\_true, y\_pred) = \\max_{1 \\le j \\le n} | y_j |, where \\, y_j = \\sum_{i=1}^{j}{y\_pred_i - y\_true_i}

    This metric can handle missing values with parameter ``missing_mode``.
    If there are too many of them in ``ignore`` mode, the result will be ``None``.

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = "per-segment", missing_mode: str = "error", **kwargs):
        """Init metric.

        Parameters
        ----------
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.
        missing_mode:
            mode of handling missing values (see :py:class:`~etna.metrics.base.MetricMissingMode`)
        kwargs:
            metric's computation arguments
        """
        max_deviation_per_output = partial(max_deviation, multioutput="raw_values")
        super().__init__(
            mode=mode,
            metric_fn=max_deviation_per_output,
            metric_fn_signature="matrix_to_array",
            missing_mode=missing_mode,
            **kwargs,
        )

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


class WAPE(MetricWithMissingHandling):
    """Weighted average percentage Error metric with multi-segment computation support.

    .. math::
        WAPE(y\_true, y\_pred) = \\frac{\\sum_{i=1}^{n} |y\_true_i - y\_pred_i|}{\\sum_{i=1}^{n}|y\\_true_i|}

    This metric can handle missing values with parameter ``missing_mode``.
    If there are too many of them in ``ignore`` mode, the result will be ``None``.

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = "per-segment", missing_mode: str = "error", **kwargs):
        """Init metric.

        Parameters
        ----------
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.
        missing_mode:
            mode of handling missing values (see :py:class:`~etna.metrics.base.MetricMissingMode`)
        kwargs:
            metric's computation arguments
        """
        wape_per_output = partial(wape, multioutput="raw_values")
        super().__init__(
            mode=mode,
            metric_fn=wape_per_output,
            metric_fn_signature="matrix_to_array",
            missing_mode=missing_mode,
            **kwargs,
        )

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


class MissingCounter(MetricWithMissingHandling):
    """Missing values counter with multi-segment computation support.

    .. math::
        MissingCounter(y\_true, y\_pred) = \\sum_{i=1}^{n}{isnan(y\_true_i)}

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = "per-segment", **kwargs):
        """Init metric.

        Parameters
        ----------
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

            See :py:class:`~etna.metrics.base.MetricAggregationMode`.
        kwargs:
            metric's computation arguments
        """
        count_missing_values_per_output = partial(count_missing_values, multioutput="raw_values")
        super().__init__(
            mode=mode,
            metric_fn=count_missing_values_per_output,
            metric_fn_signature="matrix_to_array",
            missing_mode="ignore",
            **kwargs,
        )

    @property
    def greater_is_better(self) -> None:
        """Whether higher metric value is better."""
        return None


__all__ = [
    "MAE",
    "MSE",
    "RMSE",
    "R2",
    "MSLE",
    "MAPE",
    "SMAPE",
    "MedAE",
    "Sign",
    "MaxDeviation",
    "WAPE",
    "MissingCounter",
]
