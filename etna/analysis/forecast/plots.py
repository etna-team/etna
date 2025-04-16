import itertools
import math
from copy import deepcopy
from enum import Enum
from functools import cmp_to_key
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.graphics.gofplots import qqplot
from typing_extensions import Literal

from etna.analysis.forecast.utils import _check_metrics_df_empty_segments
from etna.analysis.forecast.utils import _check_metrics_df_same_folds_for_each_segment
from etna.analysis.forecast.utils import _prepare_forecast_results
from etna.analysis.forecast.utils import _select_prediction_intervals_names
from etna.analysis.forecast.utils import _validate_intersecting_folds
from etna.analysis.forecast.utils import get_residuals
from etna.analysis.utils import _prepare_axes
from etna.datasets.utils import timestamp_range

if TYPE_CHECKING:
    from etna.datasets import TSDataset
    from etna.transforms import Transform


def _get_borders_comparator(segment_borders: pd.DataFrame) -> Callable[[str, str], int]:
    """Create comparator function to sort border names."""

    def cmp(name_a: str, name_b: str) -> int:
        """Compare two series based on their values."""
        border_a = segment_borders[name_a].values
        border_b = segment_borders[name_b].values

        if np.all(border_a == border_b):
            return 0
        elif np.all(border_a <= border_b):
            return -1
        elif np.all(border_a >= border_b):
            return 1
        else:
            raise ValueError("Detected intersection between non-equal borders!")

    return cmp


def plot_forecast(
    forecast_ts: Union["TSDataset", List["TSDataset"], Dict[str, "TSDataset"]],
    test_ts: Optional["TSDataset"] = None,
    train_ts: Optional["TSDataset"] = None,
    segments: Optional[List[str]] = None,
    n_train_samples: Optional[int] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
    prediction_intervals: bool = False,
    quantiles: Optional[List[float]] = None,
):
    """
    Plot of prediction for forecast pipeline.

    Parameters
    ----------
    forecast_ts:
        there are several options:

        #. Forecasted TSDataset with timeseries data, single-forecast mode

        #. List of forecasted TSDatasets, multi-forecast mode

        #. Dictionary with forecasted TSDatasets, multi-forecast mode

    test_ts:
        TSDataset with timeseries data
    train_ts:
        TSDataset with timeseries data
    segments:
        segments to plot; if not given plot all the segments from ``forecast_df``
    n_train_samples:
        length of history of train to plot
    columns_num:
        number of graphics columns
    figsize:
        size of the figure per subplot with one segment in inches
    prediction_intervals:
        if True prediction intervals will be drawn
    quantiles:
        List of quantiles to draw, if isn't set then quantiles from a given dataset will be used.
        In multi-forecast mode, only quantiles present in each forecast will be used.

    Raises
    ------
    ValueError:
        if the format of ``forecast_ts`` is unknown
    ValueError:
        if there is an intersection between non-equal borders
    ValueError:
        if provided quantiles are not in the datasets
    """
    forecast_results = _prepare_forecast_results(forecast_ts)
    num_forecasts = len(forecast_results.keys())

    if segments is None:
        unique_segments = set()
        for forecast in forecast_results.values():
            unique_segments.update(forecast.segments)
        segments = list(unique_segments)

    _, ax = _prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)

    if prediction_intervals:
        prediction_intervals_names = _select_prediction_intervals_names(forecast_results, quantiles)

    for i, segment in enumerate(segments):
        if train_ts is not None:
            segment_train_df = train_ts[:, segment, :][segment]
        else:
            segment_train_df = pd.DataFrame(columns=["timestamp", "target", "segment"])

        if test_ts is not None:
            segment_test_df = test_ts[:, segment, :][segment]
        else:
            segment_test_df = pd.DataFrame(columns=["timestamp", "target", "segment"])

        if n_train_samples is None:
            plot_df = segment_train_df
        elif n_train_samples != 0:
            plot_df = segment_train_df.iloc[-n_train_samples:]
        else:
            plot_df = pd.DataFrame(columns=["timestamp", "target", "segment"])

        if (train_ts is not None) and (n_train_samples != 0):
            marker = None if len(plot_df) > 1 else "o"
            ax[i].plot(plot_df.index.values, plot_df["target"].values, label="train", marker=marker)
        if test_ts is not None:
            marker = None if len(segment_test_df) > 1 else "o"
            ax[i].plot(
                segment_test_df.index.values,
                segment_test_df["target"].values,
                color="purple",
                label="test",
                marker=marker,
            )

        # plot forecast plot for each of given forecasts
        for forecast_name, forecast in forecast_results.items():
            legend_prefix = f"{forecast_name}: " if num_forecasts > 1 else ""

            segment_forecast_df = forecast[:, segment, :][segment]
            marker = None if len(segment_forecast_df) > 1 else "o"

            line = ax[i].plot(
                segment_forecast_df.index.values,
                segment_forecast_df["target"].values,
                linewidth=1,
                label=f"{legend_prefix}forecast",
                marker=marker,
            )
            forecast_color = line[0].get_color()

            # draw prediction intervals from outer layers to inner ones
            intervals = forecast.get_prediction_intervals()
            if prediction_intervals and intervals is not None:
                alpha = np.linspace(0, 1 / 2, len(prediction_intervals_names) // 2 + 2)[1:-1]

                segment_borders_df = intervals.loc[:, pd.IndexSlice[segment, :]].droplevel(level="segment", axis=1)
                comparator = _get_borders_comparator(segment_borders=segment_borders_df)
                prediction_intervals_names = sorted(prediction_intervals_names, key=cmp_to_key(comparator))

                for interval_idx in range(len(prediction_intervals_names) // 2):
                    # define upper and lower border for this iteration
                    low_border = prediction_intervals_names[interval_idx]
                    high_border = prediction_intervals_names[-interval_idx - 1]
                    values_low = segment_borders_df[low_border].values
                    values_high = segment_borders_df[high_border].values

                    # if (low_border, high_border) is the smallest interval
                    if interval_idx == len(prediction_intervals_names) // 2 - 1:
                        ax[i].fill_between(
                            segment_borders_df.index.values,
                            values_low,
                            values_high,
                            linewidth=3,
                            color=forecast_color,
                            alpha=alpha[interval_idx],
                            label=f"{legend_prefix}{low_border}-{high_border}",
                        )

                    # if there is some interval inside (low_border, high_border) we should plot around it
                    else:
                        low_next_border = prediction_intervals_names[interval_idx + 1]
                        high_prev_border = prediction_intervals_names[-interval_idx - 2]
                        values_next = segment_borders_df[low_next_border].values
                        ax[i].fill_between(
                            segment_borders_df.index.values,
                            values_low,
                            values_next,
                            linewidth=3,
                            color=forecast_color,
                            alpha=alpha[interval_idx],
                            label=f"{legend_prefix}{low_border}-{high_border}",
                        )
                        values_prev = segment_borders_df[high_prev_border].values
                        ax[i].fill_between(
                            segment_borders_df.index.values,
                            values_high,
                            values_prev,
                            linewidth=3,
                            color=forecast_color,
                            alpha=alpha[interval_idx],
                        )
                # when we can't find pair for border, we plot it separately
                if len(prediction_intervals_names) % 2 != 0:
                    remaining_border = prediction_intervals_names[len(prediction_intervals_names) // 2]
                    values = segment_borders_df[remaining_border].values
                    marker = "--" if len(values) > 1 else "d"
                    ax[i].plot(
                        segment_borders_df.index.values,
                        values,
                        marker,
                        color=forecast_color,
                        label=f"{legend_prefix}{remaining_border}",
                    )
        ax[i].set_title(segment)
        ax[i].tick_params("x", rotation=45)
        ax[i].legend(loc="upper left")


def plot_backtest(
    forecast_ts_list: List["TSDataset"],
    ts: "TSDataset",
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    history_len: Union[int, Literal["all"]] = 0,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot targets and forecast for backtest pipeline.

    This function doesn't support intersecting folds.

    Parameters
    ----------
    forecast_ts_list:
        List of TSDataset with forecast for each fold from backtest
    ts:
        dataframe of timeseries that was used for backtest
    segments:
        segments to plot
    columns_num:
        number of subplots columns
    history_len:
        length of pre-backtest history to plot, if value is "all" then plot all the history
    figsize:
        size of the figure per subplot with one segment in inches

    Raises
    ------
    ValueError:
        if ``history_len`` is negative
    ValueError:
        if folds are intersecting
    """
    if history_len != "all" and history_len < 0:
        raise ValueError("Parameter history_len should be non-negative or 'all'")

    if segments is None:
        segments = sorted(ts.segments)

    fold_numbers = pd.concat(
        [pd.Series(i, index=forecast_ts.timestamps) for i, forecast_ts in enumerate(forecast_ts_list)], axis=0
    )
    _validate_intersecting_folds(fold_numbers)

    # prepare dataframes
    df = ts.to_pandas()
    forecast_start = fold_numbers.index[0]
    history_df = df[df.index <= forecast_start]
    backtest_df = df[df.index >= forecast_start]

    # prepare colors
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = itertools.cycle(default_colors)
    lines_colors = {line_name: next(color_cycle) for line_name in ["history", "test", "forecast"]}

    _, ax = _prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)

    list_forecast_df = [forecast_ts.to_pandas() for forecast_ts in forecast_ts_list]

    # plot history
    for i, segment in enumerate(segments):
        segment_history_df = history_df[segment]

        if history_len == "all":
            ax[i].plot(segment_history_df.index, segment_history_df["target"], color=lines_colors["history"])
        elif history_len > 0:
            tail_segment_history_df = segment_history_df.tail(history_len)
            ax[i].plot(tail_segment_history_df.index, tail_segment_history_df["target"], color=lines_colors["history"])

    # Define type of forecast plot (line or scatters)
    is_full_fold = set(backtest_df.index) == set(fold_numbers.index)
    single_point_forecast = len(backtest_df) == 1
    draw_only_lines = is_full_fold and not single_point_forecast

    # plot test and forecast
    for fold_number, fold_forecast_df in enumerate(list_forecast_df):
        start_fold = fold_forecast_df.index.min()
        end_fold_exclusive = timestamp_range(start=fold_forecast_df.index.max(), periods=2, freq=ts.freq)[-1]

        fold_backtest_df = backtest_df.loc[start_fold:end_fold_exclusive]

        for i, segment in enumerate(segments):
            segment_fold_backtest_df = fold_backtest_df[segment]
            segment_fold_forecast_df = fold_forecast_df[segment]

            # draw test
            ax[i].plot(segment_fold_backtest_df.index, segment_fold_backtest_df["target"], color=lines_colors["test"])

            if draw_only_lines:
                # draw forecast
                if fold_number != fold_numbers.iloc[-1]:
                    next_fold_first_value = list_forecast_df[fold_number + 1][segment].iloc[[0]]
                    segment_fold_forecast_df = pd.concat([segment_fold_forecast_df, next_fold_first_value], axis=0)

                ax[i].plot(
                    segment_fold_forecast_df.index, segment_fold_forecast_df["target"], color=lines_colors["forecast"]
                )

            else:
                segment_fold_backtest_df = segment_fold_backtest_df.loc[segment_fold_forecast_df.index]

                # draw points on test
                ax[i].scatter(
                    segment_fold_backtest_df.index, segment_fold_backtest_df["target"], color=lines_colors["test"]
                )

                # draw forecast
                ax[i].scatter(
                    segment_fold_forecast_df.index, segment_fold_forecast_df["target"], color=lines_colors["forecast"]
                )

            # draw borders of current fold
            opacity = 0.075 * ((fold_number + 1) % 2) + 0.075
            ax[i].axvspan(
                start_fold,
                end_fold_exclusive,
                alpha=opacity,
                color="skyblue",
            )

    # add title and legend
    for i, segment in enumerate(segments):
        legend_handles = [
            Line2D([0], [0], marker="o", color=color, label=label) for label, color in lines_colors.items()
        ]
        ax[i].legend(handles=legend_handles)

        ax[i].set_title(segment)
        ax[i].tick_params("x", rotation=45)


def plot_backtest_interactive(
    forecast_ts_list: List["TSDataset"],
    ts: "TSDataset",
    segments: Optional[List[str]] = None,
    history_len: Union[int, Literal["all"]] = 0,
    figsize: Tuple[int, int] = (900, 600),
) -> go.Figure:
    """Plot targets and forecast for backtest pipeline using plotly.

    Parameters
    ----------
    forecast_ts_list:
        List of TSDataset with forecast for each fold from backtest
    ts:
        dataframe of timeseries that was used for backtest
    segments:
        segments to plot
    history_len:
        length of pre-backtest history to plot, if value is "all" then plot all the history
    figsize:
        size of the figure in pixels

    Returns
    -------
    :
        result of plotting

    Raises
    ------
    ValueError:
        if ``history_len`` is negative
    ValueError:
        if folds are intersecting
    """
    if history_len != "all" and history_len < 0:
        raise ValueError("Parameter history_len should be non-negative or 'all'")

    if segments is None:
        segments = sorted(ts.segments)

    fold_numbers = pd.concat(
        [pd.Series(i, index=forecast_ts.timestamps) for i, forecast_ts in enumerate(forecast_ts_list)], axis=0
    )
    _validate_intersecting_folds(fold_numbers)

    # prepare dataframes
    df = ts.to_pandas()
    forecast_start = fold_numbers.index[0]
    history_df = df[df.index <= forecast_start]
    backtest_df = df[df.index >= forecast_start]

    list_forecast_df = [forecast_ts.to_pandas() for forecast_ts in forecast_ts_list]

    # prepare colors
    colors = plotly.colors.qualitative.Dark24

    fig = go.Figure()

    # plot history
    for i, segment in enumerate(segments):
        segment_history_df = history_df[segment]

        if history_len == "all":
            plot_df = segment_history_df
        elif history_len > 0:
            plot_df = segment_history_df.tail(history_len)
        else:
            break
        fig.add_trace(
            go.Scattergl(
                x=plot_df.index,
                y=plot_df["target"],
                legendgroup=f"{segment}",
                name=f"{segment}",
                mode="lines",
                marker_color=colors[i % len(colors)],
                showlegend=True,
                line=dict(width=2, dash="dash"),
            )
        )

    # Define type of forecast plot (line or scatters)
    is_full_fold = set(backtest_df.index) == set(fold_numbers.index)
    single_point_forecast = len(backtest_df) == 1
    draw_only_lines = is_full_fold and not single_point_forecast

    for fold_number, fold_forecast_df in enumerate(list_forecast_df):
        start_fold = fold_forecast_df.index.min()
        end_fold_exclusive = timestamp_range(start=fold_forecast_df.index.max(), periods=2, freq=ts.freq)[-1]

        fold_backtest_df = backtest_df.loc[start_fold:end_fold_exclusive]

        for i, segment in enumerate(segments):
            segment_fold_backtest_df = fold_backtest_df[segment]
            segment_fold_forecast_df = fold_forecast_df[segment]

            # draw test
            fig.add_trace(
                go.Scattergl(
                    x=segment_fold_backtest_df.index,
                    y=segment_fold_backtest_df["target"],
                    legendgroup=f"{segment}",
                    name=f"Test: {segment}",
                    mode="lines",
                    marker_color=colors[i % len(colors)],
                    showlegend=False,
                    line=dict(width=2, dash="solid"),
                )
            )

            if draw_only_lines:
                # draw forecast
                if fold_number != len(forecast_ts_list) - 1:
                    next_fold_first_value = list_forecast_df[fold_number + 1][segment].iloc[[0]]
                    segment_fold_forecast_df = pd.concat([segment_fold_forecast_df, next_fold_first_value], axis=0)

                fig.add_trace(
                    go.Scattergl(
                        x=segment_fold_forecast_df.index,
                        y=segment_fold_forecast_df["target"],
                        legendgroup=f"{segment}",
                        name=f"Forecast: {segment}",
                        mode="lines",
                        marker_color=colors[i % len(colors)],
                        showlegend=False,
                        line=dict(width=2, dash="dot"),
                    )
                )

            else:
                segment_fold_backtest_df = segment_fold_backtest_df.loc[segment_fold_forecast_df.index]

                # draw points on test
                fig.add_trace(
                    go.Scattergl(
                        x=segment_fold_backtest_df.index,
                        y=segment_fold_backtest_df["target"],
                        legendgroup=f"{segment}",
                        name=f"Test: {segment}",
                        mode="markers",
                        marker_color=colors[i % len(colors)],
                        showlegend=False,
                    )
                )

                # draw forecast
                fig.add_trace(
                    go.Scattergl(
                        x=segment_fold_forecast_df.index,
                        y=segment_fold_forecast_df["target"],
                        legendgroup=f"{segment}",
                        name=f"Forecast: {segment}",
                        mode="markers",
                        marker_color=colors[i % len(colors)],
                        showlegend=False,
                    )
                )

            if i == 0:
                opacity = 0.075 * ((fold_number + 1) % 2) + 0.075
                fig.add_vrect(
                    x0=start_fold,
                    x1=end_fold_exclusive,
                    line_width=0,
                    fillcolor="blue",
                    opacity=opacity,
                )

    fig.update_layout(
        height=figsize[1],
        width=figsize[0],
        title="Backtest for all segments",
        xaxis_title="timestamp",
        yaxis_title="target",
        legend=dict(itemsizing="trace", title="Segments"),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                xanchor="left",
                yanchor="top",
                showactive=True,
                x=1.0,
                y=1.1,
                buttons=[
                    dict(method="restyle", args=["visible", "all"], label="show all"),
                    dict(method="restyle", args=["visible", "legendonly"], label="hide all"),
                ],
            )
        ],
        annotations=[
            dict(text="Show segments:", showarrow=False, x=1.0, y=1.08, xref="paper", yref="paper", align="left")
        ],
    )

    return fig


def plot_residuals(
    forecast_ts_list: List["TSDataset"],
    ts: "TSDataset",
    feature: Union[str, Literal["timestamp"]] = "timestamp",
    transforms: Sequence["Transform"] = (),
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot residuals for predictions from backtest against some feature.

    Parameters
    ----------
    forecast_ts_list:
        List of TSDataset with forecast for each fold from backtest
    ts:
        dataframe of timeseries that was used for backtest
    feature:
        feature name to draw against residuals, if "timestamp" plot residuals against the timestamp
    transforms:
        sequence of transforms to get feature column
    segments:
        segments to use
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches

    Raises
    ------
    ValueError:
        if feature isn't present in the dataset after applying transformations

    Notes
    -----
    Parameter ``transforms`` is necessary because some pipelines doesn't save features in their forecasts,
    e.g. :py:mod:`etna.ensembles` pipelines.
    """
    if segments is None:
        segments = sorted(ts.segments)

    _, ax = _prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)

    fold_numbers = pd.concat(
        [pd.Series(i, index=forecast_ts.timestamps) for i, forecast_ts in enumerate(forecast_ts_list)], axis=0
    )

    ts_copy = deepcopy(ts)
    ts_copy.fit_transform(transforms=transforms)
    ts_residuals = get_residuals(forecast_ts_list=forecast_ts_list, ts=ts_copy)
    df = ts_residuals.to_pandas()
    # check if feature is present in dataset
    if feature != "timestamp":
        all_features = set(df.columns.get_level_values("feature").unique())
        if feature not in all_features:
            raise ValueError("Given feature isn't present in the dataset after applying transformations")

    for i, segment in enumerate(segments):
        segment_residuals_df = df.loc[:, pd.IndexSlice[segment, :]][segment].reset_index()
        residuals = segment_residuals_df["target"].values
        feature_values = segment_residuals_df[feature].values

        # highlight different backtest folds
        if feature == "timestamp":
            folds = (i for i in range(len(forecast_ts_list)))
            for fold_number in folds:
                fold_indices = fold_numbers[fold_numbers == fold_number].index
                ax[i].axvspan(
                    fold_indices.min(),
                    fold_indices.max(),
                    alpha=0.15 * int((fold_number + 1) % 2),
                    color="skyblue",
                )

        ax[i].scatter(feature_values, residuals, c="b")

        ax[i].set_title(segment)
        ax[i].tick_params("x", rotation=45)
        ax[i].set_xlabel(feature)


def qq_plot(
    residuals_ts: "TSDataset",
    qq_plot_params: Optional[Dict[str, Any]] = None,
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot Q-Q plots for segments.

    Parameters
    ----------
    residuals_ts:
        dataset with the time series, expected to be the residuals of the model
    qq_plot_params:
        dictionary with parameters for qq plot, :py:func:`statsmodels.graphics.gofplots.qqplot` is used
    segments:
        segments to plot
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if qq_plot_params is None:
        qq_plot_params = {}
    if segments is None:
        segments = sorted(residuals_ts.segments)

    _, ax = _prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)

    residuals_df = residuals_ts.to_pandas()
    for i, segment in enumerate(segments):
        residuals_segment = residuals_df.loc[:, pd.IndexSlice[segment, "target"]]
        qqplot(residuals_segment, ax=ax[i], **qq_plot_params)
        ax[i].set_title(segment)


class PerFoldAggregation(str, Enum):
    """Enum for types of aggregation in a metric per-segment plot."""

    #: Mean aggregation.
    mean = "mean"

    #: Median aggregation.
    sum = "median"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} aggregations are allowed"
        )

    def get_function(self):
        """Get aggregation function."""
        if self.value == "mean":
            return np.nanmean
        elif self.value == "median":
            return np.nanmedian


def plot_metric_per_segment(
    metrics_df: pd.DataFrame,
    metric_name: str,
    ascending: bool = False,
    per_fold_aggregation_mode: str = PerFoldAggregation.mean.value,
    top_k: Optional[int] = None,
    barplot_params: Optional[Dict[str, Any]] = None,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot barplot with per-segment metrics.

    If for some segment all metric values are missing, it isn't plotted, and the warning is raised.

    If some segments have different set of folds with non-missing metrics,
    it can lead to incompatible values between folds. The warning is raised in such case.

    Parameters
    ----------
    metrics_df:
        dataframe with metrics calculated on the backtest
    metric_name:
        name of the metric to visualize
    ascending:

        * If True, small values at the top;

        * If False, big values at the top.

    per_fold_aggregation_mode:
        how to aggregate metrics over the folds if they aren't already aggregated
        (see :py:class:`~etna.analysis.forecast.plots.PerFoldAggregation`)
    top_k:
        number segments to show after ordering according to ``ascending``
    barplot_params:
        dictionary with parameters for plotting, :py:func:`seaborn.barplot` is used
    figsize:
        size of the figure per subplot with one segment in inches

    Raises
    ------
    ValueError:
        if ``metric_name`` isn't present in ``metrics_df``
    NotImplementedError:
        unknown ``per_fold_aggregation_mode`` is given

    Warnings
    --------
    UserWarning:
        There are segments with all missing metric values.
    UserWarning:
        Some segments have different set of folds to be aggregated on due to missing values.
    """
    if barplot_params is None:
        barplot_params = {}

    aggregation_mode = PerFoldAggregation(per_fold_aggregation_mode)

    _check_metrics_df_empty_segments(metrics_df=metrics_df, metric_name=metric_name)
    _check_metrics_df_same_folds_for_each_segment(metrics_df=metrics_df, metric_name=metric_name)

    plt.figure(figsize=figsize)

    if metric_name not in metrics_df.columns:
        raise ValueError("Given metric_name isn't present in metrics_df")

    if "fold_number" in metrics_df.columns:
        metrics_dict = (
            metrics_df.groupby("segment")
            .agg({metric_name: aggregation_mode.get_function()})
            .dropna()
            .to_dict()[metric_name]
        )
    else:
        metrics_dict = metrics_df[["segment", metric_name]].set_index("segment").dropna().to_dict()[metric_name]

    segments = np.array(list(metrics_dict.keys()))
    values = np.array(list(metrics_dict.values()))
    sort_idx = np.argsort(values)
    if not ascending:
        sort_idx = sort_idx[::-1]
    segments = segments[sort_idx][:top_k]
    values = values[sort_idx][:top_k]
    sns.barplot(x=values, y=segments, orient="h", **barplot_params, hue=segments, legend=False, palette="tab10")
    plt.title("Metric per-segment plot")
    plt.ylabel("Segment")
    plt.xlabel(metric_name)
    plt.grid()


class MetricPlotType(str, Enum):
    """Enum for types of plot in :py:func:`~etna.analysis.forecast.plots.metric_per_segment_distribution_plot`."""

    #: Histogram plot, :py:func:`seaborn.distributions.histplot` is used.
    hist = "hist"

    #: Boxplot, :py:func:`seaborn.categorical.boxplot` is used.
    box = "box"

    #: Violin plot, :py:func:`seaborn.categorical.violinplot` is used.
    violin = "violin"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} plots are allowed"
        )

    def get_function(self):
        """Get aggregation function."""
        if self.value == "hist":
            return sns.histplot
        elif self.value == "box":
            return sns.boxplot
        elif self.value == "violin":
            return sns.violinplot


def metric_per_segment_distribution_plot(
    metrics_df: pd.DataFrame,
    metric_name: str,
    per_fold_aggregation_mode: Optional[str] = None,
    plot_type: Union[Literal["hist"], Literal["box"], Literal["violin"]] = "hist",
    seaborn_params: Optional[Dict[str, Any]] = None,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot distribution of metric values over all segments.

    If for some segment all metric values are missing, it isn't plotted, and the warning is raised.

    If some segments have different set of folds with non-missing metrics,
    it can lead to incompatible values between folds. The warning is raised in such case.

    Parameters
    ----------
    metrics_df:
        dataframe with metrics calculated on the backtest
    metric_name:
        name of the metric to visualize
    per_fold_aggregation_mode:

        * If None, separate distributions for each fold will be drawn

        * If str, determines how to aggregate metrics over the folds if they aren't already aggregated
          (see :py:class:`~etna.analysis.forecast.plots.PerFoldAggregation`)

    plot_type:
        type of plot (see :py:class:`~etna.analysis.forecast.plots.MetricPlotType`)
    seaborn_params:
        dictionary with parameters for plotting
    figsize:
        size of the figure per subplot with one segment in inches

    Raises
    ------
    ValueError:
        if ``metric_name`` isn't present in ``metrics_df``
    NotImplementedError:
        unknown ``per_fold_aggregation_mode`` is given

    Warnings
    --------
    UserWarning:
        There are segments with all missing metric values.
    UserWarning:
        Some segments have different set of folds to be aggregated on due to missing values.
    """
    if seaborn_params is None:
        seaborn_params = {}

    metrics_df = metrics_df.reset_index(drop=True)
    plot_type_enum = MetricPlotType(plot_type)
    plot_function = plot_type_enum.get_function()

    plt.figure(figsize=figsize)

    if metric_name not in metrics_df.columns:
        raise ValueError("Given metric_name isn't present in metrics_df")

    _check_metrics_df_empty_segments(metrics_df=metrics_df, metric_name=metric_name)
    _check_metrics_df_same_folds_for_each_segment(metrics_df=metrics_df, metric_name=metric_name)

    # draw plot for each fold
    if per_fold_aggregation_mode is None and "fold_number" in metrics_df.columns:
        if plot_type_enum == MetricPlotType.hist:
            plot_function(data=metrics_df, x=metric_name, hue="fold_number", palette="tab10", **seaborn_params)
        else:
            plot_function(
                data=metrics_df,
                x="fold_number",
                y=metric_name,
                hue="fold_number",
                palette="tab10",
                legend=False,
                **seaborn_params,
            )
            plt.xlabel("Fold")

    # draw one plot of aggregated data
    else:
        if "fold_number" in metrics_df.columns:
            agg_func = PerFoldAggregation(per_fold_aggregation_mode).get_function()
            metrics_df = metrics_df.groupby("segment").agg({metric_name: agg_func})

        if plot_type_enum == MetricPlotType.hist:
            plot_function(data=metrics_df, x=metric_name, **seaborn_params)
        else:
            plot_function(data=metrics_df, y=metric_name, **seaborn_params)

    plt.title("Metric per-segment distribution plot")
    plt.grid()


class ComponentsMode(str, Enum):
    """Enum for components plotting modes."""

    per_component = "per-component"
    joint = "joint"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Supported modes: {', '.join([repr(m.value) for m in cls])}"
        )


def plot_forecast_decomposition(
    forecast_ts: "TSDataset",
    test_ts: Optional["TSDataset"] = None,
    mode: Union[Literal["per-component"], Literal["joint"]] = "per-component",
    segments: Optional[List[str]] = None,
    columns_num: int = 1,
    figsize: Tuple[int, int] = (10, 5),
    show_grid: bool = False,
):
    """
    Plot of prediction and its components.

    Parameters
    ----------
    forecast_ts:
        forecasted TSDataset with timeseries data, single-forecast mode
    test_ts:
        TSDataset with timeseries data
    mode:
        Components plotting type

        #. ``per-component`` -- plot each component in separate axes

        #. ``joint`` -- plot all the components in the same axis

    segments:
        segments to plot; if not given plot all the segments
    columns_num:
        number of graphics columns; when ``mode="per-component"`` all plots will be in the single column
    figsize:
        size of the figure per subplot with one segment in inches
    show_grid:
        whether to show grid for each chart

    Raises
    ------
    ValueError:
        if components aren't present in ``forecast_ts``
    NotImplementedError:
        unknown ``mode`` is given
    """
    components_mode = ComponentsMode(mode)

    if segments is None:
        segments = forecast_ts.segments

    components = forecast_ts.target_components_names

    if len(components) == 0:
        raise ValueError("No components were detected in the provided `forecast_ts`.")

    if components_mode == ComponentsMode.joint:
        num_plots = len(segments)
    else:
        # plotting target and forecast separately from components, thus +1 for each segment
        num_plots = math.ceil(len(segments) / columns_num) * columns_num * (len(components) + 1)

    _, ax = _prepare_axes(num_plots=num_plots, columns_num=columns_num, figsize=figsize, set_grid=show_grid)

    alpha = 0.5 if components_mode == ComponentsMode.joint else 1.0
    ax_array = np.asarray(ax).reshape(-1, columns_num).T.ravel()

    i = 0
    for segment in segments:
        skip_first_component = test_ts is None

        if test_ts is not None:
            segment_test_df = test_ts[:, segment, :][segment]
        else:
            segment_test_df = pd.DataFrame(columns=["timestamp", "target", "segment"])

        segment_forecast_df = forecast_ts[:, segment, :][segment]

        ax_array[i].set_title(segment)

        ax_array[i].plot(segment_forecast_df.index.values, segment_forecast_df["target"].values, label="forecast")

        if test_ts is not None:
            ax_array[i].plot(segment_test_df.index.values, segment_test_df["target"].values, label="target")

        for component in components:
            if components_mode == ComponentsMode.per_component:
                ax_array[i].legend(loc="upper left")
                ax_array[i].set_xticklabels([])
                i += 1

            ax_array[i].plot(
                segment_forecast_df.index.values, segment_forecast_df[component].values, label=component, alpha=alpha
            )

            if skip_first_component:
                # skip color for target
                ax_array[i].get_lines()[1].set_color(ax_array[i]._get_lines.get_next_color())
                skip_first_component = False

        ax_array[i].tick_params("x", rotation=45)
        ax_array[i].legend(loc="upper left")
        i += 1


def prediction_actual_scatter_plot(
    forecast_ts_list: List["TSDataset"],
    ts: "TSDataset",
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot scatter plot with forecasted/actual values for segments.

    Parameters
    ----------
    forecast_ts_list:
        List of TSDataset with forecast for each fold from backtest
    ts:
        dataframe of timeseries that was used for backtest
    segments:
        segments to plot
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if segments is None:
        segments = sorted(ts.segments)

    _, ax = _prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)

    df = ts.to_pandas()
    forecast_df = pd.concat([forecast_ts.to_pandas() for forecast_ts in forecast_ts_list], axis=0)

    for i, segment in enumerate(segments):
        forecast_segment_df = forecast_df.loc[:, pd.IndexSlice[segment, "target"]]
        segment_df = df.loc[forecast_segment_df.index, pd.IndexSlice[segment, "target"]]

        # fit a linear model
        x = forecast_segment_df.values
        y = segment_df
        model = LinearRegression()
        model.fit(X=x[:, np.newaxis], y=y)
        r2 = r2_score(y_true=y, y_pred=model.predict(x[:, np.newaxis]))

        # prepare the limits of the plot, for the identity to be from corner to corner
        x_min = min(x.min(), y.min())
        x_max = max(x.max(), y.max())
        # add some space at the borders of the plot
        x_min -= 0.05 * (x_max - x_min)
        x_max += 0.05 * (x_max - x_min)
        xlim = (x_min, x_max)
        ylim = xlim

        # make plots
        ax[i].scatter(x, y, label=f"R2: {r2:.3f}")
        x_grid = np.linspace(*xlim, 100)
        ax[i].plot(x_grid, x_grid, label="identity", linestyle="dotted", color="grey")
        ax[i].plot(
            x_grid,
            model.predict(x_grid[:, np.newaxis]),
            label=f"best fit: {model.coef_[0]:.3f} x + {model.intercept_:.3f}",
            linestyle="dashed",
            color="black",
        )
        ax[i].set_title(segment)
        ax[i].set_xlabel("$\\widehat{y}$")
        ax[i].set_ylabel("$y$")
        ax[i].set_xlim(*xlim)
        ax[i].set_ylim(*ylim)
        ax[i].legend()
