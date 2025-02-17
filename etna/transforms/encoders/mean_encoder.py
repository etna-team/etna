import reprlib
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import numba
import numpy as np
import pandas as pd
from bottleneck import nanmean

from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.distributions import FloatDistribution
from etna.transforms import IrreversibleTransform


class EncoderMode(str, Enum):
    """Enum for different encoding strategies."""

    per_segment = "per-segment"
    macro = "macro"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(f"The strategy '{value}' doesn't exist")


class MissingMode(str, Enum):
    """Enum for handle missing strategies."""

    category = "category"
    global_mean = "global_mean"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Supported types: {', '.join([repr(m.value) for m in cls])}"
        )


class MeanEncoderTransform(IrreversibleTransform):
    """
    Makes encoding of categorical feature.

    For timestamps that are before the last timestamp seen in ``fit`` transformations are made using the formula below:

    .. math::
       \\frac{TargetSum + RunningMean * Smoothing}{FeatureCount + Smoothing}

    where

    * TargetSum is the sum of target up to the current timestamp for the current category, not including the current timestamp
    * RunningMean is target mean up to the current timestamp, not including the current timestamp
    * FeatureCount is the number of categories with the same value as in the current timestamp, not including the current timestamp

    For future timestamps:

    * for known categories encoding are filled with global mean of target for these categories calculated during ``fit``
    * for unknown categories encoding are filled with global mean of target in the whole dataset calculated during ``fit``

    All types of NaN values are considering as one category.
    """

    idx = pd.IndexSlice

    def __init__(
        self,
        in_column: str,
        out_column: str,
        mode: Union[EncoderMode, str] = "per-segment",
        handle_missing: str = MissingMode.category,
        smoothing: int = 1,
    ):
        """
        Init MeanEncoderTransform.

        Parameters
        ----------
        in_column:
            categorical column to apply transform
        out_column:
            name of added column
        mode:
            mode to encode segments

            * 'per-segment' - statistics are calculated across each segment individually

            * 'macro' - statistics are calculated across all segments. In this mode transform can work with new segments that were not seen during ``fit``
        handle_missing:
            mode to handle missing values in ``in_column``

            * 'category' - NaNs they are interpreted as a separate categorical feature

            * 'global_mean' - NaNs are filled with the running mean
        smoothing:
            smoothing parameter
        """
        super().__init__(required_features=["target", in_column])
        self.in_column = in_column
        self.out_column = out_column
        self.mode = EncoderMode(mode)
        self.handle_missing = MissingMode(handle_missing)
        self.smoothing = smoothing

        self._global_means: Optional[Union[float, Dict[str, float]]] = None
        self._global_means_category: Optional[Union[Dict[str, float], Dict[str, Dict[str, float]]]] = None
        self._last_timestamp: Union[pd.Timestamp, int, None]

    def _fit(self, df: pd.DataFrame) -> "MeanEncoderTransform":
        """
        Fit encoder.

        Parameters
        ----------
        df:
            dataframe with data to fit expanding mean target encoder.

        Returns
        -------
        :
            Fitted transform
        """
        df.loc[:, pd.IndexSlice[:, self.in_column]] = df.loc[:, pd.IndexSlice[:, self.in_column]].fillna(np.NaN)

        if self.mode is EncoderMode.per_segment:
            axis = 0
            segments = df.columns.get_level_values("segment").unique().tolist()
            global_means = nanmean(df.loc[:, self.idx[:, "target"]], axis=axis)
            global_means = dict(zip(segments, global_means))

            global_means_category = {}
            for segment in segments:
                segment_df = TSDataset.to_flatten(df.loc[:, pd.IndexSlice[segment, :]])
                global_means_category[segment] = (
                    segment_df[[self.in_column, "target"]]
                    .groupby(self.in_column, dropna=False)
                    .mean()
                    .to_dict()["target"]
                )
        else:
            axis = None
            global_means = nanmean(df.loc[:, self.idx[:, "target"]], axis=axis)

            segment_df = TSDataset.to_flatten(df)
            global_means_category = (
                segment_df[[self.in_column, "target"]].groupby(self.in_column, dropna=False).mean().to_dict()["target"]
            )

        self._global_means = global_means
        self._global_means_category = global_means_category
        self._last_timestamp = df.index[-1]

        return self

    @staticmethod
    def _count_macro_running_mean(df, n_segments):
        y = df["target"]
        timestamp_count = y.groupby(df["timestamp"]).transform("count")
        timestamp_sum = y.groupby(df["timestamp"]).transform("sum")
        expanding_mean = timestamp_sum.iloc[::n_segments].cumsum() / timestamp_count.iloc[::n_segments].cumsum()
        expanding_mean = expanding_mean.repeat(n_segments)
        # first timestamp is NaN
        expanding_mean = pd.Series(index=df.index, data=expanding_mean.values).shift(n_segments)
        return expanding_mean

    @staticmethod
    @numba.njit()
    def _count_per_segment_cumstats(target: np.ndarray, categories: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ans_cumsum = np.full_like(target, np.nan)
        ans_cumcount = np.full_like(target, np.nan)
        unique_categories = np.unique(categories)
        for category in unique_categories:
            idx = np.where(category == categories)[0]
            t = target[idx]

            # Mask for valid (non-NaN) target values
            valid = ~np.isnan(t)

            # Compute cumulative sums and counts for valid values
            cumsum = np.cumsum(np.where(valid, t, 0))
            cumcount = np.cumsum(valid).astype(np.float32)

            # Shift statistics by 1 to get statistics not including current index
            cumsum = np.roll(cumsum, 1)
            cumcount = np.roll(cumcount, 1)

            cumsum[0] = np.NaN
            cumcount[0] = np.NaN

            # Handle positions with no previous valid values
            cumsum[cumcount == 0] = np.NaN
            cumcount[cumcount == 0] = np.NaN

            # Assign the computed values back to the answer arrays
            ans_cumsum[idx] = cumsum
            ans_cumcount[idx] = cumcount
        return ans_cumsum, ans_cumcount

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get encoded values for the segment.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        :
            result dataframe

        Raises
        ------
        ValueError:
            If transform isn't fitted.
        NotImplementedError:
            If there are segments that weren't present during training.
        """
        if self._global_means is None:
            raise ValueError("The transform isn't fitted!")

        segments = df.columns.get_level_values("segment").unique().tolist()
        n_segments = len(segments)
        if self.mode is EncoderMode.per_segment:
            self._global_means = cast(Dict[str, float], self._global_means)
            new_segments = set(segments) - self._global_means.keys()
            if len(new_segments) > 0:
                raise NotImplementedError(
                    f"This transform can't process segments that weren't present on train data: {reprlib.repr(new_segments)}"
                )
        df.loc[:, self.idx[:, self.in_column]] = df.loc[:, self.idx[:, self.in_column]].fillna(np.NaN)

        future_timestamps = df.index[df.index > self._last_timestamp]
        intersected_timestamps = df.index[df.index <= self._last_timestamp]

        intersected_df = df.loc[intersected_timestamps, self.idx[:, :]]
        future_df = df.loc[future_timestamps, self.idx[:, :]]

        if len(intersected_df) > 0:
            if self.mode is EncoderMode.per_segment:
                for segment in segments:
                    segment_df = TSDataset.to_flatten(intersected_df.loc[:, self.idx[segment, :]])
                    y = segment_df["target"]
                    categories = segment_df[self.in_column].values.astype(str)

                    unique_categories = np.unique(categories)
                    cat_to_int = {cat: idx for idx, cat in enumerate(unique_categories)}
                    int_categories = np.array([cat_to_int[cat] for cat in categories], dtype=np.int64)

                    # first timestamp is NaN
                    expanding_mean = y.expanding().mean().shift()

                    cumsum, cumcount = self._count_per_segment_cumstats(y.values, int_categories)
                    cumsum = pd.Series(cumsum)
                    cumcount = pd.Series(cumcount)

                    feature = (cumsum + expanding_mean * self.smoothing) / (cumcount + self.smoothing)
                    if self.handle_missing is MissingMode.global_mean:
                        nan_feature_index = segment_df[segment_df[self.in_column].isnull()].index
                        feature.loc[nan_feature_index] = expanding_mean.loc[nan_feature_index]

                    intersected_df.loc[:, self.idx[segment, self.out_column]] = feature.values

            else:
                flatten = TSDataset.to_flatten(intersected_df)
                flatten = flatten.sort_values(["timestamp", "segment"])
                running_mean = self._count_macro_running_mean(flatten, n_segments)

                temp = pd.DataFrame(index=flatten.index, columns=["cumsum", "cumcount"], dtype=float)

                timestamps = intersected_df.index
                categories = pd.unique(df.loc[:, self.idx[:, self.in_column]].values.ravel())

                cumstats = pd.DataFrame(data={"sum": np.NaN, "count": np.NaN, self.in_column: categories})
                cur_timestamp_idx = np.arange(0, len(timestamps) * n_segments, len(timestamps))
                for _ in range(len(timestamps)):
                    timestamp_df = flatten.loc[cur_timestamp_idx]

                    # statistics from previous timestamp
                    cumsum_dict = dict(cumstats[[self.in_column, "sum"]].values)
                    cumcount_dict = dict(cumstats[[self.in_column, "count"]].values)

                    # map categories for current timestamp to statistics
                    temp.loc[cur_timestamp_idx, "cumsum"] = timestamp_df[self.in_column].map(cumsum_dict)
                    temp.loc[cur_timestamp_idx, "cumcount"] = timestamp_df[self.in_column].map(cumcount_dict)

                    # count statistics for current timestamp
                    stats = (
                        timestamp_df["target"]
                        .groupby(timestamp_df[self.in_column], dropna=False)
                        .agg(["count", "sum"])
                        .reset_index()
                    )
                    # statistics become zeros for categories with target=NaN
                    stats = stats.replace({"count": 0, "sum": 0}, np.NaN)

                    # sum current and previous statistics
                    cumstats = pd.concat([cumstats, stats]).groupby(self.in_column, as_index=False, dropna=False).sum()
                    # zeros appear for categories that weren't updated in previous line and whose statistics were NaN
                    cumstats = cumstats.replace({"count": 0, "sum": 0}, np.NaN)

                    cur_timestamp_idx += 1

                feature = (temp["cumsum"] + running_mean * self.smoothing) / (temp["cumcount"] + self.smoothing)
                if self.handle_missing is MissingMode.global_mean:
                    nan_feature_index = flatten[flatten[self.in_column].isnull()].index
                    feature.loc[nan_feature_index] = running_mean.loc[nan_feature_index]

                feature = pd.DataFrame(
                    feature.values.reshape(len(timestamps), n_segments),
                    columns=pd.MultiIndex.from_product([segments, [self.out_column]], names=df.columns.names),
                    index=intersected_df.index,
                )
                intersected_df = pd.concat([intersected_df, feature], axis=1)

        if len(future_df) > 0:
            n_timestamps = len(future_df.index)
            if self.mode is EncoderMode.per_segment:
                self._global_means_category = cast(Dict[str, Dict[str, float]], self._global_means_category)
                self._global_means = cast(Dict[str, float], self._global_means)
                for segment in segments:
                    segment_df = TSDataset.to_flatten(future_df.loc[:, self.idx[segment, :]])
                    feature = segment_df[self.in_column].map(self._global_means_category[segment])
                    feature = feature.fillna(self._global_means[segment])
                    future_df.loc[:, self.idx[segment, self.out_column]] = feature.values
            else:
                flatten = TSDataset.to_flatten(future_df)
                feature = flatten[self.in_column].map(self._global_means_category)
                feature = feature.fillna(self._global_means)
                feature = pd.DataFrame(
                    feature.values.reshape(len(segments), n_timestamps).T,
                    columns=pd.MultiIndex.from_product([segments, [self.out_column]], names=df.columns.names),
                    index=future_df.index,
                )
                future_df = pd.concat([future_df, feature], axis=1)

        intersected_df = intersected_df.sort_index(axis=1)
        future_df = future_df.sort_index(axis=1)
        transformed_df = pd.concat((intersected_df, future_df), axis=0)
        return transformed_df

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return [self.out_column]

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes ``smoothing`` parameter. Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {"smoothing": FloatDistribution(low=0, high=2)}
