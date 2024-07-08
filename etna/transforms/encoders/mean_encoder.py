import reprlib
from enum import Enum
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from bottleneck import nanmean

from etna.datasets import TSDataset
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

    For timestamps that were seen in ``fit`` transformations are made using the formula below:

    .. math::
        \\frac{TargetSum + GlobalMean * Smoothing}{FeatureCount + Smoothing}
    where
        - TargetSum is the sum of target up to the current index for the current category, not including the current index
        - GlobalMean is target mean in the whole dataset
        - FeatureCount is the number of categories with the same value as in the current index, not including the current index

    For future timestamps:
        - for known categories encoding are filled with global mean of target for these categories calculated during ``fit``
        - for unknown categories encoding are filled with global mean of target in the whole dataset calculated during ``fit``

    NaNs in ``target`` are skipped.
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
        self.timestamps = set(df.index)

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
                if self.handle_missing is MissingMode.global_mean:
                    global_means_category[segment].pop(np.NaN, None)
        else:
            axis = None
            global_means = nanmean(df.loc[:, self.idx[:, "target"]], axis=axis)

            segment_df = TSDataset.to_flatten(df)
            global_means_category = (
                segment_df[[self.in_column, "target"]].groupby(self.in_column, dropna=False).mean().to_dict()["target"]
            )
            if self.handle_missing is MissingMode.global_mean:
                global_means_category.pop(np.NaN, None)

        self.global_means = global_means
        self.global_means_category = global_means_category
        return self

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
        if self.global_means is None:
            raise ValueError("The transform isn't fitted!")

        segments = df.columns.get_level_values("segment").unique().tolist()
        n_segments = len(segments)
        if self.mode is EncoderMode.per_segment:
            new_segments = set(segments) - self.global_means.keys()
            if len(new_segments) > 0:
                raise NotImplementedError(
                    f"This transform can't process segments that weren't present on train data: {reprlib.repr(new_segments)}"
                )
        current_timestamps = set(df.index)
        intersected_timestamps = self.timestamps.intersection(current_timestamps)
        future_timestamps = current_timestamps - self.timestamps

        intersected_df = TSDataset.to_flatten(df.loc[list(intersected_timestamps)])
        intersected_df = intersected_df.sort_values(["timestamp", "segment"])
        intersected_df[self.out_column] = np.NaN

        future_df = TSDataset.to_flatten(df.loc[list(future_timestamps)])
        future_df = future_df.sort_values(["timestamp", "segment"])
        future_df[self.out_column] = np.NaN

        if len(intersected_df) > 0:
            if self.mode is EncoderMode.per_segment:
                for segment in segments:
                    segment_df = intersected_df[intersected_df.segment == segment]
                    y = segment_df["target"]
                    temp = y.groupby(segment_df[self.in_column].astype(str)).agg(["cumsum", "cumcount"])
                    feature = (temp["cumsum"] - y + self.global_means[segment] * self.smoothing) / (
                        temp["cumcount"] + self.smoothing
                    )
                    intersected_df.loc[segment_df.index, self.out_column] = feature
                    if self.handle_missing is MissingMode.global_mean:
                        nan_index = segment_df[segment_df[self.in_column].isnull()].index
                        expanding_mean = y.expanding().mean().shift().fillna(0)
                        intersected_df.loc[nan_index, self.out_column] = expanding_mean.loc[nan_index]
            else:
                temp = pd.DataFrame(index=intersected_df.index, columns=["cumsum", "cumcount"], dtype=float)

                timestamps = intersected_df["timestamp"].unique()
                categories = intersected_df[self.in_column].unique()
                cumstats = pd.DataFrame(data={"sum": 0, "count": 0, self.in_column: categories})
                for timestamp in timestamps:
                    timestamp_df = intersected_df[intersected_df["timestamp"] == timestamp]
                    timestamp_index = timestamp_df.index
                    cumsum_dict = dict(cumstats[[self.in_column, "sum"]].values)
                    cumcount_dict = dict(cumstats[[self.in_column, "count"]].values)
                    temp.loc[timestamp_index, "cumsum"] = intersected_df.loc[timestamp_index, self.in_column].map(
                        cumsum_dict
                    )
                    temp.loc[timestamp_index, "cumcount"] = intersected_df.loc[timestamp_index, self.in_column].map(
                        cumcount_dict
                    )

                    stats = (
                        timestamp_df["target"]
                        .groupby(timestamp_df[self.in_column], dropna=False)
                        .agg(["count", "sum"])
                        .reset_index()
                    )
                    cumstats = pd.concat([cumstats, stats]).groupby(self.in_column, as_index=False, dropna=False).sum()

                feature = (temp["cumsum"] + self.global_means * self.smoothing) / (temp["cumcount"] + self.smoothing)
                intersected_df[self.out_column] = feature

                if self.handle_missing is MissingMode.global_mean:
                    y = intersected_df["target"]
                    nan_index = intersected_df[intersected_df[self.in_column].isnull()].index
                    timestamp_count = y.groupby(intersected_df["timestamp"]).transform("count")
                    timestamp_sum = y.groupby(intersected_df["timestamp"]).transform("sum")
                    expanding_mean = (
                        timestamp_sum.iloc[::n_segments].cumsum() / timestamp_count.iloc[::n_segments].cumsum()
                    )
                    expanding_mean = expanding_mean.repeat(n_segments)
                    expanding_mean = (
                        pd.Series(index=intersected_df.index, data=expanding_mean.values).shift(n_segments).fillna(0)
                    )
                    intersected_df.loc[nan_index, self.out_column] = expanding_mean.loc[nan_index]
                nan_target_index = intersected_df[intersected_df["target"].isnull()].index
                intersected_df.loc[nan_target_index, self.out_column] = np.NaN

        if len(future_df) > 0:
            if self.mode is EncoderMode.per_segment:
                for segment in segments:
                    segment_df = future_df[future_df.segment == segment]
                    future_df.loc[segment_df.index, self.out_column] = segment_df.loc[
                        segment_df.index, self.in_column
                    ].map(self.global_means_category[segment])
                    future_df.loc[segment_df.index, self.out_column] = future_df.loc[
                        segment_df.index, self.out_column
                    ].fillna(self.global_means[segment])
            else:
                future_df[self.out_column] = future_df[self.in_column].map(self.global_means_category)
                future_df[self.out_column] = future_df[self.out_column].fillna(self.global_means)
        transformed_df = pd.concat((intersected_df, future_df))
        transformed_df = TSDataset.to_dataset(transformed_df)
        return transformed_df

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return [self.out_column]
