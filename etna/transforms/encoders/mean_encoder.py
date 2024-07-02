import reprlib
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import Set
from pandas._libs.tslibs.timestamps import Timestamp
import numpy as np
import pandas as pd
from bottleneck import nanmean

from etna.datasets import TSDataset
from etna.transforms import IrreversibleTransform


class ImputerMode(str, Enum):
    """Enum for different encoding strategies."""

    micro = "micro"
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
    """Makes expanding mean target encoding based on categorical column."""

    idx = pd.IndexSlice

    def __init__(
        self,
        in_column: str,
        out_column: str,
        mode: str = ImputerMode.micro,
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

            * 'micro' - running mean is calculated across each segment individually

            * 'macro' - running mean is calculated across all segments
        handle_missing:
            mode to handle missing values in ``in_column``

            * 'category' - NaNs they are interpreted as a separate categorical feature

            * 'global_mean' - NaNs are filled with global mean
        smoothing:
            smoothing parameter


        """
        super().__init__(required_features=["target", in_column])
        self.in_column = in_column
        self.out_column = out_column
        self.mode = ImputerMode(mode)
        self.handle_missing = MissingMode(handle_missing)
        self.smoothing = smoothing

        self.global_means: Optional[Union[float, Dict[str, float]]] = None
        self.global_means_category: Optional[Union[Dict[str, float], Dict[str, Dict[str, float]]]] = None
        self.timestamps: Optional[Set[Timestamp]]

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

        if self.mode is ImputerMode.micro:
            axis = 0
            segments = df.columns.get_level_values("segment").unique().tolist()
            global_means = nanmean(df.loc[:, self.idx[:, "target"]], axis=axis)
            global_means = dict(zip(segments, global_means))

            global_means_category = {}
            for segment in segments:
                segment_df = TSDataset.to_flatten(df.loc[:, pd.IndexSlice[segment, :]])
                global_means_category[segment] = segment_df[[self.in_column, "target"]].groupby(self.in_column, dropna=False).mean().to_dict()["target"]
                if self.handle_missing is MissingMode.global_mean:
                    global_means_category[segment].pop(np.NaN, None)
        else:
            axis = None
            global_means = nanmean(df.loc[:, self.idx[:, "target"]], axis=axis)

            segment_df = TSDataset.to_flatten(df)
            global_means_category = segment_df[[self.in_column, "target"]].groupby(self.in_column, dropna=False).mean().to_dict()["target"]
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
        if self.mode is ImputerMode.micro:
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
            if self.mode is ImputerMode.micro:
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
                y = intersected_df["target"]
                temp = y.groupby(intersected_df[self.in_column].astype(str)).agg(["cumsum", "cumcount"])
                feature = (temp["cumsum"] - y + self.global_means * self.smoothing) / (
                        temp["cumcount"] + self.smoothing
                )
                intersected_df[self.out_column] = feature
                if self.handle_missing is MissingMode.global_mean:
                    nan_index = intersected_df[intersected_df[self.in_column].isnull()].index
                    cumcount = np.array(range(n_segments, len(intersected_df) + n_segments, n_segments))
                    expanding_sum = y.groupby(intersected_df["timestamp"]).transform("sum")
                    expanding_mean = expanding_sum.iloc[::n_segments].cumsum() / cumcount
                    expanding_mean = expanding_mean.repeat(2)
                    expanding_mean = pd.Series(index=intersected_df.index, data=expanding_mean.values).shift(2).fillna(0)
                    intersected_df.loc[nan_index, self.out_column] = expanding_mean.loc[nan_index]

        if len(future_df) > 0:
            if self.mode is ImputerMode.micro:
                for segment in segments:
                    segment_df = future_df[future_df.segment == segment]
                    future_df[self.out_column] = segment_df[self.in_column].map(
                        self.global_means_category[segment]
                    )
                    future_df = future_df.fillna(self.global_means[segment])
            else:
                future_df[self.out_column] = future_df[self.in_column].map(self.global_means_category)
                future_df = future_df.fillna(self.global_means)
        transformed_df = pd.concat((intersected_df, future_df))
        transformed_df = TSDataset.to_dataset(transformed_df)
        return transformed_df

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return [self.out_column]
