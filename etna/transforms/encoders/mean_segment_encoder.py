from typing import List

import numpy as np
import pandas as pd

from etna.transforms import IrreversibleTransform
from etna.transforms.encoders.mean_encoder import MeanEncoderTransform


class MeanSegmentEncoderTransform(IrreversibleTransform):
    """Makes expanding mean target encoding of the segment. Creates column 'segment_mean'."""

    _segment_column = "segment_column"
    out_column = "segment_mean"

    def __init__(self):
        super().__init__(required_features=["target"])
        self._mean_encoder = MeanEncoderTransform(
            in_column=self._segment_column, mode="per-segment", out_column=self.out_column, smoothing=0
        )

    def _add_segment_column(self, df):
        segments = df.columns.get_level_values("segment").unique()
        flatten_segments = np.repeat(segments.values[np.newaxis, :], len(df), axis=0)
        segment_values = pd.DataFrame(
            data=flatten_segments,
            columns=pd.MultiIndex.from_product([segments, [self._segment_column]], names=df.columns.names),
            index=df.index,
        )
        df = pd.concat([df, segment_values], axis=1).sort_index(axis=1)
        return df

    def _fit(self, df: pd.DataFrame) -> "MeanSegmentEncoderTransform":
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
        df = self._add_segment_column(df)
        self._mean_encoder._fit(df)
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
        df = self._add_segment_column(df)
        df_transformed = self._mean_encoder._transform(df)
        df_transformed = df_transformed.drop(columns=[self._segment_column], level="feature")
        return df_transformed

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return [self.out_column]
