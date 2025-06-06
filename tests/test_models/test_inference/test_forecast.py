from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from typing_extensions import get_args

from etna.datasets import TSDataset
from etna.models import AutoARIMAModel
from etna.models import BATSModel
from etna.models import CatBoostMultiSegmentModel
from etna.models import CatBoostPerSegmentModel
from etna.models import ContextRequiredModelType
from etna.models import DeadlineMovingAverageModel
from etna.models import ElasticMultiSegmentModel
from etna.models import ElasticPerSegmentModel
from etna.models import HoltModel
from etna.models import HoltWintersModel
from etna.models import LinearMultiSegmentModel
from etna.models import LinearPerSegmentModel
from etna.models import MovingAverageModel
from etna.models import NaiveModel
from etna.models import ProphetModel
from etna.models import SARIMAXModel
from etna.models import SeasonalMovingAverageModel
from etna.models import SimpleExpSmoothingModel
from etna.models import StatsForecastARIMAModel
from etna.models import StatsForecastAutoARIMAModel
from etna.models import StatsForecastAutoCESModel
from etna.models import StatsForecastAutoETSModel
from etna.models import StatsForecastAutoThetaModel
from etna.models import TBATSModel
from etna.models.nn import ChronosBoltModel
from etna.models.nn import ChronosModel
from etna.models.nn import DeepARModel
from etna.models.nn import DeepStateModel
from etna.models.nn import MLPModel
from etna.models.nn import NBeatsGenericModel
from etna.models.nn import NBeatsInterpretableModel
from etna.models.nn import PatchTSTModel
from etna.models.nn import RNNModel
from etna.models.nn import TFTModel
from etna.models.nn import TimesFMModel
from etna.models.nn.deepstate import CompositeSSM
from etna.models.nn.deepstate import WeeklySeasonalitySSM
from etna.transforms import LagTransform
from etna.transforms import SegmentEncoderTransform
from tests.test_models.test_inference.common import _test_prediction_in_sample_full
from tests.test_models.test_inference.common import _test_prediction_in_sample_suffix
from tests.test_models.test_inference.common import make_prediction
from tests.utils import convert_ts_to_int_timestamp
from tests.utils import select_segments_subset
from tests.utils import to_be_fixed


def make_forecast(model, ts, prediction_size) -> TSDataset:
    return make_prediction(model=model, ts=ts, prediction_size=prediction_size, method_name="forecast")


class TestForecastInSampleFullNoTarget:
    """Test forecast on full train dataset where target is filled with NaNs.

    Expected that NaNs are filled after prediction.
    """

    @staticmethod
    def _test_forecast_in_sample_full_no_target(ts, model, transforms):
        forecast_ts = deepcopy(ts)

        # fitting
        ts.fit_transform(transforms)
        model.fit(ts)

        # forecasting
        forecast_ts.transform(transforms)
        forecast_ts._df.loc[:, pd.IndexSlice[:, "target"]] = np.NaN
        prediction_size = len(forecast_ts.timestamps)
        forecast_ts = make_forecast(model=model, ts=forecast_ts, prediction_size=prediction_size)

        # checking
        forecast_df = forecast_ts.to_pandas(flatten=True)
        assert not np.any(forecast_df["target"].isna())

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (ProphetModel(), [], "example_tsds"),
            (ProphetModel(timestamp_column="external_timestamp"), [], "ts_with_external_timestamp"),
            (SARIMAXModel(), [], "example_tsds"),
            (AutoARIMAModel(), [], "example_tsds"),
            (HoltModel(), [], "example_tsds"),
            (HoltWintersModel(), [], "example_tsds"),
            (SimpleExpSmoothingModel(), [], "example_tsds"),
        ],
    )
    def test_forecast_in_sample_full_no_target(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_forecast_in_sample_full_no_target(ts, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
        ],
    )
    def test_forecast_in_sample_full_no_target_failed_nans_sklearn(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(ValueError, match="Input X contains NaN."):
            self._test_forecast_in_sample_full_no_target(ts, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (MovingAverageModel(window=3), [], "example_tsds"),
            (NaiveModel(lag=3), [], "example_tsds"),
            (SeasonalMovingAverageModel(), [], "example_tsds"),
            (DeadlineMovingAverageModel(window=1), [], "example_tsds"),
            (
                RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                DeepARModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                TFTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (PatchTSTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
                "example_tsds",
            ),
            (
                NBeatsInterpretableModel(input_size=1, output_size=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (NBeatsGenericModel(input_size=1, output_size=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
        ],
    )
    def test_forecast_in_sample_full_no_target_failed_not_enough_context(
        self, model, transforms, dataset_name, request
    ):
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(ValueError, match="Given context isn't big enough"):
            self._test_forecast_in_sample_full_no_target(ts, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_in_sample_full_no_target_failed_nans_nn(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(ValueError, match="There are NaNs in features"):
            self._test_forecast_in_sample_full_no_target(ts, model, transforms)

    @to_be_fixed(raises=NotImplementedError, match="This model can't make forecast on history data")
    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (BATSModel(use_trend=True), [], "example_tsds"),
            (TBATSModel(use_trend=True), [], "example_tsds"),
            (StatsForecastARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoCESModel(), [], "example_tsds"),
            (StatsForecastAutoETSModel(), [], "example_tsds"),
            (StatsForecastAutoThetaModel(), [], "example_tsds"),
        ],
    )
    def test_forecast_in_sample_full_no_target_failed_not_implemented_in_sample(
        self, model, transforms, dataset_name, request
    ):
        ts = request.getfixturevalue(dataset_name)
        self._test_forecast_in_sample_full_no_target(ts, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=7), [], "example_tsds"),
            (ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=7), [], "example_tsds"),
        ],
    )
    def test_forecast_in_sample_full_no_target_failed_chronos(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(ValueError, match="Dataset doesn't have any context timestamps."):
            self._test_forecast_in_sample_full_no_target(ts, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (
                lambda: TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=32),
                [],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_in_sample_full_no_target_failed_timesfm(self, model, transforms, dataset_name, request):
        if callable(model):
            model = model()
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(ValueError, match="Dataset doesn't have any context timestamps."):
            self._test_forecast_in_sample_full_no_target(ts, model, transforms)


class TestForecastInSampleFull:
    """Test forecast on full train dataset.

    Expected that there are no NaNs after prediction and targets are changed compared to original.
    """

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (ProphetModel(), [], "example_tsds"),
            (ProphetModel(timestamp_column="external_timestamp"), [], "ts_with_external_timestamp"),
            (SARIMAXModel(), [], "example_tsds"),
            (AutoARIMAModel(), [], "example_tsds"),
            (HoltModel(), [], "example_tsds"),
            (HoltWintersModel(), [], "example_tsds"),
            (SimpleExpSmoothingModel(), [], "example_tsds"),
        ],
    )
    def test_forecast_in_sample_full(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        _test_prediction_in_sample_full(ts, model, transforms, method_name="forecast")

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
        ],
    )
    def test_forecast_in_sample_full_failed_nans_sklearn(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(ValueError, match="Input X contains NaN"):
            _test_prediction_in_sample_full(ts, model, transforms, method_name="forecast")

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[2, 3])],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_in_sample_full_failed_nans_nn(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(ValueError, match="There are NaNs in features"):
            _test_prediction_in_sample_full(ts, model, transforms, method_name="forecast")

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (MovingAverageModel(window=3), [], "example_tsds"),
            (NaiveModel(lag=3), [], "example_tsds"),
            (SeasonalMovingAverageModel(), [], "example_tsds"),
            (DeadlineMovingAverageModel(window=1), [], "example_tsds"),
            (
                RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                DeepARModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                TFTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (PatchTSTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
                "example_tsds",
            ),
            (
                NBeatsInterpretableModel(input_size=1, output_size=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (NBeatsGenericModel(input_size=1, output_size=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
        ],
    )
    def test_forecast_in_sample_full_failed_not_enough_context(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(ValueError, match="Given context isn't big enough"):
            _test_prediction_in_sample_full(ts, model, transforms, method_name="forecast")

    @to_be_fixed(raises=NotImplementedError, match="This model can't make forecast on history data")
    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (BATSModel(use_trend=True), [], "example_tsds"),
            (TBATSModel(use_trend=True), [], "example_tsds"),
            (StatsForecastARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoCESModel(), [], "example_tsds"),
            (StatsForecastAutoETSModel(), [], "example_tsds"),
            (StatsForecastAutoThetaModel(), [], "example_tsds"),
        ],
    )
    def test_forecast_in_sample_full_not_implemented(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        _test_prediction_in_sample_full(ts, model, transforms, method_name="forecast")

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=7), [], "example_tsds"),
            (ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=7), [], "example_tsds"),
        ],
    )
    def test_forecast_in_sample_full_failed_chronos(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(ValueError, match="Dataset doesn't have any context timestamps."):
            _test_prediction_in_sample_full(ts, model, transforms, method_name="forecast")

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (
                lambda: TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=32),
                [],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_in_sample_full_failed_timesfm(self, model, transforms, dataset_name, request):
        if callable(model):
            model = model()
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(ValueError, match="Dataset doesn't have any context timestamps."):
            _test_prediction_in_sample_full(ts, model, transforms, method_name="forecast")


class TestForecastInSampleSuffixNoTarget:
    """Test forecast on suffix of train dataset where target is filled with NaNs.

    Expected that NaNs are filled after prediction.
    """

    @staticmethod
    def _test_forecast_in_sample_suffix_no_target(ts, model, transforms, num_skip_points):
        forecast_ts = deepcopy(ts)

        # fitting
        ts.fit_transform(transforms)
        model.fit(ts)

        # forecasting
        forecast_ts.transform(transforms)
        forecast_ts._df.loc[forecast_ts.timestamps[num_skip_points] :, pd.IndexSlice[:, "target"]] = np.NaN
        prediction_size = len(forecast_ts.timestamps) - num_skip_points
        forecast_ts._df = forecast_ts._df.iloc[(num_skip_points - model.context_size) :]
        forecast_ts = make_forecast(model=model, ts=forecast_ts, prediction_size=prediction_size)

        # checking
        forecast_df = forecast_ts.to_pandas(flatten=True)
        assert not np.any(forecast_df["target"].isna())

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (ProphetModel(), [], "example_tsds"),
            (ProphetModel(timestamp_column="external_timestamp"), [], "ts_with_external_timestamp"),
            (SARIMAXModel(), [], "example_tsds"),
            (AutoARIMAModel(), [], "example_tsds"),
            (HoltModel(), [], "example_tsds"),
            (HoltWintersModel(), [], "example_tsds"),
            (SimpleExpSmoothingModel(), [], "example_tsds"),
            (MovingAverageModel(window=3), [], "example_tsds"),
            (NaiveModel(lag=3), [], "example_tsds"),
            (SeasonalMovingAverageModel(), [], "example_tsds"),
            (DeadlineMovingAverageModel(window=1), [], "example_tsds"),
            (
                RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                DeepARModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                TFTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (PatchTSTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[2, 3])],
                "example_tsds",
            ),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
                "example_tsds",
            ),
            (
                NBeatsInterpretableModel(input_size=7, output_size=50, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (NBeatsGenericModel(input_size=7, output_size=50, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=7), [], "example_tsds"),
            (ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=7), [], "example_tsds"),
            (
                lambda: TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=32),
                [],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_in_sample_suffix_no_target(self, model, transforms, dataset_name, request):
        if callable(model):
            model = model()
        ts = request.getfixturevalue(dataset_name)
        self._test_forecast_in_sample_suffix_no_target(ts, model, transforms, num_skip_points=50)

    @to_be_fixed(raises=NotImplementedError, match="This model can't make forecast on history data")
    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (BATSModel(use_trend=True), [], "example_tsds"),
            (TBATSModel(use_trend=True), [], "example_tsds"),
            (StatsForecastARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoCESModel(), [], "example_tsds"),
            (StatsForecastAutoETSModel(), [], "example_tsds"),
            (StatsForecastAutoThetaModel(), [], "example_tsds"),
        ],
    )
    def test_forecast_in_sample_suffix_no_target_failed_not_implemented_in_sample(
        self, model, transforms, dataset_name, request
    ):
        ts = request.getfixturevalue(dataset_name)
        self._test_forecast_in_sample_suffix_no_target(ts, model, transforms, num_skip_points=50)


class TestForecastInSampleSuffix:
    """Test forecast on suffix of train dataset.

    Expected that there are no NaNs after prediction and targets are changed compared to original.
    """

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[2, 3])], "example_tsds"),
            (ProphetModel(), [], "example_tsds"),
            (ProphetModel(timestamp_column="external_timestamp"), [], "ts_with_external_timestamp"),
            (SARIMAXModel(), [], "example_tsds"),
            (AutoARIMAModel(), [], "example_tsds"),
            (HoltModel(), [], "example_tsds"),
            (HoltWintersModel(), [], "example_tsds"),
            (SimpleExpSmoothingModel(), [], "example_tsds"),
            (MovingAverageModel(window=3), [], "example_tsds"),
            (NaiveModel(lag=3), [], "example_tsds"),
            (SeasonalMovingAverageModel(), [], "example_tsds"),
            (DeadlineMovingAverageModel(window=1), [], "example_tsds"),
            (
                RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                DeepARModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                TFTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (PatchTSTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[2, 3])],
                "example_tsds",
            ),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
                "example_tsds",
            ),
            (
                NBeatsInterpretableModel(input_size=7, output_size=50, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (NBeatsGenericModel(input_size=7, output_size=50, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=7), [], "example_tsds"),
            (ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=7), [], "example_tsds"),
            (
                lambda: TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=32),
                [],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_in_sample_suffix(self, model, transforms, dataset_name, request):
        if callable(model):
            model = model()
        ts = request.getfixturevalue(dataset_name)
        _test_prediction_in_sample_suffix(ts, model, transforms, method_name="forecast", num_skip_points=50)

    @to_be_fixed(raises=NotImplementedError, match="This model can't make forecast on history data")
    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (BATSModel(use_trend=True), [], "example_tsds"),
            (TBATSModel(use_trend=True), [], "example_tsds"),
            (StatsForecastARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoCESModel(), [], "example_tsds"),
            (StatsForecastAutoETSModel(), [], "example_tsds"),
            (StatsForecastAutoThetaModel(), [], "example_tsds"),
        ],
    )
    def test_forecast_in_sample_suffix_failed_not_implemented_in_sample(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        _test_prediction_in_sample_suffix(ts, model, transforms, method_name="forecast", num_skip_points=50)


class TestForecastOutSample:
    """Test forecast on of future dataset.

    Expected that NaNs are filled after prediction.
    """

    @staticmethod
    def _test_forecast_out_sample(ts, model, transforms, prediction_size=5):
        # fitting
        ts.fit_transform(transforms)
        model.fit(ts)

        # forecasting
        forecast_ts = ts.make_future(future_steps=prediction_size, tail_steps=model.context_size, transforms=transforms)
        forecast_ts = make_forecast(model=model, ts=forecast_ts, prediction_size=prediction_size)

        # checking
        forecast_df = forecast_ts.to_pandas(flatten=True)
        assert not np.any(forecast_df["target"].isna())

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (AutoARIMAModel(), [], "example_tsds"),
            (ProphetModel(), [], "example_tsds"),
            (ProphetModel(timestamp_column="external_timestamp"), [], "ts_with_external_timestamp"),
            (SARIMAXModel(), [], "example_tsds"),
            (HoltModel(), [], "example_tsds"),
            (HoltWintersModel(), [], "example_tsds"),
            (SimpleExpSmoothingModel(), [], "example_tsds"),
            (MovingAverageModel(window=3), [], "example_tsds"),
            (SeasonalMovingAverageModel(), [], "example_tsds"),
            (NaiveModel(lag=3), [], "example_tsds"),
            (DeadlineMovingAverageModel(window=1), [], "example_tsds"),
            (BATSModel(use_trend=True), [], "example_tsds"),
            (TBATSModel(use_trend=True), [], "example_tsds"),
            (StatsForecastARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoCESModel(), [], "example_tsds"),
            (StatsForecastAutoETSModel(), [], "example_tsds"),
            (StatsForecastAutoThetaModel(), [], "example_tsds"),
            (
                RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                DeepARModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                TFTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (PatchTSTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
                "example_tsds",
            ),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
                "example_tsds",
            ),
            (
                NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=7), [], "example_tsds"),
            (ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=7), [], "example_tsds"),
            (
                lambda: TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=32),
                [],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_out_sample_datetime_timestamp(self, model, transforms, dataset_name, request):
        if callable(model):
            model = model()
        ts = request.getfixturevalue(dataset_name)
        self._test_forecast_out_sample(ts, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (AutoARIMAModel(), [], "example_tsds"),
            (ProphetModel(timestamp_column="external_timestamp"), [], "ts_with_external_timestamp"),
            (SARIMAXModel(), [], "example_tsds"),
            (HoltModel(), [], "example_tsds"),
            (HoltWintersModel(), [], "example_tsds"),
            (SimpleExpSmoothingModel(), [], "example_tsds"),
            (MovingAverageModel(window=3), [], "example_tsds"),
            (SeasonalMovingAverageModel(), [], "example_tsds"),
            (NaiveModel(lag=3), [], "example_tsds"),
            (BATSModel(use_trend=True), [], "example_tsds"),
            (TBATSModel(use_trend=True), [], "example_tsds"),
            (StatsForecastARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoCESModel(), [], "example_tsds"),
            (StatsForecastAutoETSModel(), [], "example_tsds"),
            (StatsForecastAutoThetaModel(), [], "example_tsds"),
            (
                RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                DeepARModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                TFTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (PatchTSTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
                "example_tsds",
            ),
            (
                NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=7), [], "example_tsds"),
            (ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=7), [], "example_tsds"),
            (
                lambda: TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=32),
                [],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_out_sample_int_timestamp(self, model, transforms, dataset_name, request):
        if callable(model):
            model = model()
        ts = request.getfixturevalue(dataset_name)
        ts_int_timestamp = convert_ts_to_int_timestamp(ts, shift=10)
        self._test_forecast_out_sample(ts_int_timestamp, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (ProphetModel(), [], "example_tsds"),
        ],
    )
    def test_forecast_out_sample_int_timestamp_not_supported(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        ts_int_timestamp = convert_ts_to_int_timestamp(ts, shift=10)
        with pytest.raises(ValueError, match="Invalid timestamp! Only datetime type is supported."):
            self._test_forecast_out_sample(ts_int_timestamp, model, transforms)

    @to_be_fixed(raises=Exception)
    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (DeadlineMovingAverageModel(window=1), [], "example_tsds"),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_out_sample_int_timestamp_failed(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        ts_int_timestamp = convert_ts_to_int_timestamp(ts, shift=10)
        self._test_forecast_out_sample(ts_int_timestamp, model, transforms)


class TestForecastOutSamplePrefix:
    """Test forecast on prefix of future dataset.

    Expected that predictions on prefix match prefix of predictions on full future dataset.
    """

    @staticmethod
    def _test_forecast_out_sample_prefix(ts, model, transforms, full_prediction_size=5, prefix_prediction_size=3):
        prediction_size_diff = full_prediction_size - prefix_prediction_size

        # fitting
        ts.fit_transform(transforms)
        model.fit(ts)

        # forecasting full
        forecast_full_ts = ts.make_future(
            future_steps=full_prediction_size, tail_steps=model.context_size, transforms=transforms
        )
        forecast_full_ts = make_forecast(model=model, ts=forecast_full_ts, prediction_size=full_prediction_size)

        # forecasting only prefix
        forecast_prefix_ts = ts.make_future(
            future_steps=full_prediction_size, tail_steps=model.context_size, transforms=transforms
        )
        forecast_prefix_ts._df = forecast_prefix_ts._df.iloc[:-prediction_size_diff]
        forecast_prefix_ts = make_forecast(model=model, ts=forecast_prefix_ts, prediction_size=prefix_prediction_size)

        # checking
        forecast_full_df = forecast_full_ts.to_pandas()
        forecast_prefix_df = forecast_prefix_ts.to_pandas()
        assert_frame_equal(forecast_prefix_df, forecast_full_df.iloc[:prefix_prediction_size])

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (AutoARIMAModel(), [], "example_tsds"),
            (ProphetModel(), [], "example_tsds"),
            (ProphetModel(timestamp_column="external_timestamp"), [], "ts_with_external_timestamp"),
            (SARIMAXModel(), [], "example_tsds"),
            (HoltModel(), [], "example_tsds"),
            (HoltWintersModel(), [], "example_tsds"),
            (SimpleExpSmoothingModel(), [], "example_tsds"),
            (MovingAverageModel(window=3), [], "example_tsds"),
            (SeasonalMovingAverageModel(), [], "example_tsds"),
            (NaiveModel(lag=3), [], "example_tsds"),
            (DeadlineMovingAverageModel(window=1), [], "example_tsds"),
            (BATSModel(use_trend=True), [], "example_tsds"),
            (TBATSModel(use_trend=True), [], "example_tsds"),
            (StatsForecastARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoCESModel(), [], "example_tsds"),
            (StatsForecastAutoETSModel(), [], "example_tsds"),
            (StatsForecastAutoThetaModel(), [], "example_tsds"),
            (
                RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                DeepARModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                TFTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (PatchTSTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
                "example_tsds",
            ),
            (
                NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=7), [], "example_tsds"),
            (ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=7), [], "example_tsds"),
            (
                lambda: TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=32),
                [],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_out_sample_prefix(self, model, transforms, dataset_name, request):
        if callable(model):
            model = model()
        ts = request.getfixturevalue(dataset_name)
        self._test_forecast_out_sample_prefix(ts, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
                "example_tsds",
            )
        ],
    )
    def test_forecast_out_sample_prefix_failed_deep_state(self, model, transforms, dataset_name, request):
        """This test is expected to fail due to sampling procedure of DeepStateModel"""
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(AssertionError):
            self._test_forecast_out_sample_prefix(ts, model, transforms)


class TestForecastOutSampleSuffix:
    """Test forecast on suffix of future dataset.

    Expected that predictions on suffix match suffix of predictions on full future dataset.
    """

    @staticmethod
    def _test_forecast_out_sample_suffix(ts, model, transforms, full_prediction_size=5, suffix_prediction_size=3):
        prediction_size_diff = full_prediction_size - suffix_prediction_size

        # fitting
        ts.fit_transform(transforms)
        model.fit(ts)

        # forecasting full
        forecast_full_ts = ts.make_future(
            future_steps=full_prediction_size, tail_steps=model.context_size, transforms=transforms
        )
        forecast_full_ts = make_forecast(model=model, ts=forecast_full_ts, prediction_size=full_prediction_size)

        # forecasting only suffix
        forecast_gap_ts = ts.make_future(
            future_steps=full_prediction_size, tail_steps=model.context_size, transforms=transforms
        )
        if isinstance(model, get_args(ContextRequiredModelType)):
            # firstly we should forecast prefix to use it as a context
            forecast_prefix_ts = deepcopy(forecast_gap_ts)
            forecast_prefix_ts._df = forecast_prefix_ts._df.iloc[:-suffix_prediction_size]
            forecast_prefix_ts = model.forecast(forecast_prefix_ts, prediction_size=prediction_size_diff)
            forecast_gap_ts._df = forecast_gap_ts._df.combine_first(forecast_prefix_ts._df)

            # forecast suffix with known context for it
            forecast_gap_ts = model.forecast(forecast_gap_ts, prediction_size=suffix_prediction_size)
        else:
            forecast_gap_ts._df = forecast_gap_ts._df.iloc[prediction_size_diff:]
            forecast_gap_ts = model.forecast(forecast_gap_ts)

        # checking
        forecast_full_df = forecast_full_ts.to_pandas()
        forecast_gap_df = forecast_gap_ts.to_pandas()
        assert_frame_equal(forecast_gap_df, forecast_full_df.iloc[prediction_size_diff:])

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (AutoARIMAModel(), [], "example_tsds"),
            (ProphetModel(), [], "example_tsds"),
            (ProphetModel(timestamp_column="external_timestamp"), [], "ts_with_external_timestamp"),
            (SARIMAXModel(), [], "example_tsds"),
            (HoltModel(), [], "example_tsds"),
            (HoltWintersModel(), [], "example_tsds"),
            (SimpleExpSmoothingModel(), [], "example_tsds"),
            (BATSModel(use_trend=True), [], "example_tsds"),
            (TBATSModel(use_trend=True), [], "example_tsds"),
            (MovingAverageModel(window=3), [], "example_tsds"),
            (SeasonalMovingAverageModel(), [], "example_tsds"),
            (NaiveModel(lag=3), [], "example_tsds"),
            (DeadlineMovingAverageModel(window=1), [], "example_tsds"),
            (PatchTSTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_out_sample_suffix(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_forecast_out_sample_suffix(ts, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (
                RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_out_sample_suffix_failed_rnn(self, model, transforms, dataset_name, request):
        """This test is expected to fail due to autoregression in RNN.

        More about it in issue: https://github.com/tinkoff-ai/etna/issues/1087
        """
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(AssertionError):
            self._test_forecast_out_sample_suffix(ts, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (
                DeepARModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_out_sample_suffix_failed_deepar(self, model, transforms, dataset_name, request):
        """This test is expected to fail due to autoregression in DeepAR."""
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(AssertionError):
            self._test_forecast_out_sample_suffix(ts, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (
                TFTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_out_sample_suffix_failed_tft(self, model, transforms, dataset_name, request):
        """This test is expected to fail due to encoder-decoder structure of TFT."""
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(AssertionError):
            self._test_forecast_out_sample_suffix(ts, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
                "example_tsds",
            )
        ],
    )
    def test_forecast_out_sample_suffix_failed_deep_state(self, model, transforms, dataset_name, request):
        """This test is expected to fail due to sampling procedure of DeepStateModel"""
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(AssertionError):
            self._test_forecast_out_sample_suffix(ts, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        (
            (
                NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
        ),
    )
    def test_forecast_out_sample_suffix_failed_nbeats(self, model, transforms, dataset_name, request):
        """This test is expected to fail due to windowed view on data in N-BEATS"""
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(AssertionError):
            self._test_forecast_out_sample_suffix(ts, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=7), [], "example_tsds"),
            (ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=7), [], "example_tsds"),
        ],
    )
    def test_forecast_out_sample_suffix_failed_chronos(self, model, transforms, dataset_name, request):
        """This test is expected to fail due to autoregression in Chronos"""
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(AssertionError):
            self._test_forecast_out_sample_suffix(ts, model, transforms)

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (
                lambda: TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=32),
                [],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_out_sample_suffix_failed_timesfm(self, model, transforms, dataset_name, request):
        """This test is expected to fail due to patch strategy in TimesFM"""
        if callable(model):
            model = model()
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(AssertionError):
            self._test_forecast_out_sample_suffix(ts, model, transforms)

    @to_be_fixed(
        raises=NotImplementedError,
        match="This model can't make forecast on out-of-sample data that goes after training data with a gap",
    )
    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (StatsForecastARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoCESModel(), [], "example_tsds"),
            (StatsForecastAutoETSModel(), [], "example_tsds"),
            (StatsForecastAutoThetaModel(), [], "example_tsds"),
        ],
    )
    def test_forecast_out_sample_suffix_failed_not_implemented(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        self._test_forecast_out_sample_suffix(ts, model, transforms)


class TestForecastMixedInOutSample:
    """Test forecast on mixture of in-sample and out-sample.

    Expected that there are no NaNs after prediction and targets are changed compared to original.
    """

    @staticmethod
    def _test_forecast_mixed_in_out_sample(ts, model, transforms, num_skip_points=50, future_prediction_size=5):
        # fitting
        ts.fit_transform(transforms)
        model.fit(ts)

        # forecasting mixed in-sample and out-sample
        future_ts = ts.make_future(future_steps=future_prediction_size, transforms=transforms)
        df_full = pd.concat((ts.to_pandas(), future_ts.to_pandas()), axis=0)
        if ts._df_exog is not None:
            df_full.drop(columns=ts._df_exog.columns, inplace=True)

        forecast_full_ts = TSDataset(df=df_full, df_exog=ts._df_exog, freq=ts.freq, known_future=ts.known_future)
        forecast_full_ts._df = forecast_full_ts._df.iloc[(num_skip_points - model.context_size) :]
        full_prediction_size = len(forecast_full_ts.timestamps) - model.context_size
        forecast_full_ts = make_forecast(model=model, ts=forecast_full_ts, prediction_size=full_prediction_size)

        # checking
        forecast_full_df = forecast_full_ts.to_pandas(flatten=True)
        assert not np.any(forecast_full_df["target"].isna())
        original_target = TSDataset.to_flatten(df_full.iloc[(num_skip_points - model.context_size) :])["target"]
        assert not forecast_full_df["target"].equals(original_target)

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (AutoARIMAModel(), [], "example_tsds"),
            (ProphetModel(), [], "example_tsds"),
            (ProphetModel(timestamp_column="external_timestamp"), [], "ts_with_external_timestamp"),
            (SARIMAXModel(), [], "example_tsds"),
            (HoltModel(), [], "example_tsds"),
            (HoltWintersModel(), [], "example_tsds"),
            (SimpleExpSmoothingModel(), [], "example_tsds"),
            (MovingAverageModel(window=3), [], "example_tsds"),
            (SeasonalMovingAverageModel(), [], "example_tsds"),
            (NaiveModel(lag=3), [], "example_tsds"),
            (DeadlineMovingAverageModel(window=1), [], "example_tsds"),
            (
                RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                DeepARModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                TFTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (PatchTSTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
                "example_tsds",
            ),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
                "example_tsds",
            ),
            (
                NBeatsInterpretableModel(input_size=7, output_size=55, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (NBeatsGenericModel(input_size=7, output_size=55, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=7), [], "example_tsds"),
            (ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=7), [], "example_tsds"),
            (
                lambda: TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=32),
                [],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_mixed_in_out_sample(self, model, transforms, dataset_name, request):
        if callable(model):
            model = model()
        ts = request.getfixturevalue(dataset_name)
        self._test_forecast_mixed_in_out_sample(ts, model, transforms)

    @to_be_fixed(raises=NotImplementedError, match="This model can't make forecast on history data")
    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (BATSModel(use_trend=True), [], "example_tsds"),
            (TBATSModel(use_trend=True), [], "example_tsds"),
            (StatsForecastARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoCESModel(), [], "example_tsds"),
            (StatsForecastAutoETSModel(), [], "example_tsds"),
            (StatsForecastAutoThetaModel(), [], "example_tsds"),
        ],
    )
    def test_forecast_mixed_in_out_sample_failed_not_implemented_in_sample(
        self, model, transforms, dataset_name, request
    ):
        ts = request.getfixturevalue(dataset_name)
        self._test_forecast_mixed_in_out_sample(ts, model, transforms)


class TestForecastSubsetSegments:
    """Test forecast on subset of segments.

    Expected that predictions on subset of segments match subset of predictions on full dataset.
    """

    def _test_forecast_subset_segments(self, ts, model, transforms, segments, prediction_size=5):
        # select subset of tsdataset
        segments = list(set(segments))
        subset_ts = select_segments_subset(ts=deepcopy(ts), segments=segments)

        # fitting
        ts.fit_transform(transforms)
        model.fit(ts)

        # forecasting full
        forecast_full_ts = ts.make_future(
            future_steps=prediction_size, tail_steps=model.context_size, transforms=transforms
        )
        forecast_full_ts = make_forecast(model=model, ts=forecast_full_ts, prediction_size=prediction_size)

        # forecasting subset of segments
        forecast_subset_ts = subset_ts.make_future(
            future_steps=prediction_size, tail_steps=model.context_size, transforms=transforms
        )
        forecast_subset_ts = make_forecast(model=model, ts=forecast_subset_ts, prediction_size=prediction_size)

        # checking
        forecast_full_df = forecast_full_ts.to_pandas()
        forecast_subset_df = forecast_subset_ts.to_pandas()
        assert_frame_equal(forecast_subset_df, forecast_full_df.loc[:, pd.IndexSlice[segments, :]])

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (AutoARIMAModel(), [], "example_tsds"),
            (ProphetModel(), [], "example_tsds"),
            (ProphetModel(timestamp_column="external_timestamp"), [], "ts_with_external_timestamp"),
            (SARIMAXModel(), [], "example_tsds"),
            (HoltModel(), [], "example_tsds"),
            (HoltWintersModel(), [], "example_tsds"),
            (SimpleExpSmoothingModel(), [], "example_tsds"),
            (MovingAverageModel(window=3), [], "example_tsds"),
            (SeasonalMovingAverageModel(), [], "example_tsds"),
            (NaiveModel(lag=3), [], "example_tsds"),
            (DeadlineMovingAverageModel(window=1), [], "example_tsds"),
            (BATSModel(use_trend=True), [], "example_tsds"),
            (TBATSModel(use_trend=True), [], "example_tsds"),
            (StatsForecastARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoCESModel(), [], "example_tsds"),
            (StatsForecastAutoETSModel(), [], "example_tsds"),
            (StatsForecastAutoThetaModel(), [], "example_tsds"),
            (
                RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                DeepARModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                TFTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (PatchTSTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
                "example_tsds",
            ),
            (
                NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=7), [], "example_tsds"),
            (
                lambda: TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=32),
                [],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_subset_segments(self, model, transforms, dataset_name, request):
        if callable(model):
            model = model()
        ts = request.getfixturevalue(dataset_name)
        self._test_forecast_subset_segments(ts, model, transforms, segments=["segment_1"])

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=1,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [SegmentEncoderTransform()],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_subset_segments_failed_deep_state(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(AssertionError):
            self._test_forecast_subset_segments(ts, model, transforms, segments=["segment_1"])

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=7), [], "example_tsds"),
        ],
    )
    def test_forecast_subset_segments_failed_chronos(self, model, transforms, dataset_name, request):
        """This test fails due to LLM non determinism on MacOS, but it passes on Linux."""
        ts = request.getfixturevalue(dataset_name)
        try:
            self._test_forecast_subset_segments(ts, model, transforms, segments=["segment_1"])
        except AssertionError:
            return

        return


class TestForecastNewSegments:
    """Test forecast on new segments.

    Expected that target values are filled after prediction.
    """

    def _test_forecast_new_segments(self, ts, model, transforms, train_segments, prediction_size=5):
        # create tsdataset with new segments
        train_segments = list(set(train_segments))
        forecast_segments = list(set(ts.segments) - set(train_segments))
        train_ts = select_segments_subset(ts=deepcopy(ts), segments=train_segments)
        test_ts = select_segments_subset(ts=deepcopy(ts), segments=forecast_segments)

        # fitting
        train_ts.fit_transform(transforms)
        model.fit(train_ts)

        # forecasting
        forecast_ts = test_ts.make_future(
            future_steps=prediction_size, tail_steps=model.context_size, transforms=transforms
        )
        forecast_ts = make_forecast(model=model, ts=forecast_ts, prediction_size=prediction_size)

        # checking
        forecast_df = forecast_ts.to_pandas(flatten=True)
        assert not np.any(forecast_df["target"].isna())

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (CatBoostMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (LinearMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (ElasticMultiSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (MovingAverageModel(window=3), [], "example_tsds"),
            (SeasonalMovingAverageModel(), [], "example_tsds"),
            (NaiveModel(lag=3), [], "example_tsds"),
            (DeadlineMovingAverageModel(window=1), [], "example_tsds"),
            (
                RNNModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                DeepARModel(input_size=1, encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (
                TFTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (PatchTSTModel(encoder_length=7, decoder_length=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (
                MLPModel(input_size=2, hidden_size=[10], decoder_length=7, trainer_params=dict(max_epochs=1)),
                [LagTransform(in_column="target", lags=[5, 6])],
                "example_tsds",
            ),
            (
                DeepStateModel(
                    ssm=CompositeSSM(seasonal_ssms=[WeeklySeasonalitySSM()]),
                    input_size=0,
                    encoder_length=7,
                    decoder_length=7,
                    trainer_params=dict(max_epochs=1),
                ),
                [],
                "example_tsds",
            ),
            (
                NBeatsInterpretableModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)),
                [],
                "example_tsds",
            ),
            (NBeatsGenericModel(input_size=7, output_size=7, trainer_params=dict(max_epochs=1)), [], "example_tsds"),
            (ChronosModel(path_or_url="amazon/chronos-t5-tiny", encoder_length=7), [], "example_tsds"),
            (ChronosBoltModel(path_or_url="amazon/chronos-bolt-tiny", encoder_length=7), [], "example_tsds"),
            (
                lambda: TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=32),
                [],
                "example_tsds",
            ),
        ],
    )
    def test_forecast_new_segments(self, model, transforms, dataset_name, request):
        if callable(model):
            model = model()
        ts = request.getfixturevalue(dataset_name)
        self._test_forecast_new_segments(ts, model, transforms, train_segments=["segment_1"])

    @pytest.mark.parametrize(
        "model, transforms, dataset_name",
        [
            (CatBoostPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (LinearPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (ElasticPerSegmentModel(), [LagTransform(in_column="target", lags=[5, 6])], "example_tsds"),
            (AutoARIMAModel(), [], "example_tsds"),
            (ProphetModel(), [], "example_tsds"),
            (ProphetModel(timestamp_column="external_timestamp"), [], "ts_with_external_timestamp"),
            (SARIMAXModel(), [], "example_tsds"),
            (HoltModel(), [], "example_tsds"),
            (HoltWintersModel(), [], "example_tsds"),
            (SimpleExpSmoothingModel(), [], "example_tsds"),
            (BATSModel(use_trend=True), [], "example_tsds"),
            (TBATSModel(use_trend=True), [], "example_tsds"),
            (StatsForecastARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoARIMAModel(), [], "example_tsds"),
            (StatsForecastAutoCESModel(), [], "example_tsds"),
            (StatsForecastAutoETSModel(), [], "example_tsds"),
            (StatsForecastAutoThetaModel(), [], "example_tsds"),
        ],
    )
    def test_forecast_new_segments_failed_per_segment(self, model, transforms, dataset_name, request):
        ts = request.getfixturevalue(dataset_name)
        with pytest.raises(NotImplementedError, match="Per-segment models can't make predictions on new segments"):
            self._test_forecast_new_segments(ts, model, transforms, train_segments=["segment_1"])
