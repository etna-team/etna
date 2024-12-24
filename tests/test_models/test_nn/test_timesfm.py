import os
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.libs.timesfm import TimesFmTorch
from etna.models.nn import TimesFMModel
from etna.pipeline import Pipeline
from etna.transforms import DateFlagsTransform
from etna.transforms import LagTransform
from etna.transforms import SegmentEncoderTransform


@pytest.fixture
def ts_increasing_integers():
    n = 128
    df = generate_ar_df(start_time="2001-01-01", periods=n, n_segments=2)
    df["target"] = list(range(n)) + list(range(100, 100 + n))
    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture
def expected_ts_increasing_integers():
    df = generate_ar_df(start_time="2001-05-09", periods=2, n_segments=2)
    df["target"] = [128.0, 129.0] + [228.0, 229.0]
    ts = TSDataset(df, freq="D")
    return ts


@pytest.mark.smoke
def test_chronos_url(tmp_path):
    model_name = "timesfm-1.0-200m-pytorch.ckpt"
    url = f"http://etna-github-prod.cdn-tinkoff.ru/timesfm/{model_name}"
    _ = TimesFMModel(path_or_url=url, cache_dir=tmp_path)
    assert os.path.exists(tmp_path / model_name)


@pytest.mark.smoke
def test_chronos_custom_cache_dir(tmp_path):
    path_or_url = "google/timesfm-1.0-200m-pytorch"
    model_name = path_or_url.split("/")[-1]
    _ = TimesFMModel(path_or_url=path_or_url, cache_dir=tmp_path)
    assert os.path.exists(tmp_path / f"models--google--{model_name}")


@pytest.mark.smoke
def test_context_size():
    model = TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=10)
    assert model.context_size == 10


@pytest.mark.smoke
def test_chronos_get_model(example_tsds):
    model = TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch")
    assert isinstance(model.get_model(), TimesFmTorch)


@pytest.mark.smoke
def test_fit(example_tsds):
    model = TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch")
    model.fit(example_tsds)


@pytest.mark.smoke
def test_predict(example_tsds):
    model = TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch")
    with pytest.raises(NotImplementedError, match="Method predict isn't currently implemented!"):
        model.predict(ts=example_tsds, prediction_size=1)


@pytest.mark.smoke
def test_forecast_warns_big_context_size(ts_increasing_integers):
    model = TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=512)
    pipeline = Pipeline(model=model, horizon=1)
    pipeline.fit(ts_increasing_integers)
    with pytest.warns(UserWarning, match="Actual length of a dataset is less that context size."):
        _ = pipeline.forecast()


@pytest.mark.parametrize("encoder_length", [32, 64, 128])
def test_forecast(ts_increasing_integers, expected_ts_increasing_integers, encoder_length):
    model = TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=encoder_length)
    pipeline = Pipeline(model=model, horizon=2)
    pipeline.fit(ts_increasing_integers)
    forecast = pipeline.forecast()
    assert_frame_equal(forecast.df, expected_ts_increasing_integers.df, atol=1)


def test_forecast_exogenous_features(ts_increasing_integers, expected_ts_increasing_integers):
    horizon = 2
    transforms = [
        SegmentEncoderTransform(),
        LagTransform(in_column="target", lags=[horizon], out_column="lag"),
        DateFlagsTransform(day_number_in_week=True, day_number_in_month=False, is_weekend=False, out_column="flag"),
    ]
    model = TimesFMModel(
        path_or_url="google/timesfm-1.0-200m-pytorch",
        encoder_length=32,
        static_categoricals=["segment_code"],
        time_varying_reals=[f"lag_{horizon}"],
        time_varying_categoricals=["flag_day_number_in_week"],
    )
    pipeline = Pipeline(model=model, transforms=transforms, horizon=horizon)
    pipeline.fit(ts_increasing_integers)
    forecast = pipeline.forecast()
    assert_frame_equal(forecast.df.loc[:, pd.IndexSlice[:, "target"]], expected_ts_increasing_integers.df, atol=1)


@pytest.mark.smoke
def test_forecast_only_target_failed_int_timestamps(example_tsds_int_timestamp):
    model = TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=32)
    pipeline = Pipeline(model=model, horizon=1)
    pipeline.fit(example_tsds_int_timestamp)
    with pytest.raises(
        NotImplementedError,
        match="Data with None frequency isn't currently implemented for forecasting without exogenous features.",
    ):
        _ = pipeline.forecast()


@pytest.mark.smoke
def test_forecast_exog_int_timestamps(example_tsds_int_timestamp):
    horizon = 2
    transforms = [SegmentEncoderTransform(), LagTransform(in_column="target", lags=[horizon], out_column="lag")]
    model = TimesFMModel(
        path_or_url="google/timesfm-1.0-200m-pytorch",
        encoder_length=32,
        static_categoricals=["segment_code"],
        time_varying_reals=[f"lag_{horizon}"],
    )
    pipeline = Pipeline(model=model, transforms=transforms, horizon=horizon)
    pipeline.fit(example_tsds_int_timestamp)
    with pytest.warns(
        UserWarning,
        match="Frequency is None. Mapping it to 0, that can be not optimal. Better to set it to known frequency",
    ):
        _ = pipeline.forecast()


@pytest.mark.smoke
@pytest.mark.parametrize("encoder_length", [16, 33])
def test_forecast_wrong_context_len(ts_increasing_integers, encoder_length):
    model = TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=encoder_length)
    pipeline = Pipeline(model=model, horizon=1)
    pipeline.fit(ts_increasing_integers)
    with pytest.raises(RuntimeError, match=r"shape .+ is invalid for input of size \d+"):
        _ = pipeline.forecast()


@pytest.mark.smoke
def test_forecast_without_fit(example_tsds):
    model = TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=32)
    pipeline = Pipeline(model=model, horizon=1)
    _ = pipeline.forecast(example_tsds)


@pytest.mark.smoke
def test_forecast_fails_components(example_tsds):
    model = TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch")
    pipeline = Pipeline(model=model, horizon=1)
    with pytest.raises(NotImplementedError, match="This mode isn't currently implemented!"):
        pipeline.forecast(ts=example_tsds, return_components=True)


@pytest.mark.smoke
def test_list_models():
    assert TimesFMModel.list_models() == ["google/timesfm-1.0-200m-pytorch"]


@pytest.mark.smoke
def test_save_load(tmp_path, ts_increasing_integers):
    path = Path(tmp_path) / "tmp.zip"
    model = TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch", encoder_length=32)
    model.save(path)
    loaded_model = TimesFMModel.load(path)

    pipeline = Pipeline(model=loaded_model, horizon=1)
    pipeline.fit(ts_increasing_integers)
    _ = pipeline.forecast()
    assert isinstance(loaded_model, TimesFMModel)


@pytest.mark.smoke
def test_params_to_tune():
    model = TimesFMModel(path_or_url="google/timesfm-1.0-200m-pytorch")
    assert len(model.params_to_tune()) == 0
