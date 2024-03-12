import pathlib
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from etna.metrics import SMAPE
from etna.models import LinearMultiSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import EmbeddingSegmentTransform
from etna.transforms.embeddings.models import TS2VecEmbeddingModel


@pytest.mark.parametrize(
    "embedding_model",
    [
        TS2VecEmbeddingModel(input_dims=3, n_epochs=1),
    ],
)
@pytest.mark.smoke
def test_fit(ts_with_exog_nan_begin, embedding_model):
    transform = EmbeddingSegmentTransform(
        in_columns=["target", "exog_1", "exog_2"], embedding_model=embedding_model, out_column="emb"
    )
    transform.fit(ts=ts_with_exog_nan_begin)


@pytest.mark.parametrize(
    "embedding_model",
    [
        TS2VecEmbeddingModel(input_dims=3, n_epochs=1),
    ],
)
@pytest.mark.smoke
def test_fit_transform(ts_with_exog_nan_begin, embedding_model):
    transform = EmbeddingSegmentTransform(
        in_columns=["target", "exog_1", "exog_2"], embedding_model=embedding_model, out_column="emb"
    )
    transform.fit_transform(ts=ts_with_exog_nan_begin)


@pytest.mark.parametrize(
    "embedding_model",
    [
        TS2VecEmbeddingModel(input_dims=1, n_epochs=1),
    ],
)
@pytest.mark.smoke
def test_fit_forecast(example_tsds, embedding_model):
    emb_transform = EmbeddingSegmentTransform(in_columns=["target"], embedding_model=embedding_model, out_column="emb")
    transforms = [emb_transform]

    pipeline = Pipeline(model=LinearMultiSegmentModel(), transforms=transforms, horizon=7)
    pipeline.fit(example_tsds).forecast()


@pytest.mark.parametrize(
    "embedding_model",
    [
        TS2VecEmbeddingModel(input_dims=1, n_epochs=1, output_dims=6),
    ],
)
@pytest.mark.smoke
def test_backtest_full_series(example_tsds, embedding_model):
    emb_transform = EmbeddingSegmentTransform(in_columns=["target"], embedding_model=embedding_model, out_column="emb")
    transforms = [emb_transform]

    pipeline = Pipeline(model=LinearMultiSegmentModel(), transforms=transforms, horizon=7)
    pipeline.backtest(ts=example_tsds, metrics=[SMAPE()], n_folds=2, n_jobs=2, joblib_params=dict(backend="loky"))


@pytest.mark.parametrize(
    "embedding_model",
    [
        TS2VecEmbeddingModel(input_dims=1, n_epochs=1, output_dims=6),
    ],
)
@pytest.mark.smoke
def test_make_future(example_tsds, embedding_model):
    emb_transform = EmbeddingSegmentTransform(in_columns=["target"], embedding_model=embedding_model, out_column="emb")
    emb_transform.fit(example_tsds)

    make_future_df = example_tsds.make_future(5, transforms=[emb_transform]).df
    values_make_future = make_future_df.loc[:, pd.IndexSlice[:, emb_transform._get_out_columns()]].values[0]

    example_tsds.transform([emb_transform])
    ts_df = example_tsds.df
    values_ts = ts_df.loc[:, pd.IndexSlice[:, emb_transform._get_out_columns()]].values[0]

    assert np.array_equal(values_make_future, values_ts)


@pytest.mark.parametrize(
    "embedding_model",
    [
        TS2VecEmbeddingModel(input_dims=3, n_epochs=1),
    ],
)
@pytest.mark.smoke
def test_save(ts_with_exog_nan_begin, tmp_path, embedding_model):
    transform = EmbeddingSegmentTransform(
        in_columns=["target", "exog_1", "exog_2"], embedding_model=embedding_model, out_column="emb"
    )
    transform.fit(ts=ts_with_exog_nan_begin)

    path = pathlib.Path(tmp_path) / "tmp.zip"
    transform.save(path=path)


@pytest.mark.parametrize(
    "embedding_model",
    [
        TS2VecEmbeddingModel(input_dims=3, n_epochs=1),
    ],
)
@pytest.mark.smoke
def test_load(ts_with_exog_nan_begin, tmp_path, embedding_model):
    transform = EmbeddingSegmentTransform(
        in_columns=["target", "exog_1", "exog_2"], embedding_model=embedding_model, out_column="emb"
    )
    transform.fit(ts=ts_with_exog_nan_begin)

    path = pathlib.Path(tmp_path) / "tmp.zip"
    transform.save(path=path)
    EmbeddingSegmentTransform.load(path=path)


@pytest.mark.parametrize(
    "output_dims, out_column, expected_out_columns",
    [(2, "emb", ["emb_0", "emb_1"]), (3, "lag", ["lag_0", "lag_1", "lag_2"])],
)
def test_get_out_columns(output_dims, out_column, expected_out_columns):
    transform = EmbeddingSegmentTransform(
        in_columns=Mock(), embedding_model=Mock(output_dims=output_dims), out_column=out_column
    )
    assert sorted(expected_out_columns) == sorted(transform._get_out_columns())


@pytest.mark.parametrize("embedding_model", [TS2VecEmbeddingModel(input_dims=3, n_epochs=1)])
def test_second_fit_not_update_state(ts_with_exog_nan_begin, embedding_model):
    transform = EmbeddingSegmentTransform(
        in_columns=["target", "exog_1", "exog_2"], embedding_model=embedding_model, out_column="emb"
    )
    first_fit_encoded = transform.fit_transform(ts=deepcopy(ts_with_exog_nan_begin))
    second_fit_encoded = transform.fit_transform(ts=deepcopy(ts_with_exog_nan_begin))
    pd.testing.assert_frame_equal(first_fit_encoded.to_pandas(), second_fit_encoded.to_pandas())


@pytest.mark.parametrize(
    "embedding_model",
    [
        TS2VecEmbeddingModel(input_dims=3, output_dims=3, n_epochs=1),
    ],
)
def test_transform_format(
    ts_with_exog_nan_begin,
    embedding_model,
    expected_columns=(
        "target",
        "exog_1",
        "exog_2",
        "embedding_segment_0",
        "embedding_segment_1",
        "embedding_segment_2",
    ),
):
    transform = EmbeddingSegmentTransform(
        in_columns=["target", "exog_1", "exog_2"], embedding_model=embedding_model, out_column="embedding_segment"
    )
    transform.fit_transform(ts=ts_with_exog_nan_begin)
    obtained_columns = set(ts_with_exog_nan_begin.columns.get_level_values("feature"))
    embedding_columns = transform.get_regressors_info()
    embeddings = ts_with_exog_nan_begin.df.loc[:, pd.IndexSlice[:, embedding_columns]].values
    assert sorted(obtained_columns) == sorted(expected_columns)
    assert np.all(embeddings == embeddings[0, :], axis=0).all()


@pytest.mark.parametrize(
    "embedding_model",
    [
        TS2VecEmbeddingModel(input_dims=3, output_dims=3, n_epochs=1),
    ],
)
def test_transform_load_pre_fitted(ts_with_exog_nan_begin, tmp_path, embedding_model):
    transform = EmbeddingSegmentTransform(
        in_columns=["target", "exog_1", "exog_2"], embedding_model=embedding_model, out_column="emb"
    )
    before_load_ts = transform.fit_transform(ts=deepcopy(ts_with_exog_nan_begin))

    path = pathlib.Path(tmp_path) / "tmp.zip"
    transform.save(path=path)

    loaded_transform = EmbeddingSegmentTransform.load(path=path)
    after_load_ts = loaded_transform.transform(ts=deepcopy(ts_with_exog_nan_begin))

    pd.testing.assert_frame_equal(before_load_ts.to_pandas(), after_load_ts.to_pandas())
