import json
import pickle

import hydra_slayer
import pytest
from ruptures import Binseg
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from etna.core import BaseMixin
from etna.ensembles import StackingEnsemble
from etna.ensembles import VotingEnsemble
from etna.libs.pytorch_lightning.callbacks import EarlyStopping
from etna.metrics import MAE
from etna.metrics import SMAPE
from etna.models import AutoARIMAModel
from etna.models import CatBoostPerSegmentModel
from etna.models import LinearPerSegmentModel
from etna.models import SklearnMultiSegmentModel
from etna.models import SklearnPerSegmentModel
from etna.models.nn import MLPModel
from etna.pipeline import Pipeline
from etna.transforms import AddConstTransform
from etna.transforms import ChangePointsTrendTransform
from etna.transforms import DensityOutliersTransform
from etna.transforms import LambdaTransform
from etna.transforms import LogTransform
from etna.transforms.decomposition.change_points_based import RupturesChangePointsModel
from etna.transforms.decomposition.change_points_based import SklearnRegressionPerIntervalModel
from tests.test_core.conftest import BaseDummy


def ensemble_samples():
    pipeline1 = Pipeline(
        model=CatBoostPerSegmentModel(),
        transforms=[
            AddConstTransform(in_column="target", value=10),
            ChangePointsTrendTransform(
                in_column="target",
                change_points_model=RupturesChangePointsModel(Binseg(model="l2", min_size=2, jump=5), n_bkps=50),
                per_interval_model=SklearnRegressionPerIntervalModel(),
            ),
        ],
        horizon=5,
    )
    pipeline2 = Pipeline(
        model=LinearPerSegmentModel(),
        transforms=[
            ChangePointsTrendTransform(
                in_column="target",
                change_points_model=RupturesChangePointsModel(
                    change_points_model=Binseg(model="l2", min_size=2, jump=5), n_bkps=50
                ),
                per_interval_model=SklearnRegressionPerIntervalModel(),
            ),
            LogTransform(in_column="target"),
        ],
        horizon=5,
    )
    return [pipeline1, pipeline2]


@pytest.mark.parametrize(
    "target_object",
    [
        AddConstTransform(in_column="target", value=10),
        ChangePointsTrendTransform(
            in_column="target",
            change_points_model=RupturesChangePointsModel(
                change_points_model=Binseg(model="l2", min_size=2, jump=5), n_bkps=50
            ),
            per_interval_model=SklearnRegressionPerIntervalModel(model=LinearRegression()),
        ),
        pytest.param(
            DensityOutliersTransform("target", distance_coef=6),
            marks=pytest.mark.xfail(
                reason="partial function after initialization instead of original function, dumps return different results"
            ),
        ),
        pytest.param(
            LambdaTransform(in_column="target", transform_func=lambda x: x - 2, inverse_transform_func=lambda x: x + 2),
            marks=pytest.mark.xfail(reason="lambdas in class attributes"),
        ),
    ],
)
def test_to_dict_transforms(target_object):
    dict_object = target_object.to_dict()
    transformed_object = hydra_slayer.get_from_params(**dict_object)
    assert json.loads(json.dumps(dict_object)) == dict_object
    assert pickle.dumps(transformed_object) == pickle.dumps(target_object)


# fmt: off
@pytest.mark.parametrize(
    "target_object, expected",
    [
        (
            DensityOutliersTransform("target", distance_coef=6),
            {'in_column': 'target', 'window_size': 15, 'distance_coef': 6, 'n_neighbors': 3, 'distance_func': 'absolute_difference', '_target_': 'etna.transforms.outliers.point_outliers.DensityOutliersTransform'}  # noqa: E501
        ),
        (
            MLPModel(decoder_length=1, hidden_size=[64, 64], input_size=1, trainer_params={"max_epochs": 100, "callbacks": [EarlyStopping(monitor="val_loss", patience=3)]}, lr=0.01, train_batch_size=32, split_params=dict(train_size=0.75)),  # noqa: E501
            {'input_size': 1, 'decoder_length': 1, 'hidden_size': [64, 64], 'encoder_length': 0, 'lr': 0.01, 'train_batch_size': 32, 'test_batch_size': 16, 'trainer_params': {'max_epochs': 100, 'callbacks': [{'monitor': 'val_loss', 'patience': 3, '_target_': 'etna.libs.pytorch_lightning.callbacks.EarlyStopping'}]}, 'train_dataloader_params': {}, 'test_dataloader_params': {}, 'val_dataloader_params': {}, 'split_params': {'train_size': 0.75}, '_target_': 'etna.models.nn.mlp.MLPModel'}  # noqa: E501
        )
    ],
)
def test_to_dict_transforms_with_expected(target_object, expected):
    dict_object = target_object.to_dict()
    assert dict_object == expected
# fmt: on


@pytest.mark.parametrize(
    "target_model",
    [
        LinearPerSegmentModel(),
        CatBoostPerSegmentModel(),
        AutoARIMAModel(),
        SklearnPerSegmentModel(regressor=RandomForestRegressor()),
        SklearnMultiSegmentModel(regressor=RandomForestRegressor()),
    ],
)
def test_to_dict_models(target_model):
    dict_object = target_model.to_dict()
    transformed_object = hydra_slayer.get_from_params(**dict_object)
    assert json.loads(json.dumps(dict_object)) == dict_object
    assert pickle.dumps(transformed_object) == pickle.dumps(target_model)


@pytest.mark.parametrize(
    "target_object",
    [
        Pipeline(
            model=CatBoostPerSegmentModel(),
            transforms=[
                AddConstTransform(in_column="target", value=10),
                ChangePointsTrendTransform(
                    in_column="target",
                    per_interval_model=SklearnRegressionPerIntervalModel(),
                    change_points_model=RupturesChangePointsModel(
                        change_points_model=Binseg(model="l2", min_size=2, jump=5), n_bkps=50
                    ),
                ),
            ],
            horizon=5,
        )
    ],
)
def test_to_dict_pipeline(target_object):
    dict_object = target_object.to_dict()
    transformed_object = hydra_slayer.get_from_params(**dict_object)
    assert json.loads(json.dumps(dict_object)) == dict_object
    assert pickle.dumps(transformed_object) == pickle.dumps(target_object)


@pytest.mark.parametrize("target_object", [MAE(mode="macro"), SMAPE()])
def test_to_dict_metrics(target_object):
    dict_object = target_object.to_dict()
    transformed_object = hydra_slayer.get_from_params(**dict_object)
    assert json.loads(json.dumps(dict_object)) == dict_object
    assert pickle.dumps(transformed_object) == pickle.dumps(target_object)


@pytest.mark.parametrize(
    "target_ensemble",
    [VotingEnsemble(pipelines=ensemble_samples(), weights=[0.4, 0.6]), StackingEnsemble(pipelines=ensemble_samples())],
)
def test_ensembles(target_ensemble):
    dict_object = target_ensemble.to_dict()
    transformed_object = hydra_slayer.get_from_params(**dict_object)
    assert json.loads(json.dumps(dict_object)) == dict_object
    assert pickle.dumps(transformed_object) == pickle.dumps(target_ensemble)


class _Dummy:
    pass


class _InvalidParsing(BaseMixin):
    def __init__(self, a: _Dummy):
        self.a = a


def test_warnings():
    with pytest.warns(Warning, match="Some of external objects in input parameters could be not written in dict"):
        _ = _InvalidParsing(_Dummy()).to_dict()


def test_to_dict_public_property_private_attribute():
    dummy = BaseDummy(a=1, b=2)
    assert dummy.to_dict() == {"a": 1, "b": 2, "_target_": "tests.test_core.conftest.BaseDummy"}
