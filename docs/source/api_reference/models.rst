.. _models:

Models
======

.. automodule:: etna.models
    :no-members:
    :no-inherited-members:

API details
-----------

.. currentmodule:: etna.models

Base:

.. autosummary::
   :toctree: api/
   :template: class.rst

   NonPredictionIntervalContextIgnorantAbstractModel
   NonPredictionIntervalContextRequiredAbstractModel
   PredictionIntervalContextIgnorantAbstractModel
   PredictionIntervalContextRequiredAbstractModel

Naive models:

.. autosummary::
   :toctree: api/
   :template: class.rst

   SeasonalMovingAverageModel
   MovingAverageModel
   NaiveModel
   DeadlineMovingAverageModel

Statistical models:

.. autosummary::
   :toctree: api/
   :template: class.rst

   AutoARIMAModel
   SARIMAXModel
   HoltWintersModel
   HoltModel
   SimpleExpSmoothingModel
   ProphetModel
   TBATSModel
   BATSModel
   StatsForecastARIMAModel
   StatsForecastAutoARIMAModel
   StatsForecastAutoCESModel
   StatsForecastAutoETSModel
   StatsForecastAutoThetaModel

ML-models:

.. autosummary::
   :toctree: api/
   :template: class.rst

   CatBoostMultiSegmentModel
   CatBoostPerSegmentModel
   ElasticMultiSegmentModel
   ElasticPerSegmentModel
   LinearMultiSegmentModel
   LinearPerSegmentModel
   SklearnMultiSegmentModel
   SklearnPerSegmentModel

Native neural network models:

.. autosummary::
   :toctree: api/
   :template: class.rst

   nn.RNNModel
   nn.MLPModel
   nn.DeepStateModel
   nn.NBeatsGenericModel
   nn.NBeatsInterpretableModel
   nn.PatchTSTModel
   nn.DeepARModel
   nn.TFTModel

Utilities for :py:class:`~etna.models.nn.deepstate.deepstate.DeepStateModel`

.. autosummary::
   :toctree: api/
   :template: class.rst

   nn.deepstate.CompositeSSM
   nn.deepstate.LevelSSM
   nn.deepstate.LevelTrendSSM
   nn.deepstate.SeasonalitySSM
   nn.deepstate.DailySeasonalitySSM
   nn.deepstate.SeasonalitySSM
   nn.deepstate.YearlySeasonalitySSM

Pretrained neural network models:

.. autosummary::
   :toctree: api/
   :template: class.rst

   nn.ChronosModel
   nn.ChronosBoltModel
   nn.TimesFMModel