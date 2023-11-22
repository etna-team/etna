.. _experimental:

Experimental
============

API details
-----------

.. currentmodule:: etna.experimental

Change-point utilities:

.. autosummary::
   :toctree: api/
   :template: base.rst

   change_points.get_ruptures_regularization

Classification of time-series:

.. autosummary::
   :toctree: api/
   :template: class.rst

   classification.TimeSeriesBinaryClassifier
   classification.PredictabilityAnalyzer
   classification.feature_extraction.TSFreshFeatureExtractor
   classification.feature_extraction.WEASELFeatureExtractor

Prediction Intervals:

.. autosummary::
   :toctree: api/
   :template: class.rst

   prediction_intervals.BasePredictionIntervals
   prediction_intervals.NaiveVariancePredictionIntervals
   prediction_intervals.ConformalPredictionIntervals

Prediction Intervals utilities:

.. autosummary::
   :toctree: api/
   :template: base.rst

   prediction_intervals.utils.residuals_matrices
