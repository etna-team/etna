REQUIRED_PARAMS = {"timestamp_column": None, "chronos_device": "auto", "timesfm_device": "gpu"}

NO_FREQ_SUPER_FAST = [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.NaiveModel", "lag": 1},
        "transforms": [],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.NaiveModel", "lag": "${horizon}"},
        "transforms": [],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": "${horizon}", "window": 3},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": "${mult:${horizon}, 3}",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
        ],
    },
    {
        "_target_": "etna.ensembles.VotingEnsemble",
        "pipelines": [
            {
                "_target_": "etna.pipeline.Pipeline",
                "horizon": "${__aux__.horizon}",
                "model": {
                    "_target_": "etna.models.SeasonalMovingAverageModel",
                    "seasonality": "${__aux__.horizon}",
                    "window": 1,
                },
                "transforms": [],
            },
            {
                "_target_": "etna.pipeline.Pipeline",
                "horizon": "${__aux__.horizon}",
                "model": {
                    "_target_": "etna.models.SeasonalMovingAverageModel",
                    "seasonality": "${__aux__.horizon}",
                    "window": 2,
                },
                "transforms": [],
            },
            {
                "_target_": "etna.pipeline.Pipeline",
                "horizon": "${__aux__.horizon}",
                "model": {
                    "_target_": "etna.models.SeasonalMovingAverageModel",
                    "seasonality": "${__aux__.horizon}",
                    "window": 7,
                },
                "transforms": [],
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-tiny.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-mini.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-small.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
]

NO_FREQ_FAST = NO_FREQ_SUPER_FAST + [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.StatsForecastAutoThetaModel",
            "decomposition_type": "additive",
            "season_length": "${horizon}",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": "${mult:${horizon}, 3}",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.StatsForecastAutoThetaModel",
            "decomposition_type": "multiplicative",
            "season_length": "${horizon}",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": "${mult:${horizon}, 3}",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.ProphetModel",
            "seasonality_mode": "multiplicative",
            "timestamp_column": "${__aux__.timestamp_column}",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            }
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.ProphetModel",
            "seasonality_mode": "additive",
            "timestamp_column": "${__aux__.timestamp_column}",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": "${mult:${horizon}, 3}",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.TimesFMModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/timesfm/timesfm-1.0-200m-pytorch.ckpt",
            "encoder_length": 512,
            "device": "${__aux__.timesfm_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-base.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {"_target_": "etna.transforms.SegmentEncoderTransform"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${arange:${horizon},${mult:${horizon}, 2}}",
                "out_column": "target_lag",
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.ElasticMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {"_target_": "etna.transforms.SegmentEncoderTransform"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${arange:${horizon},${mult:${horizon}, 2}}",
                "out_column": "target_lag",
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.ElasticPerSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${arange:${horizon},${mult:${horizon}, 2}}",
                "out_column": "target_lag",
            },
        ],
    },
]

NO_FREQ_MEDIUM = NO_FREQ_FAST + [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.HoltWintersModel",
            "damped_trend": True,
            "seasonal": "add",
            "seasonal_periods": "${horizon}",
            "trend": "add",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": "${mult:${horizon}, 3}",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.StatsForecastAutoETSModel", "season_length": "${horizon}"},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": "${mult:${horizon}, 3}",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.TimesFMModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/timesfm/timesfm-2.0-500m-pytorch.ckpt",
            "encoder_length": 2048,
            "num_layers": 50,
            "use_positional_embedding": False,
            "device": "${__aux__.timesfm_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "window_size": "${mult:${horizon}, 3}",
                "distance_coef": 3,
                "in_column": "target",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${arange:${horizon},${mult:${horizon}, 2}}",
                "out_column": "target_lag",
            },
            {
                "_target_": "etna.transforms.EmbeddingSegmentTransform",
                "in_columns": ["target"],
                "embedding_model": {
                    "_target_": "etna.transforms.embeddings.models.TS2VecEmbeddingModel.load",
                    "model_name": "ts2vec_tiny",
                    "_mode_": "call",
                },
                "training_params": {},
                "out_column": "emb",
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "window_size": "${mult:${horizon}, 3}",
                "distance_coef": 3,
                "in_column": "target",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${arange:${horizon},${mult:${horizon}, 2}}",
                "out_column": "target_lag",
            },
            {
                "_target_": "etna.transforms.EmbeddingSegmentTransform",
                "in_columns": ["target"],
                "embedding_model": {
                    "_target_": "etna.transforms.embeddings.models.TSTCCEmbeddingModel.load",
                    "model_name": "tstcc_medium",
                    "_mode_": "call",
                },
                "training_params": {},
                "out_column": "emb",
            },
        ],
    },
]

NO_FREQ_HEAVY = NO_FREQ_MEDIUM + [
    {
        "_target_": "etna.pipeline.AutoRegressivePipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "step": 1,
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "window_size": "${mult:${horizon}, 3}",
                "distance_coef": 3,
                "in_column": "target",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.ChangePointsTrendTransform",
                "change_points_model": {
                    "_target_": "etna.transforms.decomposition.change_points_based.RupturesChangePointsModel",
                    "change_points_model": {
                        "_target_": "ruptures.detection.Binseg",
                        "jump": 1,
                        "min_size": 10,
                        "model": "ar",
                    },
                    "n_bkps": 5,
                },
                "in_column": "target",
            },
            {"_target_": "etna.transforms.MeanSegmentEncoderTransform"},
            {"_target_": "etna.transforms.MinMaxScalerTransform", "in_column": "target", "inplace": True},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${arange:${horizon},${mult:${horizon}, 2}}",
                "out_column": "target_lag",
            },
            {"_target_": "etna.transforms.FourierTransform", "order": 5, "period": 365.25},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.ElasticPerSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "window_size": "${mult:${horizon}, 3}",
                "distance_coef": 3,
                "in_column": "target",
            },
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "default_value": 0,
                "in_column": "target",
                "strategy": "mean",
            },
            {"_target_": "etna.transforms.FourierTransform", "order": 10, "period": 365.25},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${arange:${horizon},${mult:${horizon}, 2}}",
                "out_column": "target_lag",
            },
            {"_target_": "etna.transforms.DeseasonalityTransform", "in_column": "target", "period": "${horizon}"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.ElasticMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.FourierTransform", "order": 10, "period": 365.25},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${arange:${horizon},${mult:${horizon}, 2}}",
                "out_column": "target_lag",
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.statsforecast.StatsForecastAutoARIMAModel",
            "kwargs": {"nmodels": 10},
            "max_D": 1,
            "max_P": 3,
            "max_Q": 3,
            "max_d": 2,
            "max_order": 5,
            "max_p": 6,
            "max_q": 6,
            "season_length": "${horizon}",
            "start_P": 1,
            "start_Q": 1,
            "start_p": 2,
            "start_q": 2,
        },
        "transforms": [
            {
                "_target_": "etna.transforms.missing_values.imputation.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            }
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-t5-large.zip",
            "encoder_length": 512,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            }
        ],
    },
]


D_SUPER_FAST = [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.NaiveModel", "lag": 1},
        "transforms": [],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.NaiveModel", "lag": 7},
        "transforms": [],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.NaiveModel", "lag": "${horizon}"},
        "transforms": [],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": 7, "window": 3},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 21,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.DeadlineMovingAverageModel", "seasonality": "month", "window": 3},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 21,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
        ],
    },
    {
        "_target_": "etna.ensembles.VotingEnsemble",
        "pipelines": [
            {
                "_target_": "etna.pipeline.Pipeline",
                "horizon": "${__aux__.horizon}",
                "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": 14, "window": 1},
                "transforms": [],
            },
            {
                "_target_": "etna.pipeline.Pipeline",
                "horizon": "${__aux__.horizon}",
                "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": 7, "window": 2},
                "transforms": [],
            },
            {
                "_target_": "etna.pipeline.Pipeline",
                "horizon": "${__aux__.horizon}",
                "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": 7, "window": 7},
                "transforms": [],
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-tiny.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-mini.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-small.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
]

D_FAST = D_SUPER_FAST + [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.StatsForecastAutoThetaModel",
            "decomposition_type": "additive",
            "season_length": 7,
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 21,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.StatsForecastAutoThetaModel",
            "decomposition_type": "multiplicative",
            "season_length": 7,
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 21,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.ProphetModel",
            "seasonality_mode": "multiplicative",
            "timestamp_column": "${__aux__.timestamp_column}",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            }
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.ProphetModel",
            "seasonality_mode": "additive",
            "timestamp_column": "${__aux__.timestamp_column}",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 21,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.TimesFMModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/timesfm/timesfm-1.0-200m-pytorch.ckpt",
            "encoder_length": 512,
            "device": "${__aux__.timesfm_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-base.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {"_target_": "etna.transforms.SegmentEncoderTransform"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}",
                "out_column": "target_lag",
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.ElasticMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {"_target_": "etna.transforms.SegmentEncoderTransform"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}",
                "out_column": "target_lag",
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.ElasticPerSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}",
                "out_column": "target_lag",
            },
        ],
    },
]

D_MEDIUM = D_FAST + [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.HoltWintersModel",
            "damped_trend": True,
            "seasonal": "add",
            "seasonal_periods": 7,
            "trend": "add",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 21,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.StatsForecastAutoETSModel", "season_length": 7},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 21,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.TimesFMModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/timesfm/timesfm-2.0-500m-pytorch.ckpt",
            "encoder_length": 2048,
            "num_layers": 50,
            "use_positional_embedding": False,
            "device": "${__aux__.timesfm_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "window_size": 21,
                "distance_coef": 3,
                "in_column": "target",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}",
                "out_column": "target_lag",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_week": True,
                "day_number_in_month": True,
                "day_number_in_year": False,
                "week_number_in_month": False,
                "week_number_in_year": False,
                "month_number_in_year": False,
                "year_number": False,
                "is_weekend": True,
                "out_column": "dt",
            },
            {
                "_target_": "etna.transforms.EmbeddingSegmentTransform",
                "in_columns": ["target"],
                "embedding_model": {
                    "_target_": "etna.transforms.embeddings.models.TS2VecEmbeddingModel.load",
                    "model_name": "ts2vec_tiny",
                    "_mode_": "call",
                },
                "training_params": {},
                "out_column": "emb",
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "window_size": 21,
                "distance_coef": 3,
                "in_column": "target",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}",
                "out_column": "target_lag",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_week": True,
                "day_number_in_month": True,
                "day_number_in_year": False,
                "week_number_in_month": False,
                "week_number_in_year": False,
                "month_number_in_year": False,
                "year_number": False,
                "is_weekend": True,
                "out_column": "dt",
            },
            {
                "_target_": "etna.transforms.EmbeddingSegmentTransform",
                "in_columns": ["target"],
                "embedding_model": {
                    "_target_": "etna.transforms.embeddings.models.TSTCCEmbeddingModel.load",
                    "model_name": "tstcc_medium",
                    "_mode_": "call",
                },
                "training_params": {},
                "out_column": "emb",
            },
        ],
    },
]

D_HEAVY = D_MEDIUM + [
    {
        "_target_": "etna.pipeline.AutoRegressivePipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "step": 1,
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.DensityOutliersTransform", "in_column": "target"},
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.ChangePointsTrendTransform",
                "change_points_model": {
                    "_target_": "etna.transforms.decomposition.change_points_based.RupturesChangePointsModel",
                    "change_points_model": {
                        "_target_": "ruptures.detection.Binseg",
                        "jump": 1,
                        "min_size": 10,
                        "model": "ar",
                    },
                    "n_bkps": 5,
                },
                "in_column": "target",
            },
            {"_target_": "etna.transforms.MeanSegmentEncoderTransform"},
            {"_target_": "etna.transforms.MinMaxScalerTransform", "in_column": "target", "inplace": True},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${step},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]}",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_month": True,
                "day_number_in_week": True,
                "is_weekend": True,
                "month_number_in_year": True,
                "week_number_in_month": True,
                "week_number_in_year": True,
                "year_number": True,
            },
            {"_target_": "etna.transforms.FourierTransform", "order": 5, "period": 365.25},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.ElasticPerSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.DensityOutliersTransform", "in_column": "target"},
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "default_value": 0,
                "in_column": "target",
                "strategy": "mean",
            },
            {"_target_": "etna.transforms.FourierTransform", "order": 10, "period": 365.25},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]}",
                "out_column": "target_lag",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_month": True,
                "day_number_in_week": True,
                "is_weekend": True,
                "month_number_in_year": False,
                "out_column": "date",
                "week_number_in_month": True,
                "week_number_in_year": False,
                "year_number": False,
            },
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "date_day_number_in_month"},
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "date_day_number_in_week"},
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "date_is_weekend"},
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "date_week_number_in_month"},
            {"_target_": "etna.transforms.DeseasonalityTransform", "in_column": "target", "period": 7},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.ElasticMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.FourierTransform", "order": 10, "period": 365.25},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]}",
                "out_column": "target_lag",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_month": True,
                "day_number_in_week": True,
                "is_weekend": True,
                "month_number_in_year": False,
                "out_column": "date",
                "week_number_in_month": True,
                "week_number_in_year": False,
                "year_number": False,
            },
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "date_day_number_in_month"},
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "date_day_number_in_week"},
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "date_is_weekend"},
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "date_week_number_in_month"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.statsforecast.StatsForecastAutoARIMAModel",
            "kwargs": {"nmodels": 10},
            "max_D": 1,
            "max_P": 3,
            "max_Q": 3,
            "max_d": 2,
            "max_order": 5,
            "max_p": 6,
            "max_q": 6,
            "season_length": 7,
            "start_P": 1,
            "start_Q": 1,
            "start_p": 2,
            "start_q": 2,
        },
        "transforms": [
            {
                "_target_": "etna.transforms.missing_values.imputation.TimeSeriesImputerTransform",
                "constant_value": 0,
                "in_column": "target",
                "seasonality": 1,
                "strategy": "forward_fill",
                "window": -1,
            }
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-t5-large.zip",
            "encoder_length": 512,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
]


H_SUPER_FAST = [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.NaiveModel", "lag": 1},
        "transforms": [],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.NaiveModel", "lag": 24},
        "transforms": [],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.NaiveModel", "lag": "${horizon}"},
        "transforms": [],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": 24, "window": 3},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 72,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-tiny.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-mini.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-small.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.DeadlineMovingAverageModel", "seasonality": "month", "window": 1},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "window_size": 72,
                "distance_coef": 3,
                "in_column": "target",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
        ],
    },
    {
        "_target_": "etna.ensembles.VotingEnsemble",
        "pipelines": [
            {
                "_target_": "etna.pipeline.Pipeline",
                "horizon": "${__aux__.horizon}",
                "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": 48, "window": 1},
                "transforms": [],
            },
            {
                "_target_": "etna.pipeline.Pipeline",
                "horizon": "${__aux__.horizon}",
                "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": 24, "window": 2},
                "transforms": [],
            },
            {
                "_target_": "etna.pipeline.Pipeline",
                "horizon": "${__aux__.horizon}",
                "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": 24, "window": 7},
                "transforms": [],
            },
        ],
    },
]

H_FAST = H_SUPER_FAST + [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.StatsForecastAutoCESModel", "season_length": 24},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 72,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.ProphetModel",
            "seasonality_mode": "multiplicative",
            "timestamp_column": "${__aux__.timestamp_column}",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 72,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.ProphetModel",
            "seasonality_mode": "additive",
            "timestamp_column": "${__aux__.timestamp_column}",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 72,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.StatsForecastAutoThetaModel",
            "decomposition_type": "multiplicative",
            "season_length": 24,
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 72,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.TimesFMModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/timesfm/timesfm-1.0-200m-pytorch.ckpt",
            "encoder_length": 512,
            "device": "${__aux__.timesfm_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-base.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {"_target_": "etna.transforms.SegmentEncoderTransform"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}",
                "out_column": "target_lag",
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.ElasticMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {"_target_": "etna.transforms.SegmentEncoderTransform"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}",
                "out_column": "target_lag",
            },
        ],
    },
]

H_MEDIUM = H_FAST + [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.ElasticPerSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "constant"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168]}",
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.SegmentEncoderTransform"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]}",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_month": True,
                "day_number_in_week": True,
                "is_weekend": True,
                "month_number_in_year": True,
                "week_number_in_month": True,
                "week_number_in_year": True,
                "year_number": True,
            },
            {
                "_target_": "etna.transforms.TimeFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "fifteen_minutes_in_hour_number": False,
                "half_day_number": True,
                "half_hour_number": False,
                "hour_number": True,
                "minute_in_hour_number": False,
                "one_third_day_number": True,
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.HoltWintersModel",
            "damped_trend": True,
            "seasonal": "add",
            "seasonal_periods": 24,
            "trend": "add",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 72,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.StatsForecastAutoETSModel", "season_length": 24},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            }
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.TimesFMModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/timesfm/timesfm-2.0-500m-pytorch.ckpt",
            "encoder_length": 2048,
            "num_layers": 50,
            "use_positional_embedding": False,
            "device": "${__aux__.timesfm_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "window_size": 72,
                "distance_coef": 3,
                "in_column": "target",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}",
                "out_column": "target_lag",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_week": True,
                "day_number_in_month": False,
                "day_number_in_year": False,
                "week_number_in_month": False,
                "week_number_in_year": False,
                "month_number_in_year": False,
                "year_number": False,
                "is_weekend": True,
                "out_column": "dt",
            },
            {
                "_target_": "etna.transforms.TimeFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "minute_in_hour_number": False,
                "fifteen_minutes_in_hour_number": False,
                "hour_number": True,
                "half_hour_number": False,
                "half_day_number": False,
                "one_third_day_number": False,
                "out_column": "ts",
            },
            {
                "_target_": "etna.transforms.EmbeddingSegmentTransform",
                "in_columns": ["target"],
                "embedding_model": {
                    "_target_": "etna.transforms.embeddings.models.TS2VecEmbeddingModel.load",
                    "model_name": "ts2vec_tiny",
                    "_mode_": "call",
                },
                "training_params": {},
                "out_column": "emb",
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "window_size": 72,
                "distance_coef": 3,
                "in_column": "target",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}",
                "out_column": "target_lag",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_week": True,
                "day_number_in_month": False,
                "day_number_in_year": False,
                "week_number_in_month": False,
                "week_number_in_year": False,
                "month_number_in_year": False,
                "year_number": False,
                "is_weekend": True,
                "out_column": "dt",
            },
            {
                "_target_": "etna.transforms.TimeFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "minute_in_hour_number": False,
                "fifteen_minutes_in_hour_number": False,
                "hour_number": True,
                "half_hour_number": False,
                "half_day_number": False,
                "one_third_day_number": False,
                "out_column": "ts",
            },
            {
                "_target_": "etna.transforms.EmbeddingSegmentTransform",
                "in_columns": ["target"],
                "embedding_model": {
                    "_target_": "etna.transforms.embeddings.models.TSTCCEmbeddingModel.load",
                    "model_name": "tstcc_medium",
                    "_mode_": "call",
                },
                "training_params": {},
                "out_column": "emb",
            },
        ],
    },
]


H_HEAVY = H_MEDIUM + [
    {
        "_target_": "etna.pipeline.AutoRegressivePipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "step": 24,
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.SegmentEncoderTransform"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${step},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743]}",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_month": True,
                "day_number_in_week": True,
                "is_weekend": True,
                "month_number_in_year": True,
                "week_number_in_month": True,
                "week_number_in_year": True,
                "year_number": True,
            },
            {
                "_target_": "etna.transforms.TimeFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "fifteen_minutes_in_hour_number": False,
                "half_day_number": True,
                "half_hour_number": False,
                "hour_number": True,
                "minute_in_hour_number": False,
                "one_third_day_number": True,
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.ElasticMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.DensityOutliersTransform", "in_column": "target"},
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "default_value": 0,
                "in_column": "target",
                "strategy": "mean",
            },
            {"_target_": "etna.transforms.MeanSegmentEncoderTransform"},
            {"_target_": "etna.transforms.FourierTransform", "order": 30, "period": 729.6},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}",
                "out_column": "target_lag",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_month": False,
                "day_number_in_week": True,
                "is_weekend": True,
                "month_number_in_year": False,
                "out_column": "date",
                "week_number_in_month": False,
                "week_number_in_year": False,
                "year_number": False,
            },
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "date_day_number_in_week"},
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "date_is_weekend"},
            {
                "_target_": "etna.transforms.TimeFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "fifteen_minutes_in_hour_number": True,
                "half_day_number": True,
                "half_hour_number": False,
                "hour_number": True,
                "minute_in_hour_number": False,
                "one_third_day_number": True,
                "out_column": "time",
            },
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "time_half_day_number"},
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "time_hour_number"},
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "time_one_third_day_number"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.ElasticPerSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.DensityOutliersTransform", "in_column": "target"},
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "default_value": 0,
                "in_column": "target",
                "strategy": "mean",
            },
            {"_target_": "etna.transforms.FourierTransform", "order": 30, "period": 729.6},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]}",
                "out_column": "target_lag",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_month": False,
                "day_number_in_week": True,
                "is_weekend": True,
                "month_number_in_year": False,
                "out_column": "date",
                "week_number_in_month": False,
                "week_number_in_year": False,
                "year_number": False,
            },
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "date_day_number_in_week"},
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "date_is_weekend"},
            {
                "_target_": "etna.transforms.TimeFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "fifteen_minutes_in_hour_number": False,
                "half_day_number": True,
                "half_hour_number": False,
                "hour_number": True,
                "minute_in_hour_number": False,
                "one_third_day_number": True,
                "out_column": "time",
            },
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "time_half_day_number"},
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "time_hour_number"},
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "time_one_third_day_number"},
            {"_target_": "etna.transforms.DeseasonalityTransform", "in_column": "target", "period": 24},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.StatsForecastAutoARIMAModel",
            "kwargs": {"nmodels": 10},
            "max_D": 1,
            "max_P": 6,
            "max_Q": 6,
            "max_d": 2,
            "max_order": 5,
            "max_p": 12,
            "max_q": 12,
            "season_length": 24,
            "start_P": 1,
            "start_Q": 1,
            "start_p": 2,
            "start_q": 2,
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            }
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-t5-large.zip",
            "encoder_length": 512,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
]


MS_SUPER_FAST = [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.NaiveModel", "lag": 1},
        "transforms": [],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.NaiveModel", "lag": 12},
        "transforms": [],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.NaiveModel", "lag": "${horizon}"},
        "transforms": [],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-tiny.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-mini.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-small.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.ensembles.VotingEnsemble",
        "pipelines": [
            {
                "_target_": "etna.pipeline.Pipeline",
                "horizon": "${__aux__.horizon}",
                "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": 12, "window": 1},
                "transforms": [],
            },
            {
                "_target_": "etna.pipeline.Pipeline",
                "horizon": "${__aux__.horizon}",
                "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": 6, "window": 2},
                "transforms": [],
            },
            {
                "_target_": "etna.pipeline.Pipeline",
                "horizon": "${__aux__.horizon}",
                "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": 12, "window": 5},
                "transforms": [],
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": 1, "window": 3},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "window_size": 24,
                "distance_coef": 3,
                "in_column": "target",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
        ],
    },
]

MS_FAST = MS_SUPER_FAST + [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.StatsForecastAutoCESModel", "season_length": 1},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 10,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.StatsForecastAutoThetaModel",
            "decomposition_type": "additive",
            "season_length": 12,
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 10,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.StatsForecastAutoThetaModel",
            "decomposition_type": "multiplicative",
            "season_length": 12,
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 10,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.TimesFMModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/timesfm/timesfm-1.0-200m-pytorch.ckpt",
            "encoder_length": 512,
            "device": "${__aux__.timesfm_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-base.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "window_size": 24,
                "distance_coef": 3,
                "in_column": "target",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}",
                "out_column": "target_lag",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_week": False,
                "day_number_in_month": False,
                "day_number_in_year": False,
                "week_number_in_month": False,
                "week_number_in_year": False,
                "month_number_in_year": True,
                "year_number": False,
                "is_weekend": False,
                "out_column": "dt",
            },
            {
                "_target_": "etna.transforms.EmbeddingSegmentTransform",
                "in_columns": ["target"],
                "embedding_model": {
                    "_target_": "etna.transforms.embeddings.models.TS2VecEmbeddingModel.load",
                    "model_name": "ts2vec_tiny",
                    "_mode_": "call",
                },
                "training_params": {},
                "out_column": "emb",
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "window_size": 24,
                "distance_coef": 3,
                "in_column": "target",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}",
                "out_column": "target_lag",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_week": False,
                "day_number_in_month": False,
                "day_number_in_year": False,
                "week_number_in_month": False,
                "week_number_in_year": False,
                "month_number_in_year": True,
                "year_number": False,
                "is_weekend": False,
                "out_column": "dt",
            },
            {
                "_target_": "etna.transforms.EmbeddingSegmentTransform",
                "in_columns": ["target"],
                "embedding_model": {
                    "_target_": "etna.transforms.embeddings.models.TSTCCEmbeddingModel.load",
                    "model_name": "tstcc_medium",
                    "_mode_": "call",
                },
                "training_params": {},
                "out_column": "emb",
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {"_target_": "etna.transforms.SegmentEncoderTransform"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}",
                "out_column": "target_lag",
            },
        ],
    },
]

MS_MEDIUM = MS_FAST + [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.SARIMAXModel",
            "order": [1, 0, 1],
            "fit_params": {"disp": 0},
            "enforce_stationarity": False,
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            }
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.ProphetModel",
            "seasonality_mode": "additive",
            "timestamp_column": "${__aux__.timestamp_column}",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 10,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.HoltWintersModel", "damped_trend": False, "seasonal": None, "trend": None},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 10,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.ProphetModel",
            "seasonality_mode": "multiplicative",
            "timestamp_column": "${__aux__.timestamp_column}",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 10,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            },
            {"_target_": "etna.transforms.SegmentEncoderTransform"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {"_target_": "etna.transforms.LinearTrendTransform", "in_column": "target"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5]}",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_month": False,
                "day_number_in_week": False,
                "is_weekend": False,
                "month_number_in_year": True,
                "week_number_in_month": False,
                "week_number_in_year": False,
                "year_number": True,
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.StatsForecastAutoETSModel", "season_length": 12},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 10,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.TimesFMModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/timesfm/timesfm-2.0-500m-pytorch.ckpt",
            "encoder_length": 2048,
            "num_layers": 50,
            "use_positional_embedding": False,
            "device": "${__aux__.timesfm_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.StatsForecastAutoCESModel", "season_length": 12},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
]


MS_HEAVY = MS_MEDIUM + [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.statsforecast.StatsForecastAutoARIMAModel",
            "kwargs": {"nmodels": 10},
            "max_D": 1,
            "max_P": 1,
            "max_Q": 1,
            "max_d": 2,
            "max_order": 5,
            "max_p": 2,
            "max_q": 2,
            "season_length": 12,
            "start_P": 1,
            "start_Q": 1,
            "start_p": 2,
            "start_q": 2,
        },
        "transforms": [
            {
                "_target_": "etna.transforms.missing_values.imputation.TimeSeriesImputerTransform",
                "constant_value": 0,
                "in_column": "target",
                "seasonality": 1,
                "strategy": "forward_fill",
                "window": -1,
            }
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-t5-large.zip",
            "encoder_length": 512,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.HoltWintersModel",
            "damped_trend": True,
            "seasonal": "add",
            "seasonal_periods": 12,
            "trend": "add",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 10,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
        ],
    },
]


W_SUPER_FAST = [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.NaiveModel", "lag": 1},
        "transforms": [],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.NaiveModel", "lag": 52},
        "transforms": [],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.NaiveModel", "lag": "${horizon}"},
        "transforms": [],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-tiny.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-mini.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-small.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.ensembles.VotingEnsemble",
        "pipelines": [
            {
                "_target_": "etna.pipeline.Pipeline",
                "horizon": "${__aux__.horizon}",
                "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": 52, "window": 1},
                "transforms": [],
            },
            {
                "_target_": "etna.pipeline.Pipeline",
                "horizon": "${__aux__.horizon}",
                "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": 26, "window": 2},
                "transforms": [],
            },
            {
                "_target_": "etna.pipeline.Pipeline",
                "horizon": "${__aux__.horizon}",
                "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": 4, "window": 12},
                "transforms": [],
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.SeasonalMovingAverageModel", "seasonality": 1, "window": 12},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "window_size": 13,
                "distance_coef": 3,
                "in_column": "target",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
        ],
    },
]

W_FAST = W_SUPER_FAST + [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.ElasticMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.FourierTransform", "order": 10, "period": 52},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}",
                "out_column": "target_lag",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_month": False,
                "day_number_in_week": False,
                "is_weekend": False,
                "month_number_in_year": False,
                "out_column": "date",
                "week_number_in_month": True,
                "week_number_in_year": False,
                "year_number": False,
            },
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "date_week_number_in_month"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.StatsForecastAutoThetaModel",
            "decomposition_type": "additive",
            "season_length": 1,
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 10,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.ElasticPerSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.DensityOutliersTransform", "in_column": "target"},
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "default_value": 0,
                "in_column": "target",
                "strategy": "mean",
            },
            {"_target_": "etna.transforms.FourierTransform", "order": 10, "period": 52},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}",
                "out_column": "target_lag",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_month": False,
                "day_number_in_week": False,
                "is_weekend": False,
                "month_number_in_year": False,
                "out_column": "date",
                "week_number_in_month": True,
                "week_number_in_year": False,
                "year_number": False,
            },
            {"_target_": "etna.transforms.OneHotEncoderTransform", "in_column": "date_week_number_in_month"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.StatsForecastAutoCESModel", "season_length": 1},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 10,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.TimesFMModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/timesfm/timesfm-1.0-200m-pytorch.ckpt",
            "encoder_length": 512,
            "device": "${__aux__.timesfm_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosBoltModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-bolt-base.zip",
            "encoder_length": 2048,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {"_target_": "etna.transforms.SegmentEncoderTransform"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}",
                "out_column": "target_lag",
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.ElasticPerSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}",
                "out_column": "target_lag",
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.StatsForecastAutoThetaModel",
            "decomposition_type": "multiplicative",
            "season_length": 13,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
]

W_MEDIUM = W_FAST + [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.HoltWintersModel", "damped_trend": False, "seasonal": None, "trend": None},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 10,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.ProphetModel",
            "seasonality_mode": "multiplicative",
            "timestamp_column": "${__aux__.timestamp_column}",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 10,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.ProphetModel",
            "seasonality_mode": "additive",
            "timestamp_column": "${__aux__.timestamp_column}",
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            }
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            },
            {"_target_": "etna.transforms.SegmentEncoderTransform"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {"_target_": "etna.transforms.LinearTrendTransform", "in_column": "target"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_month": False,
                "day_number_in_week": False,
                "is_weekend": False,
                "month_number_in_year": True,
                "week_number_in_month": True,
                "week_number_in_year": True,
                "year_number": True,
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.TimesFMModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/timesfm/timesfm-2.0-500m-pytorch.ckpt",
            "encoder_length": 2048,
            "num_layers": 50,
            "use_positional_embedding": False,
            "device": "${__aux__.timesfm_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "window_size": 13,
                "distance_coef": 3,
                "in_column": "target",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}",
                "out_column": "target_lag",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_week": False,
                "day_number_in_month": False,
                "day_number_in_year": False,
                "week_number_in_month": True,
                "week_number_in_year": False,
                "month_number_in_year": True,
                "year_number": False,
                "is_weekend": False,
                "out_column": "dt",
            },
            {
                "_target_": "etna.transforms.EmbeddingSegmentTransform",
                "in_columns": ["target"],
                "embedding_model": {
                    "_target_": "etna.transforms.embeddings.models.TS2VecEmbeddingModel.load",
                    "model_name": "ts2vec_tiny",
                    "_mode_": "call",
                },
                "training_params": {},
                "out_column": "emb",
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostMultiSegmentModel"},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "window_size": 13,
                "distance_coef": 3,
                "in_column": "target",
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}",
                "out_column": "target_lag",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_week": False,
                "day_number_in_month": False,
                "day_number_in_year": False,
                "week_number_in_month": True,
                "week_number_in_year": False,
                "month_number_in_year": True,
                "year_number": False,
                "is_weekend": False,
                "out_column": "dt",
            },
            {
                "_target_": "etna.transforms.EmbeddingSegmentTransform",
                "in_columns": ["target"],
                "embedding_model": {
                    "_target_": "etna.transforms.embeddings.models.TSTCCEmbeddingModel.load",
                    "model_name": "tstcc_medium",
                    "_mode_": "call",
                },
                "training_params": {},
                "out_column": "emb",
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.StatsForecastAutoCESModel", "season_length": 13},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.StatsForecastAutoETSModel", "season_length": 13},
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
]


W_HEAVY = W_MEDIUM + [
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.HoltWintersModel",
            "trend": "add",
            "damped_trend": True,
            "seasonal": "add",
            "seasonal_periods": 13,
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            }
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.SARIMAXModel",
            "order": [4, 0, 4],
            "fit_params": {"disp": 0},
            "enforce_stationarity": False,
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            }
        ],
    },
    {
        "_target_": "etna.pipeline.AutoRegressivePipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostPerSegmentModel"},
        "step": 1,
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "constant_value": 0,
                "in_column": "target",
                "strategy": "constant",
            },
            {
                "_target_": "etna.transforms.ChangePointsTrendTransform",
                "change_points_model": {
                    "_target_": "etna.transforms.decomposition.change_points_based.RupturesChangePointsModel",
                    "change_points_model": {
                        "_target_": "ruptures.detection.Binseg",
                        "jump": 5,
                        "min_size": 2,
                        "model": "ar",
                    },
                    "n_bkps": 3,
                },
                "in_column": "target",
            },
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${step},[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}",
                "out_column": "regressor_lag",
            },
            {
                "_target_": "etna.transforms.MeanTransform",
                "alpha": 1,
                "fillna": 0,
                "in_column": "regressor_lag_${step}",
                "min_periods": 1,
                "seasonality": 1,
                "window": 4,
            },
            {
                "_target_": "etna.transforms.MedianTransform",
                "fillna": 0,
                "in_column": "regressor_lag_${step}",
                "min_periods": 1,
                "seasonality": 1,
                "window": 4,
            },
            {
                "_target_": "etna.transforms.QuantileTransform",
                "fillna": 0,
                "in_column": "regressor_lag_${step}",
                "min_periods": 1,
                "quantile": 0.25,
                "seasonality": 1,
                "window": 4,
            },
            {
                "_target_": "etna.transforms.QuantileTransform",
                "fillna": 0,
                "in_column": "regressor_lag_${step}",
                "min_periods": 1,
                "quantile": 0.75,
                "seasonality": 1,
                "window": 4,
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_month": False,
                "day_number_in_week": False,
                "is_weekend": False,
                "month_number_in_year": False,
                "week_number_in_month": True,
                "week_number_in_year": True,
                "year_number": True,
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.StatsForecastAutoARIMAModel",
            "kwargs": {"nmodels": 10},
            "max_D": 1,
            "max_P": 2,
            "max_Q": 2,
            "max_d": 2,
            "max_order": 5,
            "max_p": 3,
            "max_q": 3,
            "season_length": 26,
            "start_P": 1,
            "start_Q": 1,
            "start_p": 2,
            "start_q": 2,
        },
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            }
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.StatsForecastAutoETSModel", "season_length": 52},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
                "window": -1,
            },
            {
                "_target_": "etna.transforms.DensityOutliersTransform",
                "distance_coef": 3,
                "in_column": "target",
                "window_size": 10,
            },
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"},
            {"_target_": "etna.transforms.RobustScalerTransform", "in_column": "target", "mode": "per-segment"},
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {"_target_": "etna.models.CatBoostPerSegmentModel"},
        "transforms": [
            {
                "_target_": "etna.transforms.TimeSeriesImputerTransform",
                "in_column": "target",
                "strategy": "forward_fill",
            },
            {
                "_target_": "etna.transforms.LagTransform",
                "in_column": "target",
                "lags": "${shift:${horizon},[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}",
            },
            {
                "_target_": "etna.transforms.DateFlagsTransform",
                "in_column": "${__aux__.timestamp_column}",
                "day_number_in_month": False,
                "day_number_in_week": False,
                "is_weekend": False,
                "month_number_in_year": True,
                "week_number_in_month": True,
                "week_number_in_year": True,
                "year_number": True,
            },
        ],
    },
    {
        "_target_": "etna.pipeline.Pipeline",
        "horizon": "${__aux__.horizon}",
        "model": {
            "_target_": "etna.models.nn.ChronosModel",
            "path_or_url": "http://etna-github-prod.cdn-tinkoff.ru/chronos/chronos-t5-large.zip",
            "encoder_length": 512,
            "device": "${__aux__.chronos_device}",
            "batch_size": 128,
        },
        "transforms": [
            {"_target_": "etna.transforms.TimeSeriesImputerTransform", "in_column": "target", "strategy": "mean"}
        ],
    },
]
