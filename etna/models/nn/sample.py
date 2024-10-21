from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.models.nn import RNNModel
from etna.pipeline import Pipeline
from etna.transforms import LagTransform

HORIZON = 5
num_lags = 5
periods = 100
n_segments = 1
freq = "10T"

df = generate_ar_df(start_time="1800-01-01", n_segments=n_segments, periods=periods, freq=freq)
df["target"] = 1
df_exog = generate_ar_df(start_time="1800-01-01", n_segments=N_SEGMENTS, periods=PERIODS + HORIZON + 1, freq=FREQ)
df_exog = df_exog.drop(columns=["target"])
ts = TSDataset(df=df, freq=freq)

model_rnn = RNNModel(
    input_size=6,
    encoder_length=HORIZON,
    decoder_length=HORIZON,
    trainer_params=dict(max_epochs=1),
    lr=1e-3,
)

pipeline_rnn = Pipeline(
    model=model_rnn,
    horizon=HORIZON,
    transforms=[transform_lag],
)

pipeline_rnn.fit(ts)
a = pipeline_rnn.forecast()
