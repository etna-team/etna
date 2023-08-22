from etna.models import SeasonalMovingAverageModel
from etna.datasets import TSDataset, generate_ar_df
import numpy as np
#from memory_profiler import profile

@profile
def generate_dataset(n_periods, n_segments) -> TSDataset:
    df = generate_ar_df(
        periods=n_periods,
        start_time="2021-06-01",
        n_segments=n_segments,
        freq="D",
    )
    df["target"] = df["target"].astype(np.int32)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")
    return ts

@profile
def run_exp():
    ts = generate_dataset(360, 100000)

    model = SeasonalMovingAverageModel(window=14, seasonality=1)

    model.fit(ts=ts)

    future = ts.make_future(future_steps=14, tail_steps=20)

    model.forecast(future, prediction_size=14)

if __name__ == '__main__':
    run_exp()
