import numpy as np
import matplotlib.pyplot as plt
import optax
import pandas as pd
from pathlib import Path


def main():
    # schedule = optax.warmup_cosine_decay_schedule(0.000001, 0.01, 1_000, 9_000)
    # xs = np.arange(10_000)
    # ys = schedule(xs) * 100
    df = pd.read_parquet(Path("metrics_maybe_fixup_large.parquet"))
    df_old = pd.read_parquet(Path("metrics_glorot.parquet"))

    dfs = [df, df_old]

    # df = df[["loss", "percent_correct",]]
    for d in dfs:
        d = d[["percent_correct"]]
        d = d.rolling(100).mean()
        d.plot()

    # columns = [column for column in df.columns if column.endswith("std")]
    # df = df[columns]
    # print(df.mean().idxmax())

    # df.mean().plot(kind="bar")
    # df.plot()

    # losses = np.load("losses.npy")
    # grad_std = np.load("percent_correct.npy")
    # plt.plot(losses)
    # plt.plot(grad_std)
    plt.show()


def plot_learning_rate():
    schedule = optax.warmup_exponential_decay_schedule(
        0.000001, 0.01, 1_000, 9_000, 0.1
    )
    xs = np.arange(10_000)
    ys = schedule(xs)

    plt.plot(ys)
    plt.show()


if __name__ == "__main__":
    main()
    # plot_learning_rate()
