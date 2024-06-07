from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple
from pathlib import Path
from functools import singledispatchmethod
import pandas as pd
from jax import Array, numpy as jnp
from jax.typing import ArrayLike


type Metrics = dict[str, Array]


class MetricsBuffer(NamedTuple):
    values: Metrics
    length: Array


class Writer(ABC):
    @abstractmethod
    def write(self, metrics: Metrics | MetricsBuffer) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...


class PandasWriter(Writer):
    data: dict[str, Any] = {}

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    @singledispatchmethod
    def write(self, metrics: Metrics | MetricsBuffer) -> None:
        pass

    #@write.register
    #def _(self, metrics: dict[str, Array]) -> None:
    #    for key, value in metrics.items():
    #        self.data.setdefault(key, []).append(value.item())

    @write.register
    def _(self, metrics: MetricsBuffer) -> None:
        for key, value in metrics.values.items():
            value_slice = value[:metrics.length].tolist()
            self.data.setdefault(key, []).extend(value_slice)

    def flush(self) -> None:
        df = pd.DataFrame(self.data)
        df.to_parquet(self.output_path)



def create_metrics_buffer(metrics: Metrics, capacity: ArrayLike) -> MetricsBuffer:
    def create_metric(value: Array) -> Array:
        shape = (capacity, *value.shape)
        metric = jnp.zeros(shape, dtype=value.dtype)
        metric = metric.at[0].set(value)

        return metric

    values = { k: create_metric(v) for k, v in metrics.items() }

    return MetricsBuffer(values=values, length=jnp.int32(0))


def append_buffer(metrics_buffer: MetricsBuffer, metrics: Metrics) -> MetricsBuffer:
    def append_value(buffer_value: Array, update: Array):
        return buffer_value.at[metrics_buffer.length].set(update)

    values = { k: append_value(v, metrics[k]) for k, v in metrics_buffer.values.items() }

    return MetricsBuffer(values=values, length=metrics_buffer.length + 1)
