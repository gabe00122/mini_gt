from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import pandas as pd
from jax.typing import ArrayLike


type Metrics = dict[str, ArrayLike]


class Writter(ABC):
    @abstractmethod
    def write(self, metrics: Metrics) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...


class PandasWritter(Writter):
    data: dict[str, Any] = {}

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def write(self, metrics: Metrics) -> None:
        for key, value in metrics.items():
            self.data.setdefault(key, []).append(value.item())

    def flush(self) -> None:
        df = pd.DataFrame(self.data)
        df.to_parquet(self.output_path)
