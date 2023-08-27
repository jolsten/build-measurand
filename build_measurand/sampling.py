from typing import Literal
import numpy as np
from .generic import MeasurandModifier

SamplingStrategy = Literal["mean", "mode"]


class Sampling(MeasurandModifier):
    window: int
    mode: SamplingStrategy

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        raise NotImplementedError
