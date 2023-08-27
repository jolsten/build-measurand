from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, ConfigDict


class MeasurandModifier(BaseModel, ABC):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        ...
