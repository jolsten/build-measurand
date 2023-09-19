from functools import cached_property
from typing import List, Optional
import numpy as np
import pyarrow as pa
from pydantic import BaseModel
from .parameter import Parameter, make_parameter
from .interp import Interp, make_interp
from .euc import EUC, make_euc

# from .sampling import SamplingStrategy


class Measurand(BaseModel):
    parameter: Parameter
    interp: Optional[Interp] = None
    euc: Optional[EUC] = None
    # sampling: Optional[Sampling] = None

    @cached_property
    def size(self) -> int:
        return self.parameter.size

    @classmethod
    def from_spec(cls, spec: str) -> "Measurand":
        return make_measurand(spec)

    def _build_ndarray(self, data: np.ndarray) -> np.ndarray:
        tmp = self.parameter._build_ndarray(data)

        if self.interp:
            tmp = self.interp.apply_ndarray(tmp, self.parameter.size)

        if self.euc:
            tmp = self.euc.apply_ndarray(tmp, self.parameter.size)

        return tmp

    def _build_paarray(self, data: pa.Table) -> pa.Array:
        tmp = self.parameter._build_paarray(data)

        if self.interp:
            tmp = self.interp.apply_paarray(tmp, self.parameter.size)

        if self.euc:
            tmp = self.euc.apply_paarray(tmp, self.parameter.size)

        return tmp


def make_measurand(spec: str, word_size: int = 8, one_based: bool = True) -> Measurand:
    parts = spec.split(";")
    parameter = make_parameter(parts[0], word_size=word_size, one_based=one_based)
    mapping = {"parameter": parameter}

    if len(parts) >= 2:
        mapping["interp"] = make_interp(parts[1])

    if len(parts) >= 3:
        mapping["euc"] = make_euc(parts[2])

    return Measurand(**mapping)
