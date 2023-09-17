from typing import Tuple
from functools import cached_property
import numpy as np
import pyarrow as pa
import pyarrow.compute as pac
from pydantic import BaseModel, Field
from .utils import _expand_component_range, _size_to_uint
from .component import make_component, Component


def make_parameter(
    spec: str, word_size: int = 8, one_based: bool = True
) -> "Parameter":
    raw_param = _expand_component_range(spec)
    components = [
        make_component(s, one_based=one_based, word_size=word_size)
        for s in raw_param.split("+")
    ]
    print(components)
    return Parameter(components=components, word_size=word_size, one_based=one_based)


class Parameter(BaseModel):
    components: Tuple[Component, ...]
    one_based: bool = Field(default=True, frozen=True)
    word_size: int = Field(default=8, frozen=True)

    def __eq__(self, other: "Parameter") -> bool:
        if len(self.components) != len(other.components):
            return False
        for c1, c2 in zip(self.components, other.components):
            if c1 != c2:
                return False
        return True

    @cached_property
    def size(self) -> int:
        return sum([c.size for c in self.components])

    @cached_property
    def output_dtype(self) -> str:
        return _size_to_uint(self.size)

    @classmethod
    def from_spec(cls, spec: str, word_size=word_size) -> "Parameter":
        return make_parameter(spec, word_size=word_size)

    def build_ndarray(self, data: np.ndarray) -> np.ndarray:
        tmp = np.atleast_2d(data)
        dtype = _size_to_uint(self.size)
        result = np.zeros(tmp.shape[0], dtype=dtype)
        size = 0
        for comp in reversed(self.components):
            result += comp.build_ndarray(tmp).astype(dtype) << size
            size += comp.size
        return result

    def build_paarray(self, data: pa.Table) -> pa.Array:
        if not isinstance(data, pa.Table):
            raise TypeError

        result = pa.array(np.zeros(len(data), dtype=self.output_dtype))
        size = 0
        for comp in reversed(self.components):
            tmp = comp.build_paarray(data).cast(self.output_dtype)
            tmp = pac.shift_left(tmp, np.uint8(size))
            result = pac.add(result, tmp)
            size += comp.size
        return result
