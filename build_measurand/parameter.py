from typing import Tuple
from functools import cached_property
import numpy as np
from pydantic import BaseModel, Field
from .utils import _expand_component_range, _size_to_uint
from .component import Component


def make_parameter(
    spec: str, word_size: int = 8, one_based: bool = True
) -> "Parameter":
    raw_param = _expand_component_range(spec)
    components = (
        Component.from_spec(s, one_based=one_based, word_size=word_size)
        for s in raw_param.split("+")
    )
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

    @classmethod
    def from_spec(cls, spec: str, word_size=word_size) -> "Parameter":
        return make_parameter(spec, word_size=word_size)

    def build(self, data: np.ndarray) -> np.ndarray:
        tmp = np.atleast_2d(data)
        print("aA", repr(tmp))

        dtype = _size_to_uint(self.size)
        print("result dtype =", dtype)
        print("aB", repr(tmp))

        result = np.zeros(tmp.shape[0], dtype=dtype)
        size = 0
        for comp in reversed(self.components):
            result += comp.build(tmp).astype(dtype) << size
            size += comp.size

        print("aC", repr(result))

        return result
