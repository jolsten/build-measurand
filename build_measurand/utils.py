import re
from typing import Tuple
import numpy as np
import pyarrow as pa
from numpy.typing import DTypeLike


def _size_to_uint(size: int) -> DTypeLike:
    if size <= 8:
        return np.dtype(np.uint8)
    elif size <= 16:
        return np.dtype(np.uint16)
    elif size <= 32:
        return np.dtype(np.uint32)
    elif size <= 64:
        return np.dtype(np.uint64)
    raise ValueError


def _expand_list(list_spec: str) -> list:
    if "-" in list_spec:
        a, b = list_spec.split("-")

        if not (all([x.isnumeric() for x in (a, b)])):
            raise ValueError(
                "_expand_list() takes a string argument whose values must be integers"
            )
        else:
            a, b = int(a), int(b)

        if a < b:
            return list(range(a, b + 1))
        else:
            return list(reversed(range(b, a + 1)))
    else:
        return list([int(list_spec)])


_RE_RAWPARAM_RANGE = re.compile(r"^(?P<range>\d+-\d+)(?P<rest>.*)$", re.IGNORECASE)


def _expand_component_range(spec: str) -> str:
    component_range = _RE_RAWPARAM_RANGE.match(spec)
    if component_range:
        word_range = component_range.group("range")
        additional = component_range.group("rest")
        words = _expand_list(word_range)
        spec = "+".join([f"{word}{additional}" for word in words])
    return spec


def _bit_range_to_mask_and_shift(lsb: int, msb: int) -> int:
    if msb < lsb:
        lsb, msb = msb, lsb
    shift = lsb
    mask = int(2 ** (msb - lsb + 1) - 1) << lsb
    return mask, shift


def _reverse_bits(x: np.ndarray, actual_size: int) -> np.ndarray:
    tmp = np.ascontiguousarray(x)
    dtype = x.dtype
    result = np.flip(
        np.ascontiguousarray(
            np.packbits(np.flip(np.unpackbits(tmp.view(np.uint8))))
        ).view(dtype)
    )
    shift = result.dtype.itemsize * 8 - actual_size
    if shift:
        result = np.right_shift(result, np.uint8(shift))
    return result


def _range_to_tuple(spec: str) -> Tuple[int, int]:
    parts = spec.split("-")
    if len(parts) == 1:
        return int(spec), int(spec)
    elif len(parts) == 2:
        a, b = [int(x) for x in parts]
        if a > b:
            a, b = b, a
        return a, b


def _numpy_2d_array_to_arrow_table(array: np.ndarray) -> pa.Table:
    arrays = [pa.array(col) for col in array.T]
    table = pa.Table.from_arrays(arrays, names=[str(i) for i in range(len(arrays))])
    return table
