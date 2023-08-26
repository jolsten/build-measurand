import re
from typing import Tuple
import numpy as np
from numpy.typing import DTypeLike


def _size_to_uint(size: int) -> DTypeLike:
    if size <= 8:
        return np.dtype('u1')
    elif size <= 16:
        return np.dtype('u2')
    elif size <= 32:
        return np.dtype('u4')
    elif size <= 64:
        return np.dtype('u8')
    raise ValueError

def _expand_list(list_spec: str) -> list:
    if '-' in list_spec:
        a, b = list_spec.split('-')

        if not(all([x.isnumeric() for x in (a,b)])):
            raise ValueError('_expand_list() takes a string argument whose values must be integers')
        else:
            a, b = int(a), int(b)

        if a < b:
            return list(range(a, b+1))
        else:
            return list(reversed(range(b, a+1)))
    else:
        return list([int(list_spec)])

_RE_RAWPARAM_RANGE = re.compile(
    r'^(?P<range>\d+-\d+)(?P<rest>.*)$',
    re.IGNORECASE
)

def _expand_component_range(spec: str) -> str:
    component_range = _RE_RAWPARAM_RANGE.match(spec)
    if component_range:
        word_range = component_range.group('range')
        additional = component_range.group('rest')
        words = _expand_list(word_range)
        spec = '+'.join(
            [f'{word}{additional}' for word in words]
        )
    return spec

def _bit_range_to_mask_and_shift(lsb: int, msb: int) -> int:
    if msb < lsb:
        lsb, msb = msb, lsb
    shift = lsb
    mask = int(2**(msb-lsb+1)-1) << lsb
    return mask, shift

def _reverse_bits(
        data: np.ndarray,
        word_size: int,
        size: int,
    ) -> np.ndarray:
    input_dtype = data.dtype
    data = np.atleast_3d(data).view('u1')
    bits = np.unpackbits(
        data,
        axis=2,
        count=word_size,
        bitorder='little',
    )
    data = np.packbits(bits, axis=-1, bitorder='big')
    data = data.view(input_dtype).byteswap().flatten()
    reverse_rshift = (_size_to_uint(word_size).itemsize*8 - size)
    if reverse_rshift:
        data = data >> reverse_rshift
    return data

def _range_to_tuple(spec: str) -> Tuple[int, int]:
    parts = spec.split('-')
    if len(parts) == 1:
        return int(spec), int(spec)
    elif len(parts) == 2:
        a, b = [int(x) for x in parts]
        if a > b:
            a, b = b, a
        return a, b