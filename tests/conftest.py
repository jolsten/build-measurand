import numpy as np
from build_measurand.utils import _size_to_uint

ARRAY_SIZE = 100


def _make_sample_data(word_size: int) -> np.ndarray:
    dtype = _size_to_uint(word_size)
    stop = 2**word_size + 1
    one = np.arange(start=1, stop=stop, dtype=np.uint16)
    one = np.fmod(one, np.uint16(2**word_size))
    return np.repeat(np.atleast_2d(one), ARRAY_SIZE, axis=0).astype(dtype)


SAMPLE_NDARRAY = {word_size: _make_sample_data(word_size) for word_size in [8, 10, 12]}
