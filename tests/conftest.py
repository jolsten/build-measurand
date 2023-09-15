import numpy as np

ARRAY_SIZE = 100

SAMPLE_DATA = {
    8: np.array([np.arange(start=1, stop=257, dtype="u1")] * ARRAY_SIZE),
    10: np.array([np.arange(start=1, stop=1025, dtype="u2")] * ARRAY_SIZE),
    12: np.array([np.arange(start=1, stop=4097, dtype="u2")] * ARRAY_SIZE),
}
