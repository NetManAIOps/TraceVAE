from typing import *

import numpy as np

__all__ = [
    'compute_cdf',
]


def compute_cdf(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # calculate bin size cdf
    hist = {}
    for v in arr:
        if v not in hist:
            hist[v] = 0
        hist[v] += 1

    keys = np.array(sorted(hist))
    values = np.array([hist[v] for v in keys], dtype=np.float64)
    values /= values.sum()
    values = np.cumsum(values)

    return keys, values
