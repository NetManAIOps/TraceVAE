from typing import *

import numpy as np

from tracegnn.constants import *


if not USE_MULTI_DIM_LATENCY_CODEC:
    __all__ = []

else:
    __all__ = [
        'encode_multi_latency',
        'decode_multi_latency',
        'encode_latency',
        'decode_latency',
    ]

    EPS = 1e-6


    def encode_multi_latency(latencies: Sequence[np.ndarray],
                             max_latency_dims: int
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode multiple latencies into (codec, onehot) feature vectors.

        If `max_latency_dims` is sufficient:

        >>> latencies = [np.array([0.0, 9.6, 10.3, 58.7, 101.2]), np.array([11.3, 0.6, 0.0, 99.1, 100.0])]
        >>> codec, onehot = encode_multi_latency(latencies, 3)
        >>> codec
        array([[-1.  , -1.  , -1.  , -0.74, -0.8 , -1.  ],
               [ 0.92, -1.  , -1.  , -0.88, -1.  , -1.  ],
               [-0.94, -0.8 , -1.  , -1.  , -1.  , -1.  ],
               [ 0.74,  0.  , -1.  ,  0.82,  0.8 , -1.  ],
               [-0.76, -1.  , -0.8 , -1.  , -1.  , -0.8 ]])
        >>> onehot
        array([[ True, False, False, False,  True, False],
               [ True, False, False,  True, False, False],
               [False,  True, False,  True, False, False],
               [False,  True, False, False,  True, False],
               [False, False,  True, False, False,  True]])
        >>> decode_multi_latency(codec, onehot, 3)
        [array([  0. ,   9.6,  10.3,  58.7, 101.2]), array([ 11.3,   0.6,   0. ,  99.1, 100. ])]

        If `max_latency_dims` is partially sufficient:

        >>> latencies = [np.array([9.6, 10.3, 58.7, 101.2]), np.array([11.3, 0.6, 99.1, 100.0])]
        >>> codec, onehot = encode_multi_latency(latencies, 2)
        >>> codec
        array([[ 0.92, -1.  , -0.74, -0.8 ],
               [-0.94, -0.8 , -0.88, -1.  ],
               [ 0.74,  0.  ,  0.82,  0.8 ],
               [-0.76,  1.  , -1.  ,  1.  ]])
        >>> onehot
        array([[ True, False, False,  True],
               [False,  True,  True, False],
               [False,  True, False,  True],
               [False,  True, False,  True]])
        >>> decode_multi_latency(codec, onehot, 2)
        [array([  9.6,  10.3,  58.7, 101.2]), array([ 11.3,   0.6,  99.1, 100. ])]

        If `max_latency_dims` is insufficient:

        >>> latencies = [np.array([9.6, 10.3, 58.7, 101.2]), np.array([11.3, 0.6, 99.1, 100.0])]
        >>> codec, onehot = encode_multi_latency(latencies, 1)
        >>> codec
        array([[ 0.92,  1.26],
               [ 1.06, -0.88],
               [10.74, 18.82],
               [19.24, 19.  ]])
        >>> onehot
        array([[ True,  True],
               [ True,  True],
               [ True,  True],
               [ True,  True]])
        >>> decode_multi_latency(codec, onehot, 1)
        [array([  9.6,  10.3,  58.7, 101.2]), array([ 11.3,   0.6,  99.1, 100. ])]
        """
        codec, onehot = [], []
        for residual in latencies:
            for i in range(max_latency_dims - 1):
                if i == 0:
                    onehot.append(residual < 10)
                else:
                    onehot.append(np.logical_and(EPS < residual, residual < 10))
                r = residual % 10
                codec.append(r)
                residual = (residual - r) / 10
            onehot.append(EPS < residual)
            codec.append(residual)
        codec, onehot = np.stack(codec, axis=-1), np.stack(onehot, axis=-1)
        codec = codec / 5. - 1  # scale to [-1, 1]
        return codec, onehot


    def decode_multi_latency(codec: np.ndarray,
                             onehot: np.ndarray,
                             max_latency_dims: int
                             ) -> List[np.ndarray]:
        if codec.shape[-1] % max_latency_dims != 0:
            raise ValueError(
                f'arr.shape[-1] % max_latency_dims != 0: '
                f'arr.shape = {codec.shape!r}, where max_latency_dims = {max_latency_dims!r}'
            )

        ret = []
        codec = (np.clip(codec, -1, 1) + 1) * 5  # scale back from [-1, 1]
        for i in range(codec.shape[-1] // max_latency_dims):
            left = i * max_latency_dims
            right = left + max_latency_dims - 1
            m = onehot[..., right]
            r = codec[..., right] * m.astype(np.float32)
            while right > left:
                r = r * 10
                right -= 1
                m |= onehot[..., right]
                r += codec[..., right]
            ret.append(r)

        return ret


    def encode_latency(latency: np.ndarray,
                       max_latency_dims: int
                       ) -> Tuple[np.ndarray, np.ndarray]:
        return encode_multi_latency([latency], max_latency_dims)


    def decode_latency(codec: np.ndarray,
                       onehot: np.ndarray,
                       max_latency_dims: int
                       ) -> np.ndarray:
        return decode_multi_latency(codec, onehot, max_latency_dims)[0]
