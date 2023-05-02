from typing import *
from tensorkit import tensor as T

from ..constants import *

__all__ = [
    'decoder_use_depth_and_idx',
]


def decoder_use_depth_and_idx(g,
                              use_depth: bool,
                              use_idx: bool
                              ) -> Optional[T.Tensor]:
    def use_tensor(name, num_classes):
        if isinstance(g, list):
            t = T.stack([g2.ndata[name] for g2 in g], axis=0)
        else:
            t = g.ndata[name]
        t_shape = T.shape(t)
        t = T.reshape(
            t,
            t_shape[:-1] + [t_shape[-1] // MAX_NODE_COUNT, MAX_NODE_COUNT]
        )
        t = T.one_hot(t, num_classes, dtype=T.float32)
        return t

    buf = []
    if use_depth:
        buf.append(use_tensor('node_depth', MAX_DEPTH + 1))
    if use_idx:
        buf.append(use_tensor('node_idx', MAX_NODE_COUNT))

    if buf:
        return T.concat(buf, axis=-1)
