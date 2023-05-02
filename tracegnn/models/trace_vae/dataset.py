from dataclasses import dataclass
from typing import *

import dgl
import mltk
import numpy as np
import torch
from tensorkit import tensor as T

from tracegnn.data import *
from tracegnn.utils import *
from .constants import *

__all__ = [
    'trace_graph_to_dgl',
    'TraceGraphDataStream',
]


def trace_graph_to_dgl(graph: TraceGraph,
                       num_node_types: int,
                       add_self_loop: bool,
                       latency_range: Optional[TraceGraphLatencyRangeFile] = None,
                       directed: Union[bool, str] = False,  # True, False or 'reverse'
                       ):
    with T.no_grad():
        gv = graph.graph_vectors()

        # build edges
        # todo: use heterogeneous graph to distinguish between "parent -> child" edge and opposite direction
        #       here we just add edges for the both direction (as an initial step)
        if directed == 'reverse':
            u = T.as_tensor(gv.v, dtype=T.int64)
            v = T.as_tensor(gv.u, dtype=T.int64)
        elif directed is True:
            u = T.as_tensor(gv.u, dtype=T.int64)
            v = T.as_tensor(gv.v, dtype=T.int64)
        elif directed is False:
            u = T.as_tensor(
                np.concatenate([gv.u, gv.v], axis=0),
                dtype=T.int64,
            )
            v = T.as_tensor(
                np.concatenate([gv.v, gv.u], axis=0),
                dtype=T.int64
            )
        else:
            raise ValueError(f'Unsupported value for directed: {directed!r}')

        g = dgl.graph((u, v), num_nodes=graph.node_count)
        if add_self_loop:
            g = dgl.add_self_loop(g)

        # node type (use nn.Embedding later to map the node type => node embedding)
        g.ndata['node_type'] = T.as_tensor(gv.node_type, dtype=T.int64)

        # the index of the node under its parent
        g.ndata['node_idx'] = T.as_tensor(gv.node_idx, dtype=T.int64)

        # node depth
        g.ndata['node_depth'] = T.as_tensor(gv.node_depth, dtype=T.int64)

        # span count
        g.ndata['span_count'] = T.as_tensor(np.minimum(gv.span_count, MAX_SPAN_COUNT), dtype=T.int64)

        # latency
        if USE_MULTI_DIM_LATENCY_CODEC:
            for pfx in ('avg_', 'max_', 'min_'):
                codec, onehot = encode_latency(getattr(gv, f'{pfx}latency'), MAX_LATENCY_DIM)
                g.ndata[f'{pfx}latency_codec'] = T.as_tensor(codec, dtype=T.float32)
                g.ndata[f'{pfx}latency_onehot'] = T.as_tensor(onehot, dtype=T.float32)
        else:
            for pfx in ('avg_', 'max_', 'min_'):
                latency_array = getattr(gv, f'{pfx}latency')
                latency = []
                for i in range(graph.node_count):
                    mu, std = latency_range[gv.node_type[i]]
                    latency.append((latency_array[i] - mu) / (std + 1e-5))
                g.ndata[f'{pfx}latency'] = T.as_tensor(np.reshape(latency, (-1, 1)), dtype=T.float32)
            g.ndata['latency'] = T.concat(
                [
                    g.ndata['avg_latency'],
                    g.ndata['min_latency'],
                    g.ndata['max_latency'],
                ],
                axis=-1,
            )

    return g


class TraceGraphDataStream(mltk.data.MapperDataStream):

    def __init__(self,
                 db: TraceGraphDB,
                 id_manager: TraceGraphIDManager,
                 batch_size: int,
                 shuffle: bool = False,
                 skip_incomplete: bool = False,
                 random_state: Optional[np.random.RandomState] = None,
                 data_count: Optional[int] = None,
                 ):
        if (data_count is not None) and (data_count < len(db)) and shuffle:
            indices = np.arange(len(db))
            np.random.shuffle(indices)
            indices = indices[:data_count]
            source_cls = lambda **kwargs: mltk.DataStream.arrays([indices], **kwargs)
        else:
            if data_count is None:
                data_count = len(db)
            source_cls = lambda **kwargs: mltk.DataStream.int_seq(data_count, **kwargs)

        source = source_cls(
            batch_size=batch_size,
            shuffle=shuffle,
            skip_incomplete=skip_incomplete,
            random_state=random_state,
        )

        def mapper(indices):
            return (np.array(
                [
                    db.get(idx)
                    for idx in indices
                ]
            ),)

        super().__init__(
            source=source,
            mapper=mapper,
            array_count=1,
            data_shapes=((),)
        )

