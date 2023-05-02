from enum import Enum
from typing import *

import mltk
import tensorkit as tk
import torch
from dgl import nn as gnn
from tensorkit import tensor as T

__all__ = [
    'PoolingType',
    'PoolingConfig',
    'make_graph_pooling',
    'graph_node_offsets',
    'RootPooling',
]


class PoolingType(str, Enum):
    ROOT = 'root'
    AVG = 'avg'
    ATTENTION = 'attention'  # graph attention pooling


class PoolingConfig(mltk.Config):
    # whether to use batch norm?
    use_batch_norm: bool = True

    # config for ATTENTION
    class attention(mltk.Config):
        hidden_layers: List[int] = []


def make_graph_pooling(feature_size: int,
                       pool_type: Union[str, PoolingType],
                       pool_config: PoolingConfig):
    layer_args = tk.layers.LayerArgs()
    layer_args.set_args(['dense'], activation=tk.layers.LeakyReLU)
    if pool_config.use_batch_norm:
        layer_args.set_args(['dense'], normalizer=tk.layers.BatchNorm)

    if pool_type == PoolingType.ROOT:
        return RootPooling()  # is this okay?
    elif pool_type == PoolingType.AVG:
        return gnn.AvgPooling()
    elif pool_type == PoolingType.ATTENTION:
        gap_nn_builder = tk.layers.SequentialBuilder(
            feature_size,
            layer_args=layer_args,
        )
        for size in pool_config.attention.hidden_layers:
            gap_nn_builder.dense(size)
        return gnn.GlobalAttentionPooling(gap_nn_builder.linear(1).build())
    else:
        raise ValueError(f'Unsupported `config.encoder.pool_type`: {pool_type!r}')


def graph_node_offsets(seglen):
    ret = torch.cumsum(
        T.concat(
            [
                T.zeros([1], dtype=T.get_dtype(seglen), device=T.get_device(seglen)),
                seglen
            ],
            axis=0
        ),
        dim=0
    )
    return ret[:-1]


class RootPooling(tk.layers.BaseLayer):

    def forward(self, graph, feat):
        return feat[graph_node_offsets(graph.batch_num_nodes())]
