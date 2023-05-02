from typing import *

import dgl
import numpy as np
import torch
from tensorkit import tensor as T

from tracegnn.models.trace_vae.constants import *
from tracegnn.models.trace_vae.types import *
from tracegnn.utils.array_buffer import ArrayBuffer

__all__ = [
    'latency_onehot_to_mask',
    'edge_logits_by_dot_product',
    'dense_to_triu',
    'triu_to_dense',
    'dense_triu_adj',
    'pad_node_feature',
    'get_moments',
    'node_count_mask',
    'collect_operation_id',
    'collect_latency_std',
    'collect_latency_reldiff',
    'collect_p_node_count',
    'collect_p_edge',
]


def latency_onehot_to_mask(onehot: T.Tensor) -> T.Tensor:
    """
    >>> onehot = T.as_tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> T.to_numpy(latency_onehot_to_mask(onehot))
    array([[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]])
    >>> T.to_numpy(latency_onehot_to_mask(T.cast(onehot, dtype=T.float32)))
    array([[1., 0., 0.],
           [1., 1., 0.],
           [1., 1., 1.]], dtype=float32)
    """
    origin_dtype = T.get_dtype(onehot)
    onehot = T.as_tensor(onehot, dtype=T.boolean)
    shape = T.shape(onehot)
    right = shape[-1] - 1
    mask = T.full(shape, False, dtype=T.boolean)
    mask[..., right] = onehot[..., right]
    while right > 0:
        old_right = right
        right -= 1
        mask[..., right] = T.logical_or(mask[..., old_right], onehot[..., right])
    return T.cast(mask, dtype=origin_dtype)


def edge_logits_by_dot_product(h: T.Tensor) -> T.Tensor:
    left = h
    right = T.swap_axes(h, -1, -2)
    return T.matmul(left, right)


def triu_mask(node_count: int) -> T.Tensor:
    return torch.triu(T.full([node_count, node_count], True, T.boolean), 1)


def dense_to_triu(x: T.Tensor, node_count: int) -> T.Tensor:
    mask = triu_mask(node_count)
    shape = T.shape(x)
    return T.reshape(x, shape[:-2] + [-1])[..., mask.reshape(-1)]


def triu_to_dense(x: T.Tensor,
                  node_count: int,
                  pad_value: Union[int, float] = 0) -> T.Tensor:
    mask = triu_mask(node_count).reshape(-1)
    ret = T.full([node_count * node_count], pad_value, dtype=T.get_dtype(x))
    ret[mask] = x
    return T.reshape(ret, [node_count, node_count])


def dense_triu_adj(g: dgl.DGLGraph, node_count: int, reverse: bool = False) -> T.Tensor:
    adj = T.zeros([node_count, node_count], dtype=T.float32)
    u, v = g.edges()
    if reverse:
        v, u = u, v
    adj[u, v] = 1
    # adj = to_dense_adj(
    #     T.stack([u, v], axis=0),
    #     max_num_nodes=node_count
    # )
    return dense_to_triu(adj, node_count)


def pad_node_feature(G: TraceGraphBatch,
                     feature_name: str,
                     max_node_count: int = MAX_NODE_COUNT):
    # inspect graph count
    graph_count = len(G.dgl_graphs)

    # inspect features
    vec = G.dgl_batch.ndata[feature_name]
    value_shape = T.shape(vec)[1:]
    dtype = T.get_dtype(vec)
    device = T.get_device(vec)

    # todo: whether or not it's better to use concat instead of copying into a new tensor?
    with T.no_grad():
        ret = T.zeros(
            [graph_count, max_node_count] + value_shape,
            dtype=dtype,
            device=device,
        )
        for i in range(graph_count):
            vec = G.dgl_graphs[i].ndata[feature_name]
            ret[i, :T.shape(vec)[0]] = vec
    return ret


def get_moments(x,
                axis: Optional[List[int]] = None,
                clip_var: bool = False,
                ) -> Tuple[T.Tensor, T.Tensor]:
    mean = T.reduce_mean(x, axis=axis)
    var = T.reduce_mean(x ** 2, axis=axis) - mean ** 2
    if clip_var:
        var = T.maximum(var, dtype=T.get_dtype(var))
    return mean, var


def node_count_mask(node_count,
                    max_node_count: int,
                    dtype: Optional[str] = None) -> T.Tensor:
    h = T.arange(0, max_node_count, dtype=T.get_dtype(node_count))
    node_count = T.expand_dim(node_count, axis=-1)
    h = h < node_count
    if dtype is not None:
        h = T.cast(h, dtype)
    return h


def collect_operation_id(buf, chain, mask=None):
    if 'node_type' in chain.p:
        node_count = T.to_numpy(chain.p['node_count'].tensor)
        node_type = chain.p['node_type'].tensor
        if len(T.shape(node_type)) == 3:
            node_type = node_type[0, ...]
        node_type = T.to_numpy(node_type)
        if mask is None:
            for i, k in enumerate(node_count):
                buf.extend(node_type[i, :k])
        else:
            for i, (k, m) in enumerate(zip(node_count, mask)):
                if m:
                    buf.extend(node_type[i, :k])


def collect_latency_std(buf, chain, mask=None):
    if 'latency' in chain.p:
        node_count = T.to_numpy(chain.p['node_count'].tensor)
        latency_std = chain.p['latency'].distribution.base_distribution.std
        if len(T.shape(latency_std)) == 4:
            latency_std = latency_std[0, ...]
        latency_std = T.to_numpy(latency_std)

        if mask is None:
            for i, k in enumerate(node_count):
                buf.extend(latency_std[i, :k, 0])
        else:
            for i, (k, m) in enumerate(zip(node_count, mask)):
                if m:
                    buf.extend(latency_std[i, :k, 0])

def collect_p_node_count(buf, chain, mask=None):
    node_count = chain.p['node_count'].distribution.probs[0]
    truth_node_count = chain.p['node_count'].tensor.unsqueeze(1)
    
    node_count_p = torch.gather(node_count, 1, truth_node_count).squeeze(-1)

    if mask is None:
        buf.extend(T.to_numpy(node_count_p))
    else:
        buf.extend(T.to_numpy(node_count_p)[mask])

def collect_p_edge(buf: ArrayBuffer, chain, mask=None):
    # prob = np.exp(T.to_numpy(chain.p.log_prob('adj'))[0])
    node_count = T.to_numpy(chain.p['node_count'].tensor)
    p_edge = chain.p['adj'].distribution.probs[0]
    truth_p_edge = chain.p['adj'].tensor
    
    if mask is None:
        for i in range(p_edge.shape[0]):
            cur_p_edge = T.to_numpy(triu_to_dense(p_edge[i], MAX_NODE_COUNT))[:node_count[i], :node_count[i]]
            cur_truth = T.to_numpy(triu_to_dense(truth_p_edge[i], MAX_NODE_COUNT))[:node_count[i], :node_count[i]]
            buf.extend(np.abs((1.0 - cur_truth) - cur_p_edge).reshape(-1))
    else:
        for i, m in enumerate(mask):
            if m:
                cur_p_edge = T.to_numpy(triu_to_dense(p_edge[i], MAX_NODE_COUNT))[:node_count[i], :node_count[i]]
                cur_truth = T.to_numpy(triu_to_dense(truth_p_edge[i], MAX_NODE_COUNT))[:node_count[i], :node_count[i]]
                buf.extend(np.abs((1.0 - cur_truth) - cur_p_edge).reshape(-1))

def collect_latency_reldiff(buf, chain, mask=None, abs=True):
    def collect_dist_val(attr=None):
        if attr is None:
            v = chain.p['latency'].tensor
        else:
            v = getattr(chain.p['latency'].distribution.base_distribution, attr)
        if len(T.shape(v)) == 4:
            v = v[0, ...]
        return T.to_numpy(v[..., 0])

    if 'latency' in chain.p:
        node_count = T.to_numpy(chain.p['node_count'].tensor)
        latency = collect_dist_val()
        latency_mean = collect_dist_val('mean')
        latency_std = collect_dist_val('std')
        rel_diff = (latency - latency_mean) / np.maximum(latency_std, 1e-7)
        if abs:
            rel_diff = np.abs(rel_diff)

        if mask is None:
            for i, k in enumerate(node_count):
                buf.extend(rel_diff[i, :k])
        else:
            for i, (k, m) in enumerate(zip(node_count, mask)):
                if m:
                    buf.extend(rel_diff[i, :k])
