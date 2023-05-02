import math
from dataclasses import dataclass
from typing import *

import networkx as nx
import numpy as np
import tensorkit as tk
from tensorkit import tensor as T

from tracegnn.data import *
from tracegnn.utils import *
from .constants import *
from .tensor_utils import *
import dgl
import torch

__all__ = [
    'flat_to_nx_graphs',
    'p_net_to_trace_graphs',
    'GraphNodeMatch', 'GraphNodeDiff',
    'diff_graph',
]


# util to reshape an array
def reshape_to(x, ndims):
    shape = T.shape(x)
    return T.reshape(x, [-1] + shape[len(shape) - ndims + 1:])


def to_scalar(x):
    return T.to_numpy(x).tolist()


def flat_to_nx_graphs(p: tk.BayesianNet,
                      id_manager: TraceGraphIDManager,
                      latency_range: TraceGraphLatencyRangeFile,
                      min_edge_weight: float = 0.2,
                      ) -> List[nx.Graph]:
    """Convert `p` net sampled from a flat TraceVAE to nx.Graph."""
    # extract features
    adjs = reshape_to(p['adj'].distribution.probs, 2)
    node_counts = T.to_numpy(reshape_to(p['node_count'].tensor, 1))
    node_types = T.to_numpy(reshape_to(p['node_type'].tensor, 2))
    # span_counts = reshape_to(p['span_count'].tensor, 2)

    if 'latency' in p:
        latency_src = T.to_numpy(reshape_to(p['latency'].distribution.base_distribution.mean, 3))
        latencies = np.zeros(latency_src.shape, dtype=np.float32)
        for i in range(node_types.shape[0]):
            for j in range(node_types.shape[1]):
                try:
                    node_type = int(node_types[i, j])
                    mu, std = latency_range[node_type]
                    latencies[i, j] = latency_src[i, j] * std + mu
                except KeyError:
                    latencies[i, j] = -1.  # todo: is this okay?
    else:
        latencies = None

    # build the graph
    ret = []
    for i, node_count in enumerate(node_counts):
        g = nx.Graph()

        # add nodes
        for j in range(node_count):
            g.add_node(j)

        # add edges
        adj = triu_to_dense(adjs[i: i+1], MAX_NODE_COUNT)
        for u in range(node_count):
            for v in range(u + 1, node_count):
                w = float(to_scalar(adj[u, v]))
                if w >= min_edge_weight:
                    g.add_edge(u, v, weight=w)

        # add node attributes
        for j in range(node_count):
            node_type = int(node_types[i, j])
            g.nodes[j]['node_type'] = node_type
            g.nodes[j]['operation'] = id_manager.operation_id.reverse_map(node_type)
            if latencies is not None:
                for k, pfx in enumerate(('avg_', 'max_', 'min_')):
                    if k < LATENCY_DIM:
                        g.nodes[j][f'{pfx}latency'] = latencies[i, j, k]

        #     g.nodes[j]['span_count'] = to_scalar(span_counts[i, j])
        #     for pfx in ('avg_', 'max_', 'min_'):
        #         g.nodes[j][f'{pfx}latency'] = latencies[f'{pfx}latency'][i, j]

        ret.append(g)

    # return the graphs
    return ret


def p_net_to_trace_graphs(p: tk.BayesianNet,
                          id_manager: TraceGraphIDManager,
                          latency_range: TraceGraphLatencyRangeFile,
                          discard_node_with_type_0: bool = True,
                          discard_node_with_unknown_latency_range: bool = True,
                          discard_graph_with_error_node_count: bool = False,
                          keep_front_shape: bool = False,
                          ) -> Union[List[Optional[TraceGraph]], np.ndarray]:
    """Convert `p` net sampled from a flat TraceVAE to TraceGraph."""
    if USE_MULTI_DIM_LATENCY_CODEC:
        raise RuntimeError(f'`USE_MULTI_DIM_LATENCY_CODEC` is not supported.')

    # find the base distribution (Normal, Categorical, OneHotCategorical)
    def find_base(t: tk.StochasticTensor):
        d = t.distribution
        while not isinstance(d, (tk.Normal,
                                 tk.Bernoulli,
                                 tk.Categorical,
                                 tk.OneHotCategorical)):
            d = d.base_distribution
        return d

    # extract features
    def get_adj(t, pad_value=0):
        t = reshape_to(t, 2)
        return np.stack(
            [
                T.to_numpy(triu_to_dense(
                    t[i: i + 1],
                    MAX_NODE_COUNT,
                    pad_value=pad_value
                ))
                for i in range(len(t))
            ],
            axis=0
        )

    def bernoulli_log_prob(l):
        # log(1 / (1 + exp(-l)) = log(exp(l) / (1 + exp(l)))
        return T.where(
            l >= 0,
            -T.log1p(T.exp(-l)),
            l - T.log1p(T.exp(l)),
        )

    def softmax_log_prob(l):
        # log(exp(l) / sum(exp(l))
        return l - T.log_sum_exp(l, axis=[-1], keepdims=True)

    front_shape = T.shape(p['adj'].tensor)[:-1]

    adjs = get_adj(p['adj'].tensor)
    adj_probs = get_adj(find_base(p['adj']).probs)
    adj_logits = get_adj(bernoulli_log_prob(find_base(p['adj']).logits), pad_value=-100000)

    node_counts = T.to_numpy(reshape_to(p['node_count'].tensor, 1))
    node_types = T.to_numpy(reshape_to(p['node_type'].tensor, 2))
    node_count_logits = T.to_numpy(reshape_to(softmax_log_prob(find_base(p['node_count']).logits), 2))
    node_type_logits = T.to_numpy(reshape_to(softmax_log_prob(find_base(p['node_type']).logits), 3))

    if 'latency' in p:
        latencies = T.to_numpy(reshape_to(p['latency'].tensor, 3))
        avg_latencies = latencies[..., 0]
        latency_means = T.to_numpy(reshape_to(find_base(p['latency']).mean, 3))
        latency_stds = T.to_numpy(reshape_to(find_base(p['latency']).std, 3))

    # build the graph
    ret = []
    for i, node_count in enumerate(node_counts):
        # extract the arrays
        adj = adjs[i][:node_count][:, :node_count]
        adj_prob = adj_probs[i][:node_count][:, :node_count]
        adj_logit = adj_logits[i]  # [:node_count][:, :node_count]
        node_type = node_types[i]  # [:node_count]
        node_mask = np.full([node_count], True, dtype=np.bool)
        node_count_logit = node_count_logits[i]
        node_type_logit = node_type_logits[i]

        if 'latency' in p:
            avg_latency = avg_latencies[i]
            latency_mean = latency_means[i]
            latency_std = latency_stds[i]

        # if `discard_node_with_type_0`, set all adjs that from / to `node_type == 0` as 0
        node_count_new = node_count
        for j in range(node_count):
            n_type = int(node_type[j])
            if (discard_node_with_type_0 and n_type == 0) or \
                    (discard_node_with_unknown_latency_range and n_type not in latency_range):
                node_mask[j] = False
                node_count_new -= 1
                adj[:, j] = 0
                adj[j, :] = 0
                adj_prob[:, j] = 0
                adj_prob[j, :] = 0

        # for each column in `adj`, if there are more than 2 candidate in-edges,
        # or no in-edge, then choose an edge sampled w.r.t. to adj_prob
        for j in range(node_count):
            if node_mask[j] and np.sum(adj[:, j]) != 1:
                prob_vec = adj_prob[:, j]
                prob_sum = np.sum(prob_vec)
                if prob_sum > 1e-7:
                    pvals = prob_vec / np.sum(prob_vec)
                    pvals_mask = pvals > 1e-7
                    indices = np.arange(len(pvals))[pvals_mask]
                    k = indices[np.argmax(np.random.multinomial(1, pvals[pvals_mask]))]
                    adj[:, j] = 0
                    adj[k, j] = 1

        # select the edges
        edges = list(zip(*np.where(adj)))
        if len(edges) < node_count_new - 1:
            # pick out the root sub-graph
            union_set = {j: -1 for j in range(node_count) if node_mask[j]}

            def find_root(s):
                t = union_set[s]
                if t == -1:
                    return s
                r = find_root(t)
                if r != t:
                    union_set[s] = r
                return r

            def link_edge(s, t):
                union_set[t] = s

            edges_new = []
            for s, t in edges:
                link_edge(s, t)
            for s, t in edges:
                if s == 0 or find_root(s) == 0:
                    edges_new.append((s, t))

            edges = edges_new
            node_count_new = len(edges_new) + 1

        if discard_graph_with_error_node_count and (node_count_new != node_count):
            ret.append(None)
            continue

        # build the trace graph
        def get_node(s):
            if s not in nodes:
                n_type = node_type[s]
                if 'latency' in p:
                    latency = avg_latency[s]
                    if n_type in latency_range:
                        mu, std = latency_range[n_type]
                        latency = latency * std + mu
                    features = TraceGraphNodeFeatures(
                        span_count=1,
                        avg_latency=latency,
                        max_latency=latency,
                        min_latency=latency,
                    )
                    avg_latency_nstd = float(
                        abs(avg_latency[s] - latency_mean[s, 0]) /
                        latency_std[s, 0]
                    )
                else:
                    features = TraceGraphNodeFeatures(
                        span_count=1,
                        avg_latency=math.nan,
                        max_latency=math.nan,
                        min_latency=math.nan,
                    )
                    avg_latency_nstd = 0

                nodes[s] = TraceGraphNode.new_sampled(
                    node_id=s,
                    operation_id=node_type[s],
                    features=features,
                    scores=TraceGraphNodeReconsScores(
                        edge_logit=0,
                        operation_logit=node_type_logit[s, n_type],
                        avg_latency_nstd=avg_latency_nstd,
                    )
                )
            return nodes[s]

        nodes = {}
        edges.sort()
        for u, v in edges:
            if node_mask[u] and node_mask[v]:
                v_node = get_node(v)
                get_node(u).children.append(v_node)
                v_node.scores.edge_logit = adj_logit[u, v]

        if 0 in nodes:
            g = TraceGraph.new_sampled(nodes[0], len(nodes), -1)
            g.merge_spans_and_assign_id()
            ret.append(g)
        else:
            ret.append(None)

    # return the graphs
    if keep_front_shape:
        ret = np.array(ret).reshape(front_shape)

    return ret


@dataclass(init=False)
class GraphNodeMatch(object):
    __slots__ = [
        'g1_to_g2',
        'g2_to_g1',
    ]

    g1_to_g2: Dict[TraceGraphNode, TraceGraphNode]
    g2_to_g1: Dict[TraceGraphNode, TraceGraphNode]

    def __init__(self):
        self.g1_to_g2 = {}
        self.g2_to_g1 = {}

    def add_match(self, node1, node2):
        self.g1_to_g2[node1] = node2
        self.g2_to_g1[node2] = node1


@dataclass(init=False)
class GraphNodeDiff(object):
    __slots__ = [
        'parent', 'depth', 'node', 'offset', 'node_count',
    ]

    parent: Optional[TraceGraphNode]
    depth: int
    node: TraceGraphNode
    offset: int  # -1: present in g but absent in g2; 1: present in g2 but absent in g1
    node_count: int  # count of nodes in this branch

    def __init__(self, parent, depth, node, offset):
        self.parent = parent
        self.depth = depth
        self.node = node
        self.offset = offset
        self.node_count = node.count_nodes()

    def __repr__(self):
        return f'GraphNodeDiff(depth={self.depth}, offset={self.offset})'


def diff_graph(g1: TraceGraph,
               g2: TraceGraph
               ) -> Tuple[GraphNodeMatch, List[GraphNodeDiff]]:
    m = GraphNodeMatch()
    ret = []

    def match_node(depth: int,
                   parent1: Optional[TraceGraphNode],
                   parent2: Optional[TraceGraphNode],
                   node1: Optional[TraceGraphNode],
                   node2: Optional[TraceGraphNode]):
        if node1 is None:
            if node2 is None:
                pass
            else:
                ret.append(GraphNodeDiff(parent=parent2, depth=depth, node=node2, offset=1))
        else:
            if node2 is None:
                ret.append(GraphNodeDiff(parent=parent1, depth=depth, node=node1, offset=-1))
            elif node1.operation_id != node2.operation_id:
                ret.append(GraphNodeDiff(parent=parent1, depth=depth, node=node1, offset=-1))
                ret.append(GraphNodeDiff(parent=parent2, depth=depth, node=node2, offset=1))
            else:
                m.add_match(node1, node2)
                c_depth = depth + 1

                i, j = 0, 0
                while i < len(node1.children) and j < len(node2.children):
                    c1 = node1.children[i]
                    c2 = node2.children[j]
                    if c1.operation_id < c2.operation_id:
                        match_node(c_depth, node1, None, c1, None)
                        i += 1
                    elif c2.operation_id < c1.operation_id:
                        match_node(c_depth, None, node2, None, c2)
                        j += 1
                    else:
                        match_node(c_depth, node1, node2, c1, c2)
                        i += 1
                        j += 1

                while i < len(node1.children):
                    c1 = node1.children[i]
                    match_node(c_depth, node1, None, c1, None)
                    i += 1

                while j < len(node2.children):
                    c2 = node2.children[j]
                    match_node(c_depth, None, node2, None, c2)
                    j += 1

    match_node(0, None, None, g1.root, g2.root)
    return m, ret


def dgl_graph_key(graph: dgl.DGLGraph) -> str:
    return edges_to_key(graph.ndata['operation_id'], *graph.edges())

@torch.jit.script
def edges_to_key(operation_id: torch.Tensor, u_list: torch.Tensor, v_list: torch.Tensor) -> str:
    mask = u_list != v_list
    u_id: List[int] = operation_id[u_list][mask].tolist()
    v_id: List[int] = operation_id[v_list][mask].tolist()

    graph_key = f'0,{operation_id[0].item()};' + ';'.join(sorted([f'{u},{v}' for (u, v) in zip(u_id, v_id)]))

    return graph_key

def trace_graph_key(graph: TraceGraph) -> str:
    def dfs(nd: TraceGraphNode, pa_id: int, cnt: int=1):
        cur_cnt = cnt * len(nd.spans)
        spans = [f'{pa_id},{nd.operation_id}'] * cur_cnt

        for child in nd.children:
            spans += dfs(child, nd.operation_id, cur_cnt)

        return spans
        
    spans = dfs(graph.root, 0, 1)

    return ';'.join(sorted(spans))
