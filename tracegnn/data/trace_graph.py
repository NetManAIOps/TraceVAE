import os
import pickle as pkl
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import *

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils import *

__all__ = [
    'TraceGraphNodeFeatures',
    'TraceGraphNodeReconsScores',
    'TraceGraphNode',
    'TraceGraphVectors',
    'TraceGraph',
    'TraceGraphIDManager',
    'load_trace_csv',
    'df_to_trace_graphs',
]


SERVICE_ID_YAML_FILE = 'service_id.yml'
OPERATION_ID_YAML_FILE = 'operation_id.yml'


@dataclass
class TraceGraphNodeFeatures(object):
    __slots__ = ['span_count', 'max_latency', 'min_latency', 'avg_latency']

    span_count: int  # number of duplicates in the parent
    avg_latency: float  # for span_count == 1, avg == max == min
    max_latency: float
    min_latency: float


@dataclass
class TraceGraphNodeReconsScores(object):
    # probability of the node
    edge_logit: float
    operation_logit: float

    # probability of the latency
    avg_latency_nstd: float  # (avg_latency - avg_latency_mean) / avg_latency_std


@dataclass
class TraceGraphSpan(object):
    __slots__ = [
        'span_id', 'start_time', 'latency',
    ]

    span_id: Optional[int]
    start_time: Optional[datetime]
    latency: float


@dataclass
class TraceGraphNode(object):
    __slots__ = [
        'node_id', 'service_id', 'operation_id',
        'features', 'children', 'spans', 'scores',
        'anomaly',
    ]

    node_id: Optional[int]  # the node id of the graph
    service_id: Optional[int]  # the service id
    operation_id: int  # the operation id
    features: TraceGraphNodeFeatures  # the node features
    children: List['TraceGraphNode']  # children nodes
    spans: Optional[List[TraceGraphSpan]]  # detailed spans information (from the original data)
    scores: Optional[TraceGraphNodeReconsScores]
    anomaly: Optional[int]  # 1: drop anomaly; 2: latency anomaly; 3: service type anomaly

    def __eq__(self, other):
        return other is self

    def __hash__(self):
        return id(self)

    @staticmethod
    def new_sampled(node_id: int,
                    operation_id: int,
                    features: TraceGraphNodeFeatures,
                    scores: Optional[TraceGraphNodeReconsScores] = None
                    ):
        return TraceGraphNode(
            node_id=node_id,
            service_id=None,
            operation_id=operation_id,
            features=features,
            children=[],
            spans=None,
            scores=scores,
            anomaly=None,
        )

    def iter_bfs(self,
                 depth: int = 0,
                 with_parent: bool = False
                 ) -> Generator[
                    Union[
                        Tuple[int, 'TraceGraphNode'],
                        Tuple[int, 'TraceGraphNode', 'TraceGraphNode']
                    ],
                    None,
                    None
                ]:
        """Iterate through the nodes in BFS order."""
        if with_parent:
            depth = depth
            level = [(self, None, 0)]

            while level:
                next_level: List[Tuple[TraceGraphNode, TraceGraphNode, int]] = []
                for nd, parent, idx in level:
                    yield depth, idx, nd, parent
                    for c_idx, child in enumerate(nd.children):
                        next_level.append((child, nd, c_idx))
                depth += 1
                level = next_level

        else:
            depth = depth
            level = [self]

            while level:
                next_level: List[TraceGraphNode] = []
                for nd in level:
                    yield depth, nd
                    next_level.extend(nd.children)
                depth += 1
                level = next_level

    def count_nodes(self) -> int:
        ret = 0
        for _ in self.iter_bfs():
            ret += 1
        return ret


@dataclass
class TraceGraphVectors(object):
    """Cached result of `TraceGraph.graph_vectors()`."""
    __slots__ = [
        'u', 'v',
        'node_type',
        'node_depth', 'node_idx',
        'span_count', 'avg_latency', 'max_latency', 'min_latency',
        'node_features',
    ]

    # note that it is guaranteed that u[i] < v[i], i.e., upper triangle matrix
    u: np.ndarray
    v: np.ndarray

    # node type
    node_type: np.ndarray

    # node depth
    node_depth: np.ndarray

    # node idx
    node_idx: np.ndarray

    # node feature
    span_count: np.ndarray
    avg_latency: np.ndarray
    max_latency: np.ndarray
    min_latency: np.ndarray


@dataclass
class TraceGraph(object):
    __slots__ = [
        'version',
        'trace_id', 'parent_id', 'root', 'node_count', 'max_depth', 'data',
    ]

    version: int  # version control
    trace_id: Optional[Tuple[int, int]]
    parent_id: Optional[int]
    root: TraceGraphNode
    node_count: Optional[int]
    max_depth: Optional[int]
    data: Dict[str, Any]  # any data about the graph

    @staticmethod
    def default_version() -> int:
        return 0x2

    @staticmethod
    def new_sampled(root: TraceGraphNode, node_count: int, max_depth: int):
        return TraceGraph(
            version=TraceGraph.default_version(),
            trace_id=None,
            parent_id=None,
            root=root,
            node_count=node_count,
            max_depth=max_depth,
            data={},
        )

    @property
    def edge_count(self) -> Optional[int]:
        if self.node_count is not None:
            return self.node_count - 1

    def iter_bfs(self,
                 with_parent: bool = False
                 ):
        """Iterate through the nodes in BFS order."""
        yield from self.root.iter_bfs(with_parent=with_parent)

    def merge_spans_and_assign_id(self):
        """
        Merge spans with the same (service, operation) under the same parent,
        and re-assign node IDs.
        """
        node_count = 0
        max_depth = 0

        for depth, parent in self.iter_bfs():
            max_depth = max(max_depth, depth)

            # assign ID to this node
            parent.node_id = node_count
            node_count += 1

            # merge the children of this node
            children = []
            for child in sorted(parent.children, key=lambda o: o.operation_id):
                if children and children[-1].operation_id == child.operation_id:
                    prev_child = children[-1]

                    # merge the features
                    f1, f2 = prev_child.features, child.features
                    f1.span_count += f2.span_count
                    f1.avg_latency += (f2.avg_latency - f1.avg_latency) * (f2.span_count / f1.span_count)
                    f1.max_latency = max(f1.max_latency, f2.max_latency)
                    f1.min_latency = min(f1.min_latency, f2.min_latency)

                    # merge the children
                    if child.children:
                        if prev_child.children:
                            prev_child.children.extend(child.children)
                        else:
                            prev_child.children = child.children

                    # merge the spans
                    if child.spans:
                        if prev_child.spans:
                            prev_child.spans.extend(child.spans)
                        else:
                            prev_child.spans = child.spans
                else:
                    children.append(child)

            # re-assign the merged children
            parent.children = children

        # record node count and depth
        self.node_count = node_count
        self.max_depth = max_depth

    def assign_node_id(self):
        """Assign node IDs to the graph nodes by pre-root order."""
        node_count = 0
        max_depth = 0

        for depth, node in self.iter_bfs():
            max_depth = max(max_depth, depth)

            # assign id to this node
            node.node_id = node_count
            node_count += 1

        # record node count and depth
        self.node_count = node_count
        self.max_depth = max_depth

    def graph_vectors(self):
        # edge index
        u = np.empty([self.edge_count], dtype=np.int64)
        v = np.empty([self.edge_count], dtype=np.int64)

        # node type
        node_type = np.zeros([self.node_count], dtype=np.int64)

        # node depth
        node_depth = np.zeros([self.node_count], dtype=np.int64)

        # node idx
        node_idx = np.zeros([self.node_count], dtype=np.int64)

        # node feature
        span_count = np.zeros([self.node_count], dtype=np.int64)
        avg_latency = np.zeros([self.node_count], dtype=np.float32)
        max_latency = np.zeros([self.node_count], dtype=np.float32)
        min_latency = np.zeros([self.node_count], dtype=np.float32)

        # X = np.zeros([self.node_count, x_dim], dtype=np.float32)

        edge_idx = 0
        for depth, idx, node, parent in self.iter_bfs(with_parent=True):
            j = node.node_id
            feat = node.features

            # node type
            node_type[j] = node.operation_id

            # node depth
            node_depth[j] = depth

            # node idx
            node_idx[j] = idx

            # node feature
            span_count[j] = feat.span_count
            avg_latency[j] = feat.avg_latency
            max_latency[j] = feat.max_latency
            min_latency[j] = feat.min_latency
            # X[parent.node_id, parent.operation_id] = 1   # one-hot encoded node feature

            # edge index
            for child in node.children:
                u[edge_idx] = node.node_id
                v[edge_idx] = child.node_id
                edge_idx += 1

        if len(u) != self.edge_count:
            raise ValueError(f'`len(u)` != `self.edge_count`: {len(u)} != {self.edge_count}')

        return TraceGraphVectors(
            # edge index
            u=u, v=v,
            # node type
            node_type=node_type,
            # node depth
            node_depth=node_depth,
            # node idx
            node_idx=node_idx,
            # node feature
            span_count=span_count,
            avg_latency=avg_latency,
            max_latency=max_latency,
            min_latency=min_latency,
        )

    def networkx_graph(self, id_manager: 'TraceGraphIDManager') -> nx.Graph:
        gv = self.graph_vectors()
        self_nodes = {nd.node_id: nd for _, nd in self.iter_bfs()}
        g = nx.Graph()
        # graph
        for k, v in self.data.items():
            g.graph[k] = v
        # nodes
        g.add_nodes_from(range(self.node_count))
        # edges
        g.add_edges_from([(i, j) for i, j in zip(gv.u, gv.v)])
        # node features
        for i in range(len(gv.node_type)):
            nd = g.nodes[i]
            nd['node_type'] = gv.node_type[i]
            nd['operation'] = id_manager.operation_id.reverse_map(gv.node_type[i])
            for attr in TraceGraphNodeFeatures.__slots__:
                nd[attr] = getattr(gv, attr)[i]
            if self_nodes[i].scores:
                nd['avg_latency_nstd'] = self_nodes[i].scores.avg_latency_nstd
        return g

    def to_bytes(self, protocol: int = pkl.DEFAULT_PROTOCOL) -> bytes:
        return pkl.dumps(self, protocol=protocol)

    @staticmethod
    def from_bytes(content: bytes) -> 'TraceGraph':
        r = pkl.loads(content)

        # for deserializing old versions of TraceGraph
        if not hasattr(r, 'version'):
            r.version = 0x0

        if r.version < 0x1:  # upgrade 0x0 => 0x2
            for _, nd in r.root.iter_bfs():
                nd.scores = None
                nd.anomaly = None
            r.version = 0x2

        if r.version < 0x2:  # upgrade 0x1 => 0x2
            for _, nd in r.root.iter_bfs():
                nd.anomaly = None
            r.version = 0x2

        return r

    def deepcopy(self) -> 'TraceGraph':
        return TraceGraph.from_bytes(self.to_bytes())


@dataclass
class TempGraphNode(object):
    __slots__ = ['trace_id', 'parent_id', 'node']

    trace_id: Tuple[int, int]
    parent_id: int
    node: 'TraceGraphNode'


class TraceGraphIDManager(object):
    __slots__ = ['root_dir', 'service_id', 'operation_id']

    root_dir: str
    service_id: IDAssign
    operation_id: IDAssign

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.service_id = IDAssign(os.path.join(self.root_dir, SERVICE_ID_YAML_FILE))
        self.operation_id = IDAssign(os.path.join(self.root_dir, OPERATION_ID_YAML_FILE))

    def __enter__(self):
        self.service_id.__enter__()
        self.operation_id.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.service_id.__exit__(exc_type, exc_val, exc_tb)
        self.operation_id.__exit__(exc_type, exc_val, exc_tb)

    @property
    def num_operations(self) -> int:
        return len(self.operation_id)

    def dump_to(self, output_dir: str):
        self.service_id.dump_to(os.path.join(output_dir, SERVICE_ID_YAML_FILE))
        self.operation_id.dump_to(os.path.join(output_dir, OPERATION_ID_YAML_FILE))


def load_trace_csv(input_path: str, is_test: bool=False) -> pd.DataFrame:
    if is_test:
        dtype = {
            'traceIdHigh': int,
            'traceIdLow': int,
            'spanId': int,
            'parentSpanId': int,
            'serviceName': str,
            'operationName': str,
            'startTime': str,
            'duration': float,
            'nanosecond': int,
            'DBhash': int,
            'nodeLatencyLabel': int,
            'graphLatencyLabel': int,
            'graphStructureLabel': int
        }
    else:
        dtype = {
            'traceIdHigh': int,
            'traceIdLow': int,
            'spanId': int,
            'parentSpanId': int,
            'serviceName': str,
            'operationName': str,
            'startTime': str,
            'duration': float,
            'nanosecond': int,
            'DBhash': int,
        }

    return pd.read_csv(
        input_path,
        engine='c',
        usecols=list(dtype),
        dtype=dtype
    )


def df_to_trace_graphs(df: pd.DataFrame,
                       id_manager: TraceGraphIDManager,
                       test_label: int = None,
                       min_node_count: int = 2,
                       max_node_count: int = 32,
                       summary_file: Optional[str] = None,
                       merge_spans: bool = False,
                       ) -> List[TraceGraph]:
    summary = []
    trace_spans = {}
    df = df[df['DBhash'] == 0]

    # read the spans
    with id_manager:
        for row in tqdm(df.itertuples(), desc='Read spans', total=len(df)):
            graph_label = 0

            if test_label is not None:
                if row.graphStructureLabel != 0:
                    graph_label = 1
                elif row.graphLatencyLabel != 0:
                    graph_label = 2
                if graph_label != test_label:
                    continue

            if row.serviceName not in id_manager.service_id._mapping:
                print(row.serviceName, ": Service not in file!")
                continue
            if f'{row.serviceName}/{row.operationName}' not in id_manager.operation_id._mapping:
                print(f'{row.serviceName}/{row.operationName}', ": Operation not in file!")
                continue

            trace_id = (row.traceIdHigh, row.traceIdLow)
            span_dict = trace_spans.get(trace_id, None)
            if span_dict is None:
                trace_spans[trace_id] = span_dict = {}

            span_latency = row.duration
            span_dict[row.spanId] = TempGraphNode(
                trace_id=trace_id,
                parent_id=row.parentSpanId,
                node=TraceGraphNode(
                    node_id=None,
                    service_id=id_manager.service_id.get_or_assign(row.serviceName),
                    operation_id=id_manager.operation_id.get_or_assign(f'{row.serviceName}/{row.operationName}'),
                    features=TraceGraphNodeFeatures(
                        span_count=1,
                        avg_latency=span_latency,
                        max_latency=span_latency,
                        min_latency=span_latency,
                    ),
                    children=[],
                    spans=[
                        TraceGraphSpan(
                            span_id=row.spanId,
                            start_time=(
                                datetime.strptime(row.startTime, '%Y-%m-%d %H:%M:%S') +
                                timedelta(microseconds=row.nanosecond / 1_000)
                            ),
                            latency=span_latency,
                        ),
                    ],
                    scores=None,
                    anomaly=None,
                )
            )

    summary.append(f'Span count: {len(trace_spans)}')

    # construct the traces
    trace_graphs = []

    if test_label is None or test_label == 0:
        graph_data = {}
    elif test_label == 1:
        graph_data = {
            'is_anomaly': True,
            'anomaly_type': 'drop'
        }
    else:
        graph_data = {
            'is_anomaly': True,
            'anomaly_type': 'latency'
        }

    for _, trace in tqdm(trace_spans.items(), total=len(trace_spans), desc='Build graphs'):
        nodes = sorted(
            trace.values(),
            key=(lambda nd: (nd.node.service_id, nd.node.operation_id, nd.node.spans[0].start_time))
        )
        for nd in nodes:
            parent_id = nd.parent_id
            if (parent_id == 0) or (parent_id not in trace):
                # if only a certain service is taken from the database, then just the sub-trees
                # of a trace are obtained, which leads to orphan nodes (parent_id != 0 and not in trace
                trace_graphs.append(TraceGraph(
                    version=TraceGraph.default_version(),
                    trace_id=nd.trace_id,
                    parent_id=nd.parent_id,
                    root=nd.node,
                    node_count=None,
                    max_depth=None,
                    data=graph_data,
                ))
            else:
                trace[parent_id].node.children.append(nd.node)

    # merge spans and assign id
    if merge_spans:
        for trace in tqdm(trace_graphs, desc='Merge spans and assign node id'):
            trace.merge_spans_and_assign_id()
    else:
        for trace in tqdm(trace_graphs, desc='Assign node id'):
            trace.assign_node_id()

    # gather the final results
    ret = []
    too_small = 0
    too_large = 0

    for trace in trace_graphs:
        if trace.node_count < min_node_count:
            too_small += 1
        elif trace.node_count > max_node_count:
            too_large += 1
        else:
            ret.append(trace)

    summary.append(f'Imported graph: {len(trace_graphs)}; dropped graph: too small = {too_small}, too large = {too_large}')
    if summary_file:
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary) + '\n')
    else:
        print('\n'.join(summary), file=sys.stderr)

    return ret
