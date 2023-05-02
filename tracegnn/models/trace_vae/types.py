from dataclasses import dataclass
from typing import *

import dgl
from tensorkit import tensor as T

from tracegnn.data import *
from tracegnn.utils import *

__all__ = ['TraceGraphBatch']


@dataclass(init=False)
class TraceGraphBatch(object):
    __slots__ = [
        'id_manager', 'latency_range',
        'trace_graphs', 'dgl_graphs', 'dgl_batch'
    ]

    id_manager: Optional[TraceGraphIDManager]
    trace_graphs: Optional[List[TraceGraph]]  # the original trace graphs
    dgl_graphs: Optional[List[dgl.DGLGraph]]  # graph components
    dgl_batch: Optional[dgl.DGLGraph]  # the batched DGL graph

    def __init__(self,
                 *,
                 id_manager: Optional[TraceGraphIDManager] = None,
                 latency_range: Optional[TraceGraphLatencyRangeFile] = None,
                 trace_graphs: Optional[List[TraceGraph]] = None,
                 dgl_graphs: Optional[List[dgl.DGLGraph]] = None,
                 dgl_batch: Optional[dgl.DGLGraph] = None,
                 ):
        if ((trace_graphs is None) or (id_manager is None)) and \
                ((dgl_graphs is None) or (dgl_batch is None)):
            raise ValueError('Insufficient arguments.')
        self.id_manager = id_manager
        self.latency_range = latency_range
        self.trace_graphs = trace_graphs
        self.dgl_graphs = dgl_graphs
        self.dgl_batch = dgl_batch

    def build_dgl(self,
                  add_self_loop: bool = True,
                  directed: Union[bool, str] = False,
                  ):
        from .dataset import trace_graph_to_dgl
        if self.dgl_graphs is None:
            with T.no_grad():
                with T.use_device('cpu'):
                    self.dgl_graphs = [
                        trace_graph_to_dgl(
                            g,
                            num_node_types=self.id_manager.num_operations,
                            add_self_loop=add_self_loop,
                            latency_range=self.latency_range,
                            directed=directed,
                        )
                        for g in self.trace_graphs
                    ]
        if self.dgl_batch is None:
            with T.no_grad():
                self.dgl_batch = dgl.batch(self.dgl_graphs).to(T.current_device())

    # @property
    # def dgl_graphs(self) -> List[dgl.DGLGraph]:
    #     if self._dgl_graphs is None:
    #         self.build_dgl()
    #     return self._dgl_graphs
    #
    # @property
    # def dgl_batch(self) -> dgl.DGLGraph:
    #     if self._dgl_batch is None:
    #         self.build_dgl()
    #     return self._dgl_batch
