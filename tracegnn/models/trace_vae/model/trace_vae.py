from enum import Enum
from typing import *

import mltk
import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.typing_ import TensorOrData

from ..constants import *
from ..tensor_utils import *
from ..types import *
from .latency_vae import *
from .operation_embedding import *
from .struct_vae import *

__all__ = [
    'TraceVAEArch',
    'TraceVAEConfig',
    'TraceVAE',
]


class TraceVAEArch(str, Enum):
    DEFAULT = 'default'


class TraceVAEConfig(mltk.Config):
    # operation embedding
    operation_embedding_dim: int = 40

    # the architecture selector
    arch: TraceVAEArch = TraceVAEArch.DEFAULT

    # the default architecture
    struct: TraceStructVAEConfig = TraceStructVAEConfig()
    latency: TraceLatencyVAEConfig = TraceLatencyVAEConfig()
    use_latency: bool = True


class TraceVAE(tk.layers.BaseLayer):

    config: TraceVAEConfig
    num_operations: int

    def __init__(self, config: TraceVAEConfig, num_operations: int):
        super().__init__()

        # ===================
        # memorize the config
        # ===================
        self.config = config
        self.num_operations = num_operations

        # ==============
        # the components
        # ==============
        self.operation_embedding = OperationEmbedding(
            num_operations=num_operations,
            embedding_dim=config.operation_embedding_dim,
        )
        if self.config.arch == TraceVAEArch.DEFAULT:
            self.struct_vae = TraceStructVAE(config.struct, self.operation_embedding)
            if self.config.use_latency:
                self.latency_vae = TraceLatencyVAE(
                    config.latency,
                    config.struct.z_dim,
                    self.operation_embedding,
                )
        else:
            raise ValueError(f'Unsupported arch: {self.config.arch!r}')

    def _is_attr_included_in_repr(self, attr: str, value: Any) -> bool:
        if attr == 'config':
            return False
        return super()._is_attr_included_in_repr(attr, value)

    def _call_graph_batch_build(self, G: TraceGraphBatch):
        G.build_dgl(
            add_self_loop=True,
            directed=False,
            # directed=('reverse' if self.config.edge.reverse_directed else False),
        )

    def q(self,
          G: TraceGraphBatch,
          observed: Optional[Mapping[str, TensorOrData]] = None,
          n_z: Optional[int] = None,
          no_latency: bool = False,
          ):
        config = self.config

        self._call_graph_batch_build(G)
        net = tk.BayesianNet(observed=observed)

        self.struct_vae.q(net, G.dgl_batch, n_z=n_z)
        if config.use_latency and not no_latency:
            self.latency_vae.q(net, G.dgl_batch, n_z=n_z)

        return net

    def p(self,
          observed: Optional[Mapping[str, TensorOrData]] = None,
          G: Optional[TraceGraphBatch] = None,  # the observed `G`
          n_z: Optional[int] = None,
          no_latency: bool = False,
          use_biased: bool = False,
          use_latency_biased: bool = False,
          latency_logstd_min: Optional[float] = None,
          latency_log_prob_weight: bool = False,
          std_limit: Optional[T.Tensor] = None,
          ) -> tk.BayesianNet:
        config = self.config

        # populate `observed` from `G` if specified, and construct net
        if G is not None:
            self._call_graph_batch_build(G)
            g = G.dgl_batch
            observed = observed or {}

            # struct
            observed['node_count'] = G.dgl_batch.batch_num_nodes()
            observed['adj'] = T.stack(
                [
                    dense_triu_adj(
                        g,
                        MAX_NODE_COUNT,
                        reverse=False,
                    )
                    for g in G.dgl_graphs
                ],
                axis=0
            )
            # observed['span_count'] = pad_node_feature(G, 'span_count')
            observed['node_type'] = pad_node_feature(G, 'node_type')

            # latency
            latency = pad_node_feature(G, 'latency')[..., :LATENCY_DIM]
            if config.latency.decoder.clip_latency_to_one_dim:
                latency = latency[..., :1]
            observed['latency'] = latency
        else:
            g = None

        # the Bayesian net
        net = tk.BayesianNet(observed=observed)

        # call components
        self.struct_vae.p(net, g, n_z=n_z, use_biased=use_biased)
        if config.use_latency and not no_latency:
            g = net.meta['g']
            self.latency_vae.p(
                net,
                g,
                n_z=n_z,
                use_biased=use_biased and use_latency_biased,
                latency_logstd_min=latency_logstd_min,
                latency_log_prob_weight=latency_log_prob_weight,
                std_limit=std_limit,
            )

        return net
