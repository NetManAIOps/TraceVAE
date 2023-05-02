from typing import *

import dgl
import mltk
import tensorkit as tk
from tensorkit import tensor as T

from ..constants import *
from ..distributions import *
from ..tensor_utils import node_count_mask
from .gnn_layers import *
from .model_utils import *
from .operation_embedding import *
from .pooling import *
from .realnvp_flow import *

__all__ = [
    'TraceLatencyVAEConfig',
    'TraceLatencyVAE',
]


class TraceLatencyVAEConfig(mltk.Config):
    # whether to use the operation embedding? (but grad will be blocked)
    use_operation_embedding: bool = True

    # the dimension of z2 (to encode latency)
    z2_dim: int = 10

    # the config of posterior / prior flow
    realnvp: RealNVPFlowConfig = RealNVPFlowConfig()

    # whether to use BatchNorm?
    use_batch_norm: bool = True

    class encoder(mltk.Config):
        # ================
        # h(G) for q(z2|G)
        # ================
        # the gnn layer config
        gnn: GNNLayerConfig = GNNLayerConfig()

        # the gnn layer sizes for q(z2|...)
        gnn_layers: List[int] = [500, 500, 500, 500]

        # whether to stop gradient to operation_embedding along this path?
        operation_embedding_stop_grad: bool = True

        # =============
        # graph pooling
        # =============
        pool_type: PoolingType = PoolingType.AVG
        pool_config: PoolingConfig = PoolingConfig()

        # =======
        # q(z2|G)
        # =======
        z2_logstd_min: Optional[float] = -7
        z2_logstd_max: Optional[float] = 2

        # whether to use realnvp posterior flow?
        use_posterior_flow: bool = False

    class decoder(mltk.Config):
        # ====================
        # decoder architecture
        # ====================
        use_prior_flow: bool = False

        # p(z2|z) n_mixtures
        z2_prior_mixtures: int = 1

        # whether z2 should condition on z?
        condition_on_z: bool = True

        # z2 given z hidden layers
        z2_given_z_stop_grad: bool = True
        z2_given_z_hidden_layers: List[int] = [250, 250]
        z2_logstd_min: Optional[float] = -5
        z2_logstd_max: Optional[float] = 2

        # =======
        # latency
        # =======
        # gnn layer config
        gnn: GNNLayerConfig = GNNLayerConfig()

        # the node types from node embedding e
        gnn_layers: List[int] = [500, 500, 500, 500]

        # hidden layers for graph embedding from z
        graph_embedding_layers: List[int] = [500, 500]

        # size of the latent embedding e
        latent_embedding_size: int = 40

        # whether to stop gradient to operation_embedding along this path?
        operation_embedding_stop_grad: bool = True

        # ==============
        # p(latency|...)
        # ==============
        # the minimum value for latency logstd
        latency_logstd_min: Optional[float] = -7

        # whether to use mask on p(latency|...)?
        use_latency_mask: bool = True

        # whether to clip the latency to one dim even if three dim is provided?
        clip_latency_to_one_dim: bool = False

        # whether to use biased in p(latency|...)?
        use_biased_latency: bool = False

        # whether to use `AnomalyDetectionNormal`?
        use_anomaly_detection_normal: bool = False

        # the `std_threshold` for AnomalyDetectionNormal or BiasedNormal in testing
        biased_normal_std_threshold: float = 4.0

        # the `std_threshold` for SafeNormal in training
        safe_normal_std_threshold: float = 6.0


class TraceLatencyVAE(tk.layers.BaseLayer):

    config: TraceLatencyVAEConfig
    num_node_types: int

    def __init__(self,
                 config: TraceLatencyVAEConfig,
                 z_dim: int,  # z dimension of the struct_vae
                 operation_embedding: OperationEmbedding,
                 ):
        super().__init__()

        # ===================
        # memorize the config
        # ===================
        self.config = config
        self.z_dim = z_dim

        # =============================
        # node embedding for operations
        # =============================
        self.operation_embedding = operation_embedding
        self.num_node_types = operation_embedding.num_operations

        # ========================
        # standard layer arguments
        # ========================
        layer_args = tk.layers.LayerArgs()
        layer_args.set_args(['dense'], activation=tk.layers.LeakyReLU)
        if config.use_batch_norm:
            layer_args.set_args(['dense'], normalizer=tk.layers.BatchNorm)

        # ===========================
        # q(z2|adj,node_type,latency)
        # ===========================
        if config.use_operation_embedding:
            input_size = self.operation_embedding.embedding_dim
        else:
            input_size = self.num_node_types
        output_size, gnn_layers = make_gnn_layers(
            config.encoder.gnn,
            (
                input_size +
                LATENCY_DIM  # avg, min, max
            ),
            config.encoder.gnn_layers,
        )
        self.qz2_gnn_layers = GNNSequential(
            gnn_layers + [
                make_graph_pooling(
                    output_size,
                    config.encoder.pool_type,
                    config.encoder.pool_config
                ),
            ]
        )
        self.qz2_mean = tk.layers.Linear(output_size, config.z2_dim)
        self.qz2_logstd = tk.layers.Linear(output_size, config.z2_dim)

        if config.encoder.use_posterior_flow:
            self.qz_flow = make_realnvp_flow(config.z2_dim, config.realnvp)

        # ================
        # p(z2) or p(z2|z)
        # ================
        if config.decoder.condition_on_z:
            if config.decoder.use_prior_flow and config.decoder.z2_prior_mixtures > 1:
                raise ValueError(f'`use_prior_flow == True` and `z2_prior_mixtures > 1` cannot be both True.')

            n_mixtures = config.decoder.z2_prior_mixtures
            z2_given_z_builder = tk.layers.SequentialBuilder(
                self.z_dim,
                layer_args=layer_args
            )
            for size in config.decoder.z2_given_z_hidden_layers:
                z2_given_z_builder.dense(size)
            self.z2_given_z_hidden_layers = z2_given_z_builder.build(flatten_to_ndims=True)
            self.pz2_mean = z2_given_z_builder.as_input().linear(config.z2_dim * n_mixtures).build()
            self.pz2_logstd = z2_given_z_builder.as_input().linear(config.z2_dim * n_mixtures).build()

        if config.decoder.use_prior_flow:
            self.pz2_flow = make_realnvp_flow(config.z2_dim, config.realnvp).invert()

        # node features from gnn
        input_size = config.z2_dim

        if config.use_operation_embedding:
            input_size += self.operation_embedding.embedding_dim
        else:
            input_size += self.num_operations

        output_size, gnn_layers = make_gnn_layers(
            config.decoder.gnn,
            input_size,
            config.decoder.gnn_layers,
        )
        self.pG_node_features = GNNSequential(
            gnn_layers +
            [
                GraphConv(  # p(latency|e)
                    output_size,
                    2 * LATENCY_DIM  # (mean, logstd) * (avg, min, max)
                ),
            ]
        )

    def _is_attr_included_in_repr(self, attr: str, value: Any) -> bool:
        if attr == 'config':
            return False
        return super()._is_attr_included_in_repr(attr, value)

    def q(self,
          net: tk.BayesianNet,
          g: dgl.DGLGraph,
          n_z: Optional[int] = None):
        config = self.config

        # compose feature vector
        if config.use_operation_embedding:
            h2 = self.operation_embedding(g.ndata['node_type'])
            if config.encoder.operation_embedding_stop_grad:
                h2 = T.stop_grad(h2)
        else:
            h2 = T.one_hot(
                g.ndata['node_type'],
                self.num_node_types,
                dtype=T.float32,
            )
        h = T.concat([h2, g.ndata['latency'][..., :LATENCY_DIM]], axis=-1)

        # feed into gnn and get node embeddings
        h = self.qz2_gnn_layers(g, h)

        # mean and logstd for q(z2|G)
        z2_mean = self.qz2_mean(h)
        z2_logstd = T.maybe_clip(
            self.qz2_logstd(h),
            min_val=config.encoder.z2_logstd_min,
            max_val=config.encoder.z2_logstd_max,
        )

        # add 'z2' random variable
        qz2 = tk.Normal(mean=z2_mean, logstd=z2_logstd, event_ndims=1)
        if config.encoder.use_posterior_flow:
            qz2 = tk.FlowDistribution(qz2, self.qz_flow)
        z2 = net.add('z2', qz2, n_samples=n_z)

    def p(self,
          net: tk.BayesianNet,
          g: dgl.DGLGraph,
          n_z: Optional[int] = None,
          use_biased: bool = False,
          latency_logstd_min: Optional[float] = None,
          latency_log_prob_weight: bool = False,
          std_limit: Optional[T.Tensor] = None,
          ):
        config = self.config

        # sample z2 ~ p(z2) or p(z2|z)
        if config.decoder.condition_on_z:
            h = net['z'].tensor
            if config.decoder.z2_given_z_stop_grad:
                h = T.stop_grad(h)
            h = self.z2_given_z_hidden_layers(h)
            z2_mean = self.pz2_mean(h)
            z2_logstd = T.maybe_clip(
                self.pz2_logstd(h),
                min_val=config.decoder.z2_logstd_min,
                max_val=config.decoder.z2_logstd_max,
            )

            n_mixtures = config.decoder.z2_prior_mixtures
            if n_mixtures > 1:
                z2_mean_list = T.split(z2_mean, [config.z2_dim] * n_mixtures, axis=-1)
                z2_logstd_list = T.split(z2_logstd, [config.z2_dim] * n_mixtures, axis=-1)
                pz2 = tk.Mixture(
                    categorical=tk.Categorical(
                        logits=T.zeros(T.shape(z2_mean)[:-1] + [n_mixtures]),
                    ),
                    components=[
                        tk.Normal(mean=mu, logstd=logstd, event_ndims=1)
                        for mu, logstd in zip(z2_mean_list, z2_logstd_list)
                    ],
                    reparameterized=True,
                )
            else:
                pz2 = tk.Normal(mean=z2_mean, logstd=z2_logstd, event_ndims=1)
        else:
            pz2 = tk.UnitNormal([1, config.z2_dim], event_ndims=1)

        if config.decoder.use_prior_flow:
            pz2 = tk.FlowDistribution(pz2, self.pz2_flow)

        z2 = net.add('z2', pz2, n_samples=n_z)

        # z2 as context
        z2_shape = T.shape(z2.tensor)
        h = T.reshape(z2.tensor, z2_shape[:-1] + [1, z2_shape[-1]])

        # concat with node type information
        if config.use_operation_embedding:
            h2 = self.operation_embedding(net['node_type'].tensor)
            if config.decoder.operation_embedding_stop_grad:
                h2 = T.stop_grad(h2)
        else:
            h2 = T.one_hot(
                net['node_type'].tensor,
                self.num_node_types,
                dtype=T.float32,
            )
        h = T.broadcast_concat(h, h2, axis=-1)
        h2 = None

        # node_features from gnn
        h_shape = T.shape(h)
        h = T.reshape(
            h,
            h_shape[:-3] + [h_shape[-3] * h_shape[-2], h_shape[-1]]
        )
        node_features = self.pG_node_features(g, h)

        # mean & logstd for p(latency|z2,G)
        if latency_logstd_min is not None:
            if config.decoder.latency_logstd_min is not None:
                latency_logstd_min = max(
                    latency_logstd_min,
                    config.decoder.latency_logstd_min
                )
        else:
            latency_logstd_min = config.decoder.latency_logstd_min

        latency_mean = T.reshape(
            node_features[..., :LATENCY_DIM],  # avg, min, max
            h_shape[:-1] + [LATENCY_DIM]
        )
        latency_logstd = T.maybe_clip(
            T.reshape(
                node_features[..., LATENCY_DIM: LATENCY_DIM*2],
                h_shape[:-1] + [LATENCY_DIM]
            ),
            min_val=latency_logstd_min,
        )

        if std_limit is not None:
            logstd_limit = T.log(
                T.clip_left(
                    std_limit[net['node_type'].tensor],
                    1e-7
                )
            )
            logstd_limit = T.stop_grad(logstd_limit)
            logstd_limit = T.expand_dim(logstd_limit, axis=-1)
            latency_logstd = T.minimum(latency_logstd, logstd_limit)

        # clip the latency
        if config.decoder.clip_latency_to_one_dim:
            latency_mean = latency_mean[..., :1]
            latency_logstd = latency_logstd[..., :1]

        # p(latency|z2,G)
        if config.decoder.use_latency_mask:
            inner_event_ndims = 0
        else:
            inner_event_ndims = 2

        if self.training:
            p_latency = SafeNormal(
                std_threshold=config.decoder.safe_normal_std_threshold,
                mean=latency_mean,
                logstd=latency_logstd,
                event_ndims=inner_event_ndims,
            )
        elif use_biased and config.decoder.use_biased_latency:
            if config.decoder.use_anomaly_detection_normal:
                p_latency = AnomalyDetectionNormal(
                    std_threshold=config.decoder.biased_normal_std_threshold,
                    bias_alpha=MAX_NODE_COUNT,
                    bias_threshold=0.5,
                    mean=latency_mean,
                    logstd=latency_logstd,
                    event_ndims=inner_event_ndims,
                )
            else:
                p_latency = BiasedNormal(
                    alpha=MAX_NODE_COUNT,
                    std_threshold=config.decoder.biased_normal_std_threshold,
                    mean=latency_mean,
                    logstd=latency_logstd,
                    event_ndims=inner_event_ndims,
                )
        else:
            p_latency = tk.Normal(
                mean=latency_mean,
                logstd=latency_logstd,
                event_ndims=inner_event_ndims,
            )

        if config.decoder.use_latency_mask:
            # mask
            mask = node_count_mask(
                net['node_count'].tensor,
                MAX_NODE_COUNT,
                dtype=T.boolean,
            )
            mask = T.stop_grad(mask)
            mask = T.expand_dim(mask, axis=-1)

            # log_prob_weight
            if latency_log_prob_weight:
                log_prob_weight = T.cast(net['node_count'].tensor, dtype=T.float32)
                log_prob_weight = T.float_scalar(MAX_NODE_COUNT) / log_prob_weight
                log_prob_weight = T.reshape(log_prob_weight, T.shape(log_prob_weight) + [1, 1])
                log_prob_weight = T.stop_grad(log_prob_weight)
            else:
                log_prob_weight = None

            # p(latency|...)
            p_latency = MaskedDistribution(p_latency, mask, log_prob_weight, event_ndims=2)

        latency = net.add('latency', p_latency)
