import mltk
import tensorkit as tk

__all__ = [
    'RealNVPFlowConfig',
    'make_realnvp_flow',
]


class RealNVPFlowConfig(mltk.Config):
    flow_levels: int = 5
    coupling_hidden_layer_count: int = 1
    coupling_hidden_layer_units: int = 64
    coupling_layer_scale: str = 'sigmoid'
    strict_invertible: bool = False


def make_realnvp_flow(z_dim: int, flow_config: RealNVPFlowConfig):
    flows = []
    for i in range(flow_config.flow_levels):
        # act norm
        flows.append(tk.flows.ActNorm(z_dim))

        # coupling layer
        n1 = z_dim // 2
        n2 = z_dim - n1
        b = tk.layers.SequentialBuilder(
            n1,
            layer_args=tk.layers.LayerArgs().
                set_args(['dense'], activation=tk.layers.LeakyReLU)
        )
        for j in range(flow_config.coupling_hidden_layer_count):
            b.dense(flow_config.coupling_hidden_layer_units)
        shift_and_pre_scale = tk.layers.Branch(
            branches=[
                # shift
                b.as_input().linear(n2, weight_init=tk.init.zeros).build(),
                # pre_scale
                b.as_input().linear(n2, weight_init=tk.init.zeros).build(),
            ],
            shared=b.build(),
        )
        flows.append(tk.flows.CouplingLayer(
            shift_and_pre_scale, scale=flow_config.coupling_layer_scale))

        # feature rearrangement by invertible dense
        flows.append(tk.flows.InvertibleDense(z_dim, strict=flow_config.strict_invertible))

    return tk.flows.SequentialFlow(flows)
