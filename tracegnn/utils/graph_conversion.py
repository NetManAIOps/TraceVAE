import networkx as nx
import numpy as np
from tracegnn.data.trace_graph import TraceGraphIDManager


def np_to_nx(DV: np.ndarray, DE: np.ndarray, id_manager: TraceGraphIDManager) -> nx.Graph:
    """
        DV: [n x d]
        DE: [n x n x 1] or [n x n]
    """
    # Reshape DE to [n x n]
    if len(DE.shape) == 3:
        DE = DE[:,:,0]

    # Choose Nodes
    nodes_idx = (1.0-np.sum(DV[:,:len(id_manager.operation_id)], axis=-1)) < np.max(DV[:,:len(id_manager.operation_id)], axis=-1)
    DV = DV[nodes_idx]
    DE = DE[nodes_idx][:, nodes_idx]

    DE = (DE + DE.T) / 2

    # Get Node Type
    node_type = np.argmax(DV[:,:len(id_manager.operation_id)], axis=-1)

    # Generate nx Graph
    g: nx.Graph = nx.from_numpy_matrix(DE, create_using=nx.Graph)
    
    for i in range(len(g.nodes)):
        g.nodes[i]['node_type'] = node_type[i]
        g.nodes[i]['operation'] = id_manager.operation_id.reverse_map(node_type[i])

    # MST
    # g = nx.maximum_spanning_tree(g)

    return g
