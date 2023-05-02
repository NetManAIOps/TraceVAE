import tensorkit as tk
from tensorkit import tensor as T
from torch.nn import Embedding

__all__ = [
    'OperationEmbedding',
]


class OperationEmbedding(tk.layers.BaseLayer):

    num_operations: int
    embedding_dim: int

    def __init__(self, num_operations: int, embedding_dim: int):
        super().__init__()
        self.num_operations = num_operations
        self.embedding_dim = embedding_dim
        self.node_embedding = Embedding(num_operations, embedding_dim)

    def forward(self, node_type: T.Tensor) -> T.Tensor:
        node_type, shape = T.flatten_to_ndims(node_type, 1)
        node_type = self.node_embedding(node_type)
        node_type = T.unflatten_from_ndims(node_type, shape)
        return node_type
