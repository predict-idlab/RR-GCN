# Based on PyG's RGCNConv implementation
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/rgcn_conv.py
from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.rgcn_conv import masked_edge_index
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul

from .util import glorot_seed


class RandomRGCNConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        seed: int = None,
        **kwargs,
    ):
        r"""Random graph convolution operation, characterized by a single seed.

        Args:
            in_channels (int or tuple):
                Size of each input sample. A tuple
                corresponds to the sizes of source and target dimensionalities.
                In case no input features are given, this argument should
                correspond to the number of nodes in your graph.

            out_channels (int):
                Size of each output sample.

            num_relations (int):
                Number of relations.

            seed (int):
                Random seed (fully characterizes the layer).

            **kwargs (optional):
                Additional arguments of
                :class:`torch_geometric.nn.conv.MessagePassing`.
        """
        super().__init__(aggr="mean", node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.in_channels_l = in_channels[0]
        self.in_channels_r = in_channels[0]

        if seed is None:
            self.seed = np.random.randint(1000000)
        else:
            self.seed = seed

        rng = np.random.default_rng(self.seed)
        self.seeds = rng.integers(1e6, size=num_relations + 1)
        self.root = self.seeds[-1]

    def forward(
        self,
        x: Union[OptTensor, Tuple[OptTensor, Tensor]],
        edge_index: Adj,
        edge_type: OptTensor = None,
    ):
        r"""
        Args:
            x:
                The input node features. Can be either a :obj:`[num_nodes,
                in_channels]` node feature matrix, or an optional
                one-dimensional node index tensor (in which case input features
                are treated as trainable node embeddings).
                Furthermore, :obj:`x` can be of type :obj:`tuple` denoting
                source and destination node features.

            edge_index (LongTensor or SparseTensor):
                The edge indices.

            edge_type:
                The one-dimensional relation type/index for each edge in
                :obj:`edge_index`.
                Should be only :obj:`None` in case :obj:`edge_index` is of type
                :class:`torch_sparse.tensor.SparseTensor`.
                (default: :obj:`None`)
        """
        x_l: OptTensor = None
        x_l = x

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)
        for i in range(self.num_relations):
            weight = glorot_seed(
                (self.in_channels_l, self.out_channels),
                seed=self.seeds[i],
                device=x_r.device,
            )

            tmp = masked_edge_index(edge_index, edge_type == i)

            if x_l.dtype == torch.long:
                out += self.propagate(tmp, x=weight[x_l], size=size)
            else:
                out += self.propagate(tmp, x=(x_l @ weight), size=size)
            del weight

        if self.root is not None:
            root = glorot_seed(
                (self.in_channels_l, self.out_channels),
                seed=self.root,
                device=x_r.device,
            )
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        h = x
        return matmul(adj_t, h, reduce=self.aggr)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, num_relations={self.num_relations})"
        )
