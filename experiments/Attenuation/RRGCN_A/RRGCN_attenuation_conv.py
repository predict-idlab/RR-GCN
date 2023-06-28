import uuid
import os
import tempfile
import contextlib
from typing import *

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.rgcn_conv import masked_edge_index
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import spmm, degree

from .utils import glorot_seed


class RRGCN_Attenuation(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        seed: int = None,
        attenuation : bool = False,
        device : Union[torch.device, str] = "cuda",
        low_mem : bool = False,
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
            attenuation (bool, optional):
                Should attenuation variables be used. Defaults to False
            device (torch.device or str, optional):
                Device to generate tensor on. Defaults to "cuda".
            low_mem (boolean, optional):
                Lower the memory usage of the model during training by using checkpointing for the propagation step 
                (for large graphs the peak memory usage grows very quickly here) 
            **kwargs (optional):
                Additional arguments of
                :class:`torch_geometric.nn.conv.MessagePassing`.
        """
        super(RRGCN_Attenuation, self).__init__(aggr="mean", node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.device = device
        self.low_mem = low_mem

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.in_channels_l = in_channels[0]
        self.in_channels_r = in_channels[0]
        
#         self.propagate_type = {'adj_t' : Union[Adj, torch.Tensor], 'x' : torch.Tensor, 'size' : Tuple[int]}

        if seed is None:
            self.seed = np.random.randint(1000000)
        else:
            self.seed = seed

        rng = np.random.default_rng(self.seed)
        self.seeds = rng.integers(1e6, size=num_relations + 1)
        self.root = self.seeds[-1] if kwargs.get('root', True) else None
        
        self.attenuation = nn.Parameter(torch.empty((num_relations,), dtype = torch.float)) if attenuation else None
        self.reset_parameters()
        
    def reset_parameters(self):
        if isinstance(self.attenuation, nn.Parameter):
            torch.nn.init.normal_(self.attenuation, 0.5, 1/self.num_relations)

    def forward(
        self,
        x: Union[OptTensor, Tuple[OptTensor, torch.Tensor]],
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

        x_r: torch.Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        # Legacy code. torch_sparse is dropped. Use torch.tensor.sparse
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
            tmp = edge_index.clone()
        elif isinstance(edge_index, torch.Tensor) and edge_index.is_sparse:
            # Get the edge types
            edge_type = edge_index._values()
        
        assert edge_type is not None
        
        dev = self.device

        out = torch.zeros(x_r.size(0), self.out_channels, device = dev)
        
        def _loop(i):
            mask = edge_type == i
            if isinstance(edge_index, SparseTensor): # legacy code
                tmp = tmp.detach()
                row, col, value = edge_index.coo()
                row = row[mask]
                col = col[mask]
                value = value[mask]
                tmp.storage._row = row.contiguous()
                tmp.storage._col = col.contiguous()
                tmp.storage._value = value.contiguous()
                tmp.storage._rowptr = None
                tmp.storage._csc2csr = None
                tmp.storage._csr2csc = None

            elif isinstance(edge_index, torch.Tensor): # normal path
                if edge_index.is_sparse: # sparse edge index tensor
                    # Select only the ith relation in the edge types and create a new edge index
                    values = torch.ones_like(edge_index._values()[mask], dtype= torch.float)
                    indices = edge_index._indices()[:, mask]
                    tmp = torch.sparse_coo_tensor(indices, values, size = edge_index.shape).coalesce()
                else: # dense adjacency matrix tensor
                    tmp = edge_index[:,mask]

            att = self.attenuation[i] if self.attenuation is not None else None
            
            # Attenuation weights
            if att is None:
                weight = glorot_seed(
                                    (self.in_channels_l, self.out_channels),
                                    seed=self.seeds[i],
                                    device = dev,
                                )
            else:
                weight = (att * glorot_seed(
                                            (self.in_channels_l, self.out_channels),
                                            seed=self.seeds[i],
                                            device = dev,
                                        )
                         )
            
            # Propagation
            if x_l.dtype == torch.long:
                res = self.propagate(tmp, x=weight[x_l],size=size)
            else:
                res = (self.propagate(tmp, x= x_l, size=size,) @ weight)
                
            # Release unnecessary random transformation matrix
            del weight
            
            return res
        
        out = torch.zeros(x_r.size(0), self.out_channels, device = self.device)

        for i in range(self.num_relations):
            if self.low_mem:
                out = ckpt.checkpoint(_loop, i, use_reentrant= False, preserve_rng_state = False) + out
            else:
                out = out + _loop(i)

        if self.root is not None:
            # for self relation
            root = glorot_seed(
                (self.in_channels_l, self.out_channels),
                seed=self.root,
                device = self.device,
            )

            out = ((root[x_r] if x_r.dtype == torch.long else x_r @ root) + out)
            
        return out


    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Union[SparseTensor, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        if isinstance(adj_t, torch.Tensor) and adj_t.is_sparse:
            deg = degree(adj_t.indices()[0], num_nodes=adj_t.size(0))
            return (torch.sparse.mm(adj_t, x).div(deg.view(-1, 1))).nan_to_num(0.)
        adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, num_relations={self.num_relations})"
        )