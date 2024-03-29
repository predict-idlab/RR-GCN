from typing import Dict, Optional, Tuple, Union

import torch

from torch_sparse import SparseTensor
import torch_sparse
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import spmm, degree

def calc_ppv(
    x: torch.Tensor, adj_t: Union[torch.Tensor, torch_sparse.SparseTensor]
) -> torch.Tensor:
    """Calculates 1-hop proportion of positive values per representation dimension
    Args:
        x (torch.Tensor):
            Input node representations.
        adj_t (torch.Tensor or torch_sparse.SparseTensor):
            Adjacency matrix. Either in 2-row head/tail format or using a SparseTensor.
    Returns:
        torch.Tensor: Proportion of positive values features.
    """
    if isinstance(adj_t, torch.Tensor):
        if adj_t.is_sparse:
            values = torch.ones_like(adj_t._values(), dtype=torch.float)
            edge_index = torch.sparse_coo_tensor(indices=adj_t._indices(), values=values, size=adj_t.shape).coalesce()
            deg = degree(edge_index.indices()[0], num_nodes=edge_index.size(0))
            return (torch.sparse.mm(edge_index, ((x > 0).float())).div(deg.view(-1, 1))).nan_to_num(0.)
        else:
            adj_t = SparseTensor(row=adj_t[0], col=adj_t[1]).to(x.device).t()
    adj_t = adj_t.set_value(None, layout=None)
    return torch_sparse.matmul(adj_t, (x > 0).float(), reduce="mean")

def fan_out_normal_seed(
    shape: Tuple,
    device: Union[torch.device, str] = "cuda",
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Randomly generates a tensor based on a seed and normal initialization
    with std 1/fan_out.
    Args:
        shape (Tuple):
            Desired shape of the tensor.
        device (torch.device or str, optional):
            Device to generate tensor on. Defaults to "cuda".
        seed (int, optional):
            The seed. Defaults to 42.
        dtype (torch.dtype, optional):
            Tensor type. Defaults to torch.float32.
    Returns:
        torch.Tensor: The randomly generated tensor
    """
    torch.manual_seed(seed)
    a = torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(a, std=1 / (shape[1]))
    return a

def glorot_seed(
    shape: Tuple,
    device: Union[torch.device, str] = "cuda",
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Randomly generates a tensor based on a seed and Glorot initialization.
    Args:
        shape (Tuple):
            Desired shape of the tensor.
        device (torch.device or str, optional):
            Device to generate tensor on. Defaults to "cuda".
        seed (int, optional):
            The seed. Defaults to 42.
        dtype (torch.dtype, optional):
            Tensor type. Defaults to torch.float32.
    Returns:
        torch.Tensor: The randomly generated tensor
    """
    torch.manual_seed(seed)
    a = torch.zeros(shape, device=device, dtype=dtype)
    glorot(a)
    return a
