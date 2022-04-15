from typing import Tuple, Union

import torch
import torch_sparse
from torch_geometric.nn.inits import glorot, uniform
from torch_sparse import SparseTensor


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
        adj_t = SparseTensor(row=adj_t[0], col=adj_t[1]).to(x.device).t()
    adj_t = adj_t.set_value(None, layout=None)
    return torch_sparse.matmul(adj_t, (x > 0).float(), reduce="mean")


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


def uniform_seed(
    shape: Tuple,
    device: Union[torch.device, str] = "cuda",
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Randomly generates a tensor based on a seed and uniform initialization.

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
    torch.nn.init.uniform_(a, a=-1, b=1)
    return a


def fan_out_uniform_seed(
    shape: Tuple,
    device: Union[torch.device, str] = "cuda",
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Randomly generates a tensor based on a seed and uniform initialization
    between -1/fan_out and 1/fan_out.

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
    torch.nn.init.uniform_(a, a=-1 / shape[1], b=1 / shape[1])
    return a


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
    a = torch.zeros(shape, device=device, dtype=dtype)
    torch.nn.init.normal_(a, std=1 / shape[1])
    return a
