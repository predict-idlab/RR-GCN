from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch_sparse
from torch_geometric.utils.subgraph import k_hop_subgraph
from torch_sparse import SparseTensor
from tqdm import tqdm

from .random_rgcn_conv import RandomRGCNConv, glorot_seed


def calc_ppv(
    x: torch.Tensor, adj_t: Union[torch.Tensor, torch_sparse.SparseTensor]
) -> torch.Tensor:
    """Calculates 1-hop proportion of positive values per representation dimension

    Args:
        x (torch.Tensor): Input node representations.
        adj_t (torch.Tensor or torch_sparse.SparseTensor): Adjacency matrix.
            Either in 2-row head/tail format or using a SparseTensor.

    Returns:
        torch.Tensor: Proportion of positive values features.
    """
    if isinstance(adj_t, torch.Tensor):
        adj_t = SparseTensor(row=adj_t[0], col=adj_t[1]).to(x.device).t()
    adj_t = adj_t.set_value(None, layout=None)
    return torch_sparse.matmul(adj_t, (x > 0).float(), reduce="mean")


class RRGCNEmbedder(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_layers: int,
        num_relations: int,
        emb_size: int,
        device: Union[torch.device, str] = "cuda",
        ppv: bool = True,
        seed: int = 42,
    ):
        """Random Relational Graph Convolutional Network Knowledge Graph Embedder.

        Args:
            num_nodes (int): Number of nodes in the KG.
            num_layers (int): Number of random graph convolutions.
            num_relations (int): Number of relations in the KG.
            emb_size (int): Desired embedding width.
            device (torch.device or str, optional): PyTorch device to calculate
                embeddings on. Defaults to "cuda".
            ppv (bool, optional): If True, concatenate PPV features to embeddings
                (this effectively doubles the embedding width). Defaults to True.
            seed (int, optional): Seed used to generate random transformations
                (fully characterizes the embedder). Defaults to 42.
        """
        super().__init__()
        self.device = device
        self.ppv = ppv
        self.num_nodes = num_nodes
        self.emb_size = emb_size
        self.seed = seed
        self.num_layers = num_layers

        self.layers = [
            RandomRGCNConv(emb_size, emb_size, num_relations, seed=seed)
            for _ in range(num_layers)
        ]

    def forward(
        self,
        edge_index: Union[torch.Tensor, torch_sparse.SparseTensor],
        edge_type: Optional[torch.Tensor] = None,
        node_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculates node embeddings for a (sub)graph specified by
        a typed adjacency matrix

        Args:
            edge_index (torch.Tensor or torch_sparse.SparseTensor): Adjacency matrix.
                Either in 2-row head/tail format or using a SparseTensor.
            edge_type (torch.Tensor, optional): Types for each edge in `edge_index`.
                Can be omitted if `edge_index` is a SparseTensor where types are
                included as values. Defaults to None.
            node_idx (torch.Tensor, optional): Useful for batched embedding calculation.
                Mapping from node indices used in the given (sub)graph's adjancency
                matrix to node indices in the original graph. Defaults to None.

        Returns:
            torch.Tensor: Node embeddings for given (sub)graph.
        """
        if node_idx is None:
            node_idx = torch.arange(edge_index.max() + 1)

        # Use kwargs to support both torch_sparse adjacency tensors
        # and edge_index, edge_type
        kwargs = {"edge_index": edge_index}
        if edge_type is not None:
            kwargs = {**kwargs, "edge_type": edge_type}

        # Generate initital node embeddings on CPU and only transfer
        # necessary nodes to GPU
        x = glorot_seed(
            (self.num_nodes, self.emb_size),
            seed=self.seed,
            device="cpu",
        )[node_idx, :]
        x = self.layers[0](x.to(self.device), **kwargs)

        if self.ppv:
            # Calculate proportion of positive values in 1-hop neighbourhood
            # after first convolution
            ppv = calc_ppv(x, edge_index)

            # Free GPU memory for next conv layer
            ppv = ppv.cpu()

        for conv in self.layers[1:]:
            x = F.relu(x)
            x = conv(x, **kwargs)

            if self.ppv:
                # Free GPU memory for PPV calculation
                x = x.cpu()

                # Return PPV to GPU
                ppv = ppv.to(self.device)

                # Random message passing with previous PPV as features
                ppv = conv(ppv, **kwargs)

                # Calculate new PPV
                ppv = calc_ppv(ppv, edge_index)

                # Free GPU memory for next conv layer
                ppv = ppv.cpu()

                # Return conv activations to GPU
                x = x.to(self.device)

        if self.ppv:
            # Return conv activations to GPU
            x = x.cpu()

            # Concatenate final conv activations and PPV features
            x = torch.hstack((x, ppv))

        return x

    @torch.no_grad()
    def embeddings(
        self,
        edge_index: Union[torch.Tensor, torch_sparse.SparseTensor],
        edge_type: Optional[torch.Tensor] = None,
        batch_size: int = 0,
        idx: Optional[torch.Tensor] = None,
        subgraph: bool = True,
    ) -> torch.Tensor:
        """Generate embeddings for a given set of nodes of interest.

        Args:
            edge_index (torch.Tensor or torch_sparse.SparseTensor): Adjacency matrix.
                Either in 2-row head/tail format or using a SparseTensor.
            edge_type (torch.Tensor, optional): Types for each edge in `edge_index`.
                Can be omitted if `edge_index` is a SparseTensor where types are
                included as values. Defaults to None.
            batch_size (int, optional): Number of nodes in a single batch.
                For every batch, a subgraph with number of hops equal to the number
                of graph convolutions around the included nodes is extracted and used
                for message passing. If `batch_size` is 0, all nodes of interest
                are contained in a single batch. Defaults to 0.
            idx (torch.Tensor, optional):
                Node indices to extract embeddings for (e.g. indices for
                train- and test entities). If None, extracts embeddings for all nodes
                in the graph. Defaults to None.
            subgraph (bool, optional): If False, the function does not take a
                k-hop subgraph before executing message passing. This is useful for
                small graphs where embeddings can be extracted full-batch and
                calculating the subgraph comes with a significant overhead.
                Defaults to True.

        Returns:
            torch.Tensor: Node embeddings for given nodes of interest
        """
        self.eval()

        if idx is None:
            # Generate embeddings for all nodes
            all_nodes = torch.arange(edge_index.max() + 1)
        else:
            # Only generate embeddings for subset of nodes
            # (e.g. labelled train + test nodes)
            all_nodes = idx

        if batch_size < 1:
            # Full-batch
            batches = [all_nodes]
        else:
            # Split nodes to generate embeddings for into batches
            batches = all_nodes.split(batch_size)

        embs = []
        for batch in tqdm(batches):
            if subgraph:
                # Calculate batch subgraph with smaller number of nodes and edges
                nodes, sub_edge_index, mapping, edge_preserved = k_hop_subgraph(
                    batch, self.num_layers, edge_index, relabel_nodes=True
                )
                sub_edge_type = edge_type[edge_preserved]
            else:
                nodes, sub_edge_index, mapping, sub_edge_type = (
                    torch.arange(edge_index.max() + 1),
                    edge_index,
                    batch,
                    edge_type,
                )

            adj_t = (
                SparseTensor(
                    row=sub_edge_index[0],
                    col=sub_edge_index[1],
                    value=sub_edge_type,
                )
                .to(self.device)
                .t()
            )

            # Calculate embeddings for all nodes participating in batch and then select
            # the queried nodes
            emb = self(adj_t, node_idx=nodes)[mapping].detach().cpu()
            embs.append(emb)

        return torch.concat(embs)
