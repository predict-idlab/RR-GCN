from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import sklearn
import sklearn.base
import sklearn.preprocessing
import torch
import torch.nn.functional as F
import torch_sparse
from torch_geometric.utils.subgraph import k_hop_subgraph
from torch_sparse import SparseTensor
from tqdm import tqdm

from .node_encoder import NodeEncoder
from .random_rgcn_conv import RandomRGCNConv
from .util import calc_ppv


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
            num_nodes (int):
                Number of nodes in the KG.

            num_layers (int):
                Number of random graph convolutions.

            num_relations (int):
                Number of relations in the KG.

            emb_size (int):
                Desired embedding width.

            device (torch.device or str, optional):
                PyTorch device to calculate embeddings on. Defaults to "cuda".

            ppv (bool, optional):
                If True, concatenate PPV features to embeddings (this effectively
                doubles the embedding width). Defaults to True.

            seed (int, optional):
                Seed used to generate random transformations (fully characterizes the
                embedder). Defaults to 42.
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

        self.ne = NodeEncoder(
            emb_size=emb_size, num_nodes=num_nodes, seed=self.seed, device=device
        )

    def forward(
        self,
        edge_index: Union[torch.Tensor, torch_sparse.SparseTensor],
        edge_type: Optional[torch.Tensor] = None,
        node_features: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        node_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculates node embeddings for a (sub)graph specified by
        a typed adjacency matrix

        Args:
            edge_index (torch.Tensor or torch_sparse.SparseTensor):
                Adjacency matrix. Either in 2-row head/tail format or using
                a SparseTensor.

            edge_type (torch.Tensor, optional):
                Types for each edge in `edge_index`. Can be omitted if `edge_index` is
                a SparseTensor where types are included as values. Defaults to None.

            node_features (Dict[int, Tuple[torch.Tensor, torch.Tensor]], optional):
                Dictionary with featured node type identifiers as keys, and tuples
                of node indices and initial features as values.

                For example, if nodes `[3, 5, 7]` are literals of type `5`
                with numeric values `[0.7, 0.1, 0.5]`, `node_features` should be:
                `{5: (torch.tensor([3, 5, 7]), torch.tensor([0.7], [0.1], [0.5]))}`

                Featured nodes are not limited to numeric literals, e.g. word embeddings
                can also be passed for string literals.

                The node indices used to specify the locations of literal nodes
                should be included in `node_idx` (if supplied).

            node_idx (torch.Tensor, optional):
                Useful for batched embedding calculation. Mapping from node indices
                used in the given (sub)graph's adjancency matrix to node indices in the
                original graph. Defaults to None.

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

        x = self.ne(node_features, node_idx)

        x = self.layers[0](x, **kwargs)

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

    def get_last_fit_scalers(self) -> Dict[int, sklearn.base.TransformerMixin]:
        """If during the last call to `embeddings()`, scalers were fit,
        returns the per featured node fitted sklearn scalers.

        Returns:
            Dict[int, sklearn.base.TransformerMixin]: the fitted scalers
        """
        return self.node_features_scalers

    @torch.no_grad()
    def embeddings(
        self,
        edge_index: Union[torch.Tensor, torch_sparse.SparseTensor],
        edge_type: Optional[torch.Tensor] = None,
        batch_size: int = 0,
        node_features: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        node_features_scalers: Optional[
            Union[Dict[int, sklearn.base.TransformerMixin], str]
        ] = "standard",
        idx: Optional[torch.Tensor] = None,
        subgraph: bool = True,
    ) -> torch.Tensor:
        """Generate embeddings for a given set of nodes of interest.

        Args:
            edge_index (torch.Tensor or torch_sparse.SparseTensor):
                Adjacency matrix. Either in 2-row head/tail format or using a
                SparseTensor.

            edge_type (torch.Tensor, optional):
                Types for each edge in `edge_index`. Can be omitted if `edge_index` is a
                SparseTensor where types are included as values. Defaults to None.

            batch_size (int, optional):
                Number of nodes in a single batch. For every batch, a subgraph with
                number of hops equal to the number of graph convolutions around the
                included nodes is extracted and used for message passing. If
                `batch_size` is 0, all nodes of interest are contained in a
                single batch. Defaults to 0.

            node_features (Dict[int, Tuple[torch.Tensor, torch.Tensor]], optional):
                Dictionary with featured node type identifiers as keys, and tuples
                of node indices and initial features as values.

                For example, if nodes `[3, 5, 7]` are literals of type `5`
                with numeric values `[0.7, 0.1, 0.5]`, `node_features` should be:
                `{5: (torch.tensor([3, 5, 7]), torch.tensor([0.7], [0.1], [0.5]))}`

                Featured nodes are not limited to numeric literals, e.g. word embeddings
                can also be passed for string literals.

                The node indices used to specify the locations of literal nodes
                should be included in `idx` (if supplied).

           node_features_scalers (Dict[int, TransformerMixin] or str, optional):
                Dictionary with featured node type identifiers as keys, and sklearn
                scalers as values. If scalers are not fit, they will be fit on the data.
                The fit scalers can be retrieved using `.get_last_fit_scalers()`.
                Can also be "standard", "robust", "power", "quantile" as shorthands for
                an unfitted StandardScaler, RobustScaler, PowerTransformer and
                QuantileTransformer respectively.
                If None, no scaling is applied. Defaults to "standard".

            idx (torch.Tensor, optional):
                Node indices to extract embeddings for (e.g. indices for
                train- and test entities). If None, extracts embeddings for all nodes
                in the graph. Defaults to None.

            subgraph (bool, optional):
                If False, the function does not take a k-hop subgraph before executing
                message passing. This is useful for small graphs where embeddings can be
                extracted full-batch and calculating the subgraph comes with a
                significant overhead. Defaults to True.

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

        normalized_node_features = node_features
        if node_features is not None and node_features_scalers is not None:
            normalized_node_features = deepcopy(node_features)

            if isinstance(node_features_scalers, str):
                assert node_features_scalers in [
                    "standard",
                    "robust",
                    "power",
                    "quantile",
                ]

                kwargs = {}
                if node_features_scalers == "standard":
                    scaler = sklearn.preprocessing.StandardScaler
                elif node_features_scalers == "robust":
                    scaler = sklearn.preprocessing.RobustScaler
                elif node_features_scalers == "power":
                    scaler = sklearn.preprocessing.PowerTransformer
                elif node_features_scalers == "quantile":
                    scaler = sklearn.preprocessing.QuantileTransformer
                    kwargs = {**kwargs, "output_distribution": "normal"}

                self.node_features_scalers = {
                    k: scaler(**kwargs) for k, _ in node_features.items()
                }
            else:
                self.node_features_scalers = deepcopy(node_features_scalers)

            for type_id, (typed_idx, feat) in node_features.items():
                if (
                    type_id not in self.node_features_scalers
                    or self.node_features_scalers[type_id] is None
                ):
                    continue

                if not hasattr(self.node_features_scalers[type_id], "n_features_in_"):
                    self.node_features_scalers[type_id] = self.node_features_scalers[
                        type_id
                    ].fit(feat)

                normalized_node_features[type_id] = (
                    typed_idx,
                    torch.tensor(
                        self.node_features_scalers[type_id].transform(feat),
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )

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

            sub_node_features = None
            if node_features is not None:
                sub_node_features = {}
                for type_id, (typed_idx, feat) in normalized_node_features.items():
                    mask = torch.isin(typed_idx.to(nodes.device), nodes)
                    selected_idx = typed_idx[mask]
                    if selected_idx.numel() == 0:
                        continue

                    sub_node_features[type_id] = (selected_idx, feat[mask])

            # Calculate embeddings for all nodes participating in batch and then select
            # the queried nodes
            emb = (
                self(adj_t, node_idx=nodes, node_features=sub_node_features)[mapping]
                .detach()
                .cpu()
            )
            embs.append(emb)

        return torch.concat(embs)
