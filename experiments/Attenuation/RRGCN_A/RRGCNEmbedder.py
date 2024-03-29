from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import sklearn
import sklearn.base
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch_geometric.utils.subgraph import k_hop_subgraph
from torch_sparse import SparseTensor
from tqdm import tqdm
import torch.utils.checkpoint as ckpt
from torch_geometric.utils import degree

from .node_encoder import NodeEncoder
from .RRGCN_attenuation_conv import RRGCN_Attenuation
from .utils import calc_ppv


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
        relation_attenuation : bool = False,
        node_attenuation : Union[int, None] = None,
        low_mem_training: bool = False,
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
            relation_attenuation (bool, optional):
                Should attenuation be used on the random weight tranforms. Defaults to False
            node_attenuation (int or None, optional):
                Attenuation type to be applied on the node embeddings. Can be None, 0, or 1.
                None: No node attenuation, 0: single node attenuation, 1: per node attenuation.
            low_mem_training (bool, optional):
                Flag to switch to low gpu memory usage by checkpointing the propagation step of convolution
                
        """
        super().__init__()
        self.device = device
        self.ppv = ppv
        self.num_nodes = num_nodes
        self.emb_size = emb_size
        self.seed = seed
        self.num_layers = num_layers
        self.min_node_degree = 0

        self.layers = nn.ModuleList([
            RRGCN_Attenuation(emb_size, emb_size, num_relations, seed=seed, attenuation = relation_attenuation, device = device, low_mem = low_mem_training)
            for _ in range(num_layers)
        ])

        self.ne = NodeEncoder(
            emb_size=emb_size, num_nodes=num_nodes, seed=self.seed, device=device, attenuation_type = node_attenuation
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

        x = self.layers[0](x = x, **kwargs)

        if self.ppv:
            # Calculate proportion of positive values in 1-hop neighbourhood
            # after first convolution
            ppv = calc_ppv(x, edge_index)

            # Free GPU memory for next conv layer
            ppv = ppv.cpu()

        for i, conv in enumerate(self.layers[1:]):
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

            # Concatenate final conv activations and PPV features
            x = torch.hstack((x, ppv.to(self.device)))

        return x

    def get_last_fit_scalers(self) -> Dict[int, sklearn.base.TransformerMixin]:
        """If during the last call to `embeddings()`, scalers were fit,
        returns the per featured node fitted sklearn scalers.
        Returns:
            Dict[int, sklearn.base.TransformerMixin]: the fitted scalers
        """
        return self.node_features_scalers

    def estimated_peak_memory_usage(
        self,
        edge_index: Union[torch.Tensor, torch_sparse.SparseTensor],
        batch_size: int = 0,
        idx: Optional[torch.Tensor] = None,
        subgraph: bool = True,
        **kwargs
    ):
        """Calculates the theoretical peak memory usage for a set of arguments
        given to `RRGCNEmbedder.embeddings()`
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
            int: Theoretical peak memory usage in number of bytes
        """
        if subgraph:
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

            num_nodes_per_batch = []
            for batch in batches:
                # Calculate batch subgraph with smaller number of nodes and edges
                nodes, _, _, _ = k_hop_subgraph(
                    batch, self.num_layers, edge_index, relabel_nodes=True
                )
                num_nodes_per_batch.append(len(nodes))

            max_num_nodes = max(num_nodes_per_batch)
        else:
            max_num_nodes = edge_index.max() + 1

        factor = 4  # 4 if self.ppv else 3
        return factor * max_num_nodes * self.emb_size * 4  # in number of bytes
    
    # @torch.no_grad()
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
        # self.eval()

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

        embs = torch.tensor([], device = self.device)
        for batch in batches:
            if self.min_node_degree > 0:
                degrees = pd.Series(edge_index[0].cpu().numpy()).value_counts()
                low_degree = set(
                    degrees[degrees < self.min_node_degree].index.to_numpy()
                )
                low_degree = torch.tensor(
                    list(low_degree.difference(set(all_nodes.cpu().numpy())))
                ).to(edge_index.device)

                degree_mask = ~(
                    torch.isin(edge_index[0], low_degree)
                    | torch.isin(edge_index[1], low_degree)
                ).to(edge_index.device)
                deg_edge_index = edge_index[:, degree_mask]
                deg_edge_type = edge_type[degree_mask]
            else:
                deg_edge_index = edge_index
                deg_edge_type = edge_type

            if subgraph:
                # Calculate batch subgraph with smaller number of nodes and edges
                nodes, sub_edge_index, mapping, edge_preserved = k_hop_subgraph(
                    batch, self.num_layers, deg_edge_index, relabel_nodes=True
                )
                sub_edge_type = deg_edge_type[edge_preserved]
            else:
                nodes, sub_edge_index, mapping, sub_edge_type = (
                    torch.arange(edge_index.max() + 1),
                    deg_edge_index,
                    batch,
                    deg_edge_type,
                )

            adj_t = (
                torch.sparse_coo_tensor(
                    indices = sub_edge_index,
                    values = sub_edge_type,
                ).t()
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
            emb = self(adj_t, node_idx=nodes, node_features=sub_node_features)[mapping]
            
            embs = torch.concat([embs, emb])

        return embs.to(self.device)