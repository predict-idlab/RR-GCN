from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint as ckpt

from .utils import fan_out_normal_seed

class NodeEncoder(nn.Module):
    def __init__(
        self,
        emb_size: int,
        num_nodes: int,
        seed: int = 42,
        device: Union[torch.device, str] = "cuda",
        attenuation_type: Union[int, None] = None,
    ):
        """Random (untrained) node encoder for the initial node embeddings,
        supports initial feature vectors (i.e. literal values, e.g. floats or
        sentence/word embeddings).
        The encoder supports nodes of different types, that each have different
        associated feature vectors. Every different "featured" node type should
        have an associated integer identifier.
        Args:
            emb_size (int):
                Desired embedding width.
            num_nodes (int):
                Number of nodes in the KG.
            seed (torch.Tensor, optional):
                Seed used to generate random transformations (fully characterizes the
                embedder). Defaults to 42.
            device (Union[torch.device, str], optional):
                PyTorch device to calculate embeddings on. Defaults to "cuda".
            attenuation_type (int or None, Optional]):
                Attenuation type to be applied on the node embeddings. Can be None, 0, or 1.
                None: No node attenuation, 0: single node attenuation, 1: per node attenuation.
                
        """
        super(NodeEncoder, self).__init__()
        self.emb_size = emb_size
        self.device = device
        self.seed = seed
        self.num_nodes = num_nodes
        self.attenuation_type = attenuation_type
        
        if attenuation_type is None:
            self.attenuation = None
        elif attenuation_type == 0:
            # single scalar to alter the weights
            self.attenuation = nn.Parameter(torch.empty((1,), dtype = torch.float), requires_grad = True)
        elif attenuation_type == 1:
            # scalar per node 
            self.attenuation = nn.Parameter(torch.empty((num_nodes,), dtype = torch.float), requires_grad = True)
        else:
            raise ValueError(f"Expected None, 0 or 1, got {attenuation_type} for attenuation_type")
            
        
        self.reset_parameters()

       
    
    def reset_parameters(self):
        if isinstance(self.attenuation, nn.Parameter):
            torch.nn.init.normal_(self.attenuation, 0.5, 1/self.num_nodes)

    def forward(
        self,
        node_features: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        node_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encodes nodes into an initial (random) representation, with nodes with
        intial features (e.g. numeric literals) taken into account.
        Args:
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
                If None, all nodes are assumed to be feature-less. Defaults to None.
            node_idx (torch.Tensor, optional):
                Useful for batched embedding calculation. Mapping from node indices
                used in the given (sub)graph's adjancency matrix to node indices in the
                original graph. Defaults to None.
        Returns:
            torch.Tensor: Initial node representations
        """
        if node_idx is None:
            node_idx = torch.arange(self.num_nodes).to(self.device)
        else:
            node_idx = node_idx.to(self.device)

        # use fan_out_seed instead of glorot to make range independent of
        # the number of nodes
        node_embs = fan_out_normal_seed(
            (self.num_nodes, self.emb_size),
            seed=self.seed,
            device=self.device,
        )

        if node_features is not None:
            for type_id, (idx, feat) in node_features.items():
                assert (
                    idx.dtype == torch.long
                ), f"Node indices for {type_id} should be long"
                idx = idx.ravel()

                assert torch.isin(
                    idx.to('cpu'), node_idx.to('cpu')
                ).all(), (
                    f"Featured node indices for {type_id} should all be"
                    + "in the current subgraph"
                )
                assert len(feat.shape) == 2, (
                    f"Node feature tensor for {type_id} should be two-dimensional"
                    + "First dimension number of nodes of this type, second dimension"
                    + "number of features."
                )
                assert feat.shape[0] == idx.numel(), (
                    f"Featured node index tensor for {type_id} should have"
                    + "as many elements as there are feature vectors"
                )

                # Use fan_out_seed instead of glorot to easily make the variance
                # of featured nodes equal to those of unfeatured nodes:
                #
                # var(feat @ random_transform) = var(feat) * var(random_transform)
                #                                * num_features
                #                               (var of prod of indep 0-mean distr +
                #                                var of sum of indep distr,
                #                                because of matmul)
                #
                # we want to scale feat such that
                #   var(feat_scaled @ random_transform) == var(node_features)
                # so: var(feat_scaled) =
                #               var(node_features) /
                #               (var(random_tranfsorm) * num_features)
                #
                # we set var(random_transform) equal to var(node_features), thus
                # var(feat_scaled) should be 1/num_features
                random_transform = fan_out_normal_seed(
                    (feat.shape[1], self.emb_size),
                    seed=self.seed + type_id,
                    device=self.device,
                )

                # Here, features are assumed to be normalized. Before matmul with
                # random transform, divide by sqrt(fan_in) to make sure resulting embs
                # have the same variance as non-featured nodes
                node_embs[idx, :] = (
                    (feat / (float(feat.shape[1]) ** (1 / 2)))
                    @ random_transform
                )
                
        # perform the attenuation on the node embeddings
        if self.attenuation_type is None:
            return node_embs[node_idx,:]
        elif self.attenuation_type == 1:
            return (node_embs[node_idx,:] * self.attenuation[node_idx,None])
        elif self.attenuation_type == 0:
            return (node_embs[node_idx,:] * self.attenuation)
        else:
            raise ValueError(f"Expected attenuation_type of 0 or 1 or None, got {self.attenuation_type}")