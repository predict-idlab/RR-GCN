# RR-GCN: Random Relational Graph Convolutional Networks
[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) code for the paper ["R-GCN: The R Could Stand for Random"](https://arxiv.org/abs/2203.02424)

**RR-GCN** is an extension of [Relational Graph Convolutional Networks (R-GCN)](https://arxiv.org/pdf/1703.06103.pdf) in which the weights are randomly initialised and kept frozen (i.e. no training step is required). As such, our technique is unsupervised and the produced embeddings can be used for multiple downstream ML tasks. Surprisingly, empirical results indicate that the embeddings produced by our RR-GCN can be competitive, and even sometimes outperform, the original R-GCN.

## Minimal example
This snippet generates train and test node embeddings for a KG stored as a PyG Data object. 
```python
from rrgcn import RRGCNEmbedder
embedder = RRGCNEmbedder(num_nodes=data.num_nodes, 
                         num_relations=dataset.num_relations, 
                         num_layers=2, 
                         emb_size=2048,
                         ppv=True)
embeddings = embedder.embeddings(data.edge_index, 
                           data.edge_type,
                           batch_size=0, # full-batch
                           idx=torch.hstack((data.train_idx, data.test_idx)))
                           
train_embeddings = embeddings[:len(data.train_idx)]
test_embeddings = embeddings[len(data.train_idx):] 
```
[Documentation for `RRGCNEmbedder`](rrgcn/random_rgcn_embedder.py#L43) 

[Documentation for `embeddings()`](rrgcn/random_rgcn_embedder.py#L161) 

[Example notebook for the AIFB dataset.](examples/aifb.ipynb)


## Installation
After installing the correct PyG version for your PyTorch/CUDA installation as documented in their [installation instructions](https://github.com/pyg-team/pytorch_geometric#installation), install the RR-GCN package using:

`pip install git+https://github.com/predict-idlab/RR-GCN`

## Cite
```
@article{degraeve2022rgcn,
    author = {Degraeve, Vic and Vandewiele, Gilles and Ongenae, Femke and Van Hoecke, Sofie},
    title = {R-GCN: The R Could Stand for Random},
    journal = {arXiv},
    year = {2022},
    url = {https://arxiv.org/abs/2203.02424}
}
```
