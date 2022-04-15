# RR-GCN: Random Relational Graph Convolutional Networks 
[![PyPi](https://badge.fury.io/py/rrgcn.svg)](https://pypi.org/project/rrgcn) [![Documentation Status](https://readthedocs.org/projects/rr-gcn/badge/?version=latest)](https://rr-gcn.readthedocs.io/en/latest/?badge=latest) [![Downloads](https://img.shields.io/pypi/dw/rrgcn.svg?logo=pypi&color=1082C2)](https://img.shields.io/pypi/dw/rrgcn.svg?logo=pypi&color=1082C2) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/r-gcn-the-r-could-stand-for-random/node-classification-on-mutag)](https://paperswithcode.com/sota/node-classification-on-mutag?p=r-gcn-the-r-could-stand-for-random)

---


[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) code for the paper ["R-GCN: The R Could Stand for Random"](https://arxiv.org/abs/2203.02424)

**RR-GCN** is an extension of [Relational Graph Convolutional Networks (R-GCN)](https://arxiv.org/pdf/1703.06103.pdf) in which the weights are randomly initialised and kept frozen (i.e. no training step is required). As such, our technique is unsupervised and the produced embeddings can be used for any downstream ML task/model. Surprisingly, empirical results indicate that the embeddings produced by our RR-GCN can be competitive to, and even sometimes outperform, end-to-end R-GCNs.

## Minimal example
This snippet generates train and test node embeddings for a KG stored as a PyG Data object. 
```python
from rrgcn import RRGCNEmbedder
embedder = RRGCNEmbedder(num_nodes=data.num_nodes, 
                         num_relations=dataset.num_relations, 
                         num_layers=2, 
                         emb_size=2048,
                         device="cuda",
                         ppv=True)
embeddings = embedder.embeddings(data.edge_index, 
                           data.edge_type,
                           batch_size=0, # full-batch
                           idx=torch.hstack((data.train_idx, data.test_idx)))
                           
train_embeddings = embeddings[:len(data.train_idx)]
test_embeddings = embeddings[len(data.train_idx):] 
```
[Check our documentation for more information](https://rr-gcn.readthedocs.io/en/latest/index.html)

[Example notebook for AIFB](examples/aifb.ipynb)

[Example notebook for MUTAG (illustration of numeric literal support)](examples/mutag_literals.ipynb)

[Example notebook for AM (illustration of numeric & textual literal support)](examples/am_literals.ipynb)

[Example notebook for ogbn-mag (illustration of support for initial node features)](examples/ogbn_mag.ipynb)

## Installation
After installing the correct PyG version for your PyTorch/CUDA installation as documented in their [installation instructions](https://github.com/pyg-team/pytorch_geometric#installation), install the RR-GCN package using:

`pip install rrgcn`

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
