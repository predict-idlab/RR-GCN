.. rrgcn documentation master file, created by
   sphinx-quickstart on Sat Apr  2 12:36:19 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RR-GCN's documentation!
=================================

**RR-GCN** is an extension of `Relational Graph Convolutional Networks (R-GCN) <https://arxiv.org/pdf/1703.06103.pdf>`__ in which the weights are randomly initialised and kept frozen (i.e. no training step is required). As such, our technique is unsupervised and the produced embeddings can be used for any downstream ML task/model. Surprisingly, empirical results indicate that the embeddings produced by our RR-GCN can be competitive to, and even sometimes outperform, end-to-end R-GCNs.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   RR-GCN <rrgcn.rst>
   Contributing, Citing and Contact <ccc.rst>