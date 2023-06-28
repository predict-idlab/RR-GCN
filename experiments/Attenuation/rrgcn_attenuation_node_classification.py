import gc
import os
import itertools
import logging
import sys
import pickle as pkl
from argparse import ArgumentParser

import torch_geometric
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.functional import recall, precision, f1_score, accuracy
import pytorch_lightning as pl

from RRGCN_A.RRGCNEmbedder import RRGCNEmbedder
from dataset import EntLightningDataset, KG_LightningDataset

class Model(pl.LightningModule):
    def __init__(self, relation_attenuation, node_attenuation = None, emb_dim = 512, num_classes = None, num_nodes = None, 
                 num_rel = None, ppv = False, num_rrgcn = 3, num_mlp = 1, dropout = 0.1, subgraph = True, feature_scaler = None,
                lr = 0.0001, low_mem_training = False, device = 'cuda'):
        super(Model, self).__init__()
        assert num_classes is not None
        self.subgraph = subgraph
        self.lr = lr
        self.feature_scaler = feature_scaler
        self.embedding = RRGCNEmbedder(num_nodes=num_nodes, num_relations=num_rel, num_layers = num_rrgcn,
                                       emb_size = emb_dim, ppv = ppv, device = device, relation_attenuation = relation_attenuation,
                                       node_attenuation = node_attenuation, low_mem_training = low_mem_training,
                                      )
        hidden_size = (ppv+1)*emb_dim
        self.linear = nn.Sequential(nn.GELU(),
                            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout)) for _ in range(num_mlp - 1)],
                            nn.Linear(hidden_size, num_classes)
        )
        self.num_classes = num_classes
        self.validation_step_outputs = []
        self.save_hyperparameters()
        
    def forward(self, edge_index, edge_type, idx):
        embs = self.embedding.embeddings(edge_index, edge_type, batch_size = 0, node_features = None, 
                                         node_features_scalers = self.feature_scaler, idx = idx, subgraph = self.subgraph)

        y = self.linear(embs)
        return embs, y
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)
    
    def training_step(self, batch, batch_idx):
        edge_index, edge_type, idx, y = batch[0]
        _, preds = self(edge_index, edge_type, idx)
        loss = F.cross_entropy(preds, y)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=1)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        edge_index, edge_type, idx, y = batch[0]
        _, preds = self(edge_index, edge_type, idx)
        loss = F.cross_entropy(preds, y)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=1)
        self.validation_step_outputs.append({"pred":preds,"y": y})
        return {"pred":preds,"y": y,"loss": loss}
    
    def on_validation_epoch_end(self):
        outputs= self.validation_step_outputs
        preds = torch.stack([o['pred'] for o in outputs]).view(-1,self.num_classes)
        y = torch.stack([o['y'] for o in outputs]).view(-1)
        
        log_metric = {
            'weighted_recall' : recall(preds, y,num_classes = self.num_classes, task = 'multiclass', average= 'weighted').cpu().tolist(),
            'weighted_precision' : precision(preds, y ,num_classes = self.num_classes, task = 'multiclass', average = 'weighted').cpu().tolist(),
            'weighted_f1_score' : f1_score(preds, y ,num_classes = self.num_classes, task = 'multiclass', average = 'weighted').cpu().tolist(),
            'accuracy' : accuracy(preds, y, num_classes = self.num_classes, task = 'multiclass',).cpu().tolist()

        }
        class_metrics = {
            'recall' : recall(preds, y, num_classes = self.num_classes, task = 'multiclass', average= None).cpu().tolist(),
            'precision' : precision(preds, y, num_classes = self.num_classes, task = 'multiclass', average = None).cpu().tolist(),
            'f1_score' : f1_score(preds, y, num_classes = self.num_classes, task = 'multiclass', average = None).cpu().tolist(),
        }
        macro_metrics = {
            'macro_recall' : recall(preds, y,num_classes = self.num_classes, task = 'multiclass', average= 'macro').cpu().tolist(),
            'macro_precision' : precision(preds, y ,num_classes = self.num_classes, task = 'multiclass', average = 'macro').cpu().tolist(),
            'macro_f1_score' : f1_score(preds, y ,num_classes = self.num_classes, task = 'multiclass', average = 'macro').cpu().tolist(),
        }
        self.log_dict(log_metric, on_step=False, on_epoch=True, prog_bar = True, logger = True, sync_dist=True, rank_zero_only=True, batch_size=1)
        self.log_dict(macro_metrics, on_step=False, on_epoch=True, prog_bar=False, logger = True, sync_dist=True, rank_zero_only=True, batch_size=1)
        
        for i,(p,r,f1) in enumerate(zip(*class_metrics.values())):
            class_prefix = f"Class_{i}_"
            self.log(class_prefix+'recall', r, on_step=False, on_epoch=True, prog_bar = False, logger = True, sync_dist=True, rank_zero_only=True, batch_size=1)
            self.log(class_prefix+'precision', p, on_step=False, on_epoch=True, prog_bar = False, logger = True, sync_dist=True, rank_zero_only=True, batch_size=1)
            self.log(class_prefix+'f1_score', f1, on_step=False, on_epoch=True, prog_bar = False, logger = True, sync_dist=True, rank_zero_only=True, batch_size=1)
            
        self.validation_step_outputs.clear()
            
def pl_train(config, dataset_name, epochs, lr, **kwargs):
    num_classes, num_nodes, num_relations = None,None,None
    if dataset_name in ['AIFB','AM','BGS','MUTAG']:
        
        root = './'
        name = dataset_name
        datamodule = EntLightningDataset(root=root, name=name)
        datamodule.prepare_data()
        datamodule.setup(None)
        num_classes = datamodule.num_classes
        num_nodes = datamodule.num_nodes
        num_relations = datamodule.num_relations
        datamodule.num_workers = 2
    else:
        root = './'
        name = dataset_name
        datamodule = KG_LightningDataset(root=root, name=name)
        datamodule.prepare_data()
        datamodule.setup(None)
        num_classes = datamodule.num_classes
        num_nodes = datamodule.num_nodes
        num_relations = datamodule.num_relations

    rel_att = config['rel_att']
    node_att = config['node_att']
    ppv = config['ppv']
    net = Model(relation_attenuation = rel_att, node_attenuation = node_att, 
                num_nodes = num_nodes, num_rel = num_relations, ppv = ppv,
                num_classes = num_classes, lr = lr, **kwargs)

    csv_logger = pl.loggers.CSVLogger('./logs',name = f'{dataset_name}_{rel_att}_{node_att}_{ppv}')
    trainer = pl.Trainer(
        logger = [csv_logger],
        max_epochs = epochs,
        num_sanity_val_steps=0,
        log_every_n_steps = 1,
        enable_checkpointing = False,
        enable_model_summary = False,
        check_val_every_n_epoch = 50,
        accelerator="gpu",
        devices=1,
        callbacks = [
            pl.callbacks.EarlyStopping(monitor='val_loss', 
                                    mode = 'min',
                                    patience = 3),
        ],
    )
    trainer.fit(net, datamodule)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p","--parameter-list", help="Delimited parameter list", type=str)
    parser.add_argument("-d","--datasets", help="List of delimited datasets", type=str)
    parser.add_argument("-e","--epochs", type=int, default=3000)
    parser.add_argument("-lr", help="Learning rate for models", type=float, default=0.001)
    args = parser.parse_args()
    
    if args.parameter_list is not None:
        params = args.parameter_list.split(',')
        assert len(params) % 3 == 0, "List of params must be divisible by 3"
        params = [tuple(u) for u in zip(params[::3],params[1::3],params[2::3])]
    else:
        params = list(itertools.product([True,False],[None, 0 , 1],[True, False])) # relational attenuation, node attenuation, ppv
        
    if args.datasets is not None:
        dataset_names = args.datasets.split(',')
    else:
        dataset_names = ["AIFB","MUTAG","BGS","AM"]
    
    model_params = {
        'AIFB' : { 'num_rrgcn' : 4, 'emb_dim' : 256}, #{N: 8285, Rel: 45, E: 29043}
        'MUTAG' : { 'num_rrgcn' : 5, 'emb_dim' : 512}, # {N: 23644, Rel: 23, E: 74227}
        'BGS' : { 'num_rrgcn' : 5, 'emb_dim' : 256}, # {N: 333845, Rel: 103, E: 916199}
        'AM' : { 'num_rrgcn' : 2, 'emb_dim' : 256}, # {N: 1666764 Rel: 133, E: 5988321}
        'amplus' : { 'num_rrgcn' : 5, 'emb_dim' : 32},
        'dblp' : { 'num_rrgcn' : 5, 'emb_dim' : 16},
        'dmgfull' : { 'num_rrgcn' : 2, 'emb_dim' : 32},
        'dmg777k' : { 'num_rrgcn' : 2, 'emb_dim' : 32},
        'mdgenre' : { 'num_rrgcn' : 5, 'emb_dim' : 32}
    }# optimal hyperparameters for rrgcn w/o ppv

    MAX_RETRY = 2
    for dataset_name in dataset_names:
        res = {}
        for param in params:
            # Note the retry only works if using cuda memory else it will run twice for no reason
            retry = 0 # 0 -> normal traing, 1-> use checkpointing on propagation in convolution
            while retry < MAX_RETRY:
                torch.cuda.reset_peak_memory_stats(device = torch.device('cuda'))
                config = {"rel_att" : param[0], "node_att": param[1], 'ppv' : param[2]}
                print(f"{dataset_name}, {param[0]}, {param[1]}, {param[2]}")
                try:
                    pl_train(config, dataset_name, epochs = args.epochs, lr = args.lr, low_mem_training = retry, **model_params[dataset_name])
                    break
                except torch.cuda.OutOfMemoryError as e:
                    if retry < MAX_RETRY:
                        print('| WARNING: ran out of memory, retrying param')
                        torch.cuda.empty_cache()
                        gc.collect()
                        retry += 1
                    else:
                        raise e