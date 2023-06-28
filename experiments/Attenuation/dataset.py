import logging
import os
import os.path as osp
from collections import Counter
from collections.abc import Sequence
from typing import Callable, List, Optional
from pathlib import Path
import sys

import numpy as np
import torch
# from torch.utils.data import DataLoader

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
)
from torch_geometric.data.makedirs import makedirs
from torch_geometric.loader import DataListLoader as DataLoader

from pytorch_lightning import LightningDataModule

class KGBench(InMemoryDataset):
    r"""http://kgbench.info
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset
    """

    def __init__(self, root: str, name: str):
        self.name = name.lower()
        assert self.name in ["amplus", "dblp", "dmgfull", "dmg777k", "mdgenre"]

        if not self.is_downloaded(root):
            Path(root).mkdir(exist_ok=True)
            self.download(root)

        with hide_stdout():
            self.data = self.load(root, self.name)

    @property
    def num_relations(self) -> int:
        return self.data.edge_type.max().item() + 1

    @property
    def num_classes(self) -> int:
        return self.data.train_y.max().item() + 1

    def is_downloaded(self, root):
        return (Path(root) / "kgbench").exists()

    def download(self, root):
        os.system(f"git lfs clone https://github.com/pbloem/kgbench-data.git {str(Path(root) / 'kgbench')}")

    def load(self, root, name):
        sys.path.append(str(Path(root) / "kgbench"))
        import kgbench as kg

        kg_data = kg.load(name, torch=True, final=False)
        train_idx, train_y = kg_data.training[:, 0], kg_data.training[:, 1]
        val_idx, val_y = kg_data.withheld[:, 0], kg_data.withheld[:, 1]

        # If final=True, then withheld will contain the testing set
        kg_data = kg.load(name, torch=True, final=True)
        test_idx, test_y = kg_data.withheld[:, 0], kg_data.withheld[:, 1]

        edge_type = torch.hstack(
            (2 * kg_data.triples[:, 1].T, 2 * kg_data.triples[:, 1].T + 1)
        )
        edge_index = torch.hstack(
            (kg_data.triples[:, [0, 2]].T, kg_data.triples[:, [2, 0]].T)
        )

        data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            train_idx=train_idx,
            train_y=train_y,
            test_idx=test_idx,
            test_y=test_y,
            val_idx=val_idx,
            val_y=val_y,
            num_nodes=kg_data.num_entities,
            num_classes = kg_data.num_classes,
            num_relations = kg_data.num_relations,
        )

        y = torch.full(
            (data.num_nodes,), -1, dtype=data.train_y.dtype, device=data.train_y.device
        )
        y[data.train_idx] = data.train_y
        y[data.test_idx] = data.test_y
        y[data.val_idx] = data.val_y

        del kg_data
        return data
    
class hide_stdout(object):
    def __enter__(self):
        self.level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, *args):
        logging.getLogger().setLevel(self.level)
        
        
class KG_LightningDataset(LightningDataModule):
    def __init__(self, root: str = "./", name:str = "amplus"):
        super().__init__()
        self.root = root
        self.name = name
        self.num_workers = 4
        
    def is_downloaded(self, root):
        return (Path(root) / "kgbench").exists()
    
    def download(self, root):
        os.system(f"git lfs clone https://github.com/pbloem/kgbench-data.git {str(Path(root) / 'kgbench')}")
        
        
    def prepare_data(self):
        root = self.root
        name = self.name.lower()
        assert name in ["amplus", "dblp", "dmgfull", "dmg777k", "mdgenre"]

        if not self.is_downloaded(root):
            Path(root).mkdir(exist_ok=True)
            self.download(root)
            
    def setup(self, stage):
        name = self.name
        root = self.root
        # for all stages the steps are the same
        if self.is_downloaded(root):
            raise MisconfigurationError("Must run prepare_data first before running setup")
        self.name = name

        with hide_stdout():
            sys.path.append(str(Path(root) / "kgbench"))
            import kgbench as kg

            kg_data = kg.load(self.name, torch=True, final=False)
            train_idx, train_y = kg_data.training[:, 0], kg_data.training[:, 1]
            val_idx, val_y = kg_data.withheld[:, 0], kg_data.withheld[:, 1]

            # If final=True, then withheld will contain the testing set
            kg_data = kg.load(self.name, torch=True, final=True)
            test_idx, test_y = kg_data.withheld[:, 0], kg_data.withheld[:, 1]

            edge_type = torch.hstack(
                (2 * kg_data.triples[:, 1].T, 2 * kg_data.triples[:, 1].T + 1)
            )
            edge_index = torch.hstack(
                (kg_data.triples[:, [0, 2]].T, kg_data.triples[:, [2, 0]].T)
            )

            self.train_data = [(edge_index, edge_type, train_idx.to(torch.long), train_y.to(torch.long))]
            self.val_data = [(edge_index, edge_type, val_idx.to(torch.long), val_y.to(torch.long))]
            self.test_data = [(edge_index, edge_type, test_idx.to(torch.long), test_y.to(torch.long))]

            self.num_classes = kg_data.num_classes
            self.num_relations = kg_data.num_relations
            self.num_nodes = kg_data.num_entities

            del kg_data
            
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = 1, num_workers = self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size = 1, num_workers = self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size = 1, num_workers = self.num_workers)
    
class EntLightningDataset(LightningDataModule):
    def __init__(self, root: str = "./", name:str = "amplus"):
        super().__init__()
        self.root = root
        self.name = name.lower()
        self.num_workers = 4
        assert self.name in ['aifb', 'am', 'mutag', 'bgs']
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['processed_data.pt']
    
    @property
    def raw_file_names(self) -> List[str]:
        return [
            f'{self.name}_stripped.nt.gz',
            'completeDataset.tsv',
            'trainingSet.tsv',
            'testSet.tsv',
        ]
    
    @property
    def processed_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = self.processed_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.processed_dir, f) for f in to_list(files)]
    
    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = self.raw_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.raw_dir, f) for f in to_list(files)]
        
    def prepare_data(self):
        if files_exist(self.raw_paths):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        url = 'https://data.dgl.ai/dataset/{}.tgz'
        path = download_url(url.format(self.name), self.root)
        extract_tar(path, self.raw_dir)
        os.unlink(path)
        
    def setup(self, stage):
        if files_exist(self.processed_paths):
            self.train_data,self.val_data,self.num_nodes, self.num_relations, self.num_classes = torch.load(self.processed_paths[0])
            return
        import gzip

        import pandas as pd
        import rdflib as rdf

        graph_file, task_file, train_file, test_file = self.raw_paths

        with hide_stdout():
            g = rdf.Graph()
            with gzip.open(graph_file, 'rb') as f:
                g.parse(file=f, format='nt')

        freq = Counter(g.predicates())

        relations = sorted(set(g.predicates()), key=lambda p: -freq.get(p, 0))
        subjects = set(g.subjects())
        objects = set(g.objects())
        nodes = list(subjects.union(objects))

        N = len(nodes)
        R = 2 * len(relations)

        relations_dict = {rel: i for i, rel in enumerate(relations)}
        nodes_dict = {node: i for i, node in enumerate(nodes)}

        edges = []
        for s, p, o in g.triples((None, None, None)):
            src, dst, rel = nodes_dict[s], nodes_dict[o], relations_dict[p]
            edges.append([src, dst, 2 * rel])
            edges.append([dst, src, 2 * rel + 1])

        edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        perm = (N * R * edges[0] + R * edges[1] + edges[2]).argsort()
        edges = edges[:, perm]

        edge_index, edge_type = edges[:2], edges[2]

        if self.name == 'am':
            label_header = 'label_cateogory'
            nodes_header = 'proxy'
        elif self.name == 'aifb':
            label_header = 'label_affiliation'
            nodes_header = 'person'
        elif self.name == 'mutag':
            label_header = 'label_mutagenic'
            nodes_header = 'bond'
        elif self.name == 'bgs':
            label_header = 'label_lithogenesis'
            nodes_header = 'rock'

        labels_df = pd.read_csv(task_file, sep='\t')
        labels_set = set(labels_df[label_header].values.tolist())
        labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}
        nodes_dict = {str(key): val for key, val in nodes_dict.items()}

        train_labels_df = pd.read_csv(train_file, sep='\t')
        train_indices, train_labels = [], []
        for nod, lab in zip(train_labels_df[nodes_header].values,
                            train_labels_df[label_header].values):
            train_indices.append(nodes_dict[nod])
            train_labels.append(labels_dict[lab])

        train_idx = torch.tensor(train_indices, dtype=torch.long)
        train_y = torch.tensor(train_labels, dtype=torch.long)

        test_labels_df = pd.read_csv(test_file, sep='\t')
        test_indices, test_labels = [], []
        for nod, lab in zip(test_labels_df[nodes_header].values,
                            test_labels_df[label_header].values):
            test_indices.append(nodes_dict[nod])
            test_labels.append(labels_dict[lab])

        test_idx = torch.tensor(test_indices, dtype=torch.long)
        test_y = torch.tensor(test_labels, dtype=torch.long)
        
        train_data = Data(edge_index=edge_index, edge_type=edge_type,
                    idx=train_idx, y=train_y, num_nodes=N)
        val_data = Data(edge_index=edge_index, edge_type=edge_type,
                    idx=test_idx, y=test_y, num_nodes=N)
        
        # self.train_data, self.train_slices = InMemoryDataset.collate([train_data])
        # self.val_data, self.val_slices = InMemoryDataset.collate([val_data])
        
        self.train_data = [train_data.edge_index, train_data.edge_type, train_data.idx, train_data.y]
        self.val_data = [val_data.edge_index, val_data.edge_type, val_data.idx, val_data.y]
        
        self.num_nodes = N
        self.num_relations = edge_type.max().item() + 1
        self.num_classes = train_y.max().item() + 1
        
        makedirs(self.processed_dir)        
        torch.save((self.train_data,self.val_data,self.num_nodes, self.num_relations, self.num_classes), 
                   self.processed_paths[0])
        

    def train_dataloader(self):
        return DataLoader([self.train_data], batch_size = 1, num_workers = self.num_workers)
    
    def val_dataloader(self):
        return DataLoader([self.val_data], batch_size = 1, num_workers = self.num_workers)
    
def to_list(value) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]
    
def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])
      
    
        
    
    
        

        