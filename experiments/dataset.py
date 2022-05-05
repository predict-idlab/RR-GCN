# Almost exactly the same as
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/entities.py
# but original implementation was not deterministic
import logging
import os
import os.path as osp
from collections import Counter
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
)


class SmallKG(InMemoryDataset):
    url = "https://data.dgl.ai/dataset/{}.tgz"

    def __init__(self, root: str, name: str):
        self.name = name.lower()
        assert self.name in ["aifb", "am", "mutag", "bgs"]
        super().__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def num_relations(self) -> int:
        return self.data.edge_type.max().item() + 1

    @property
    def num_classes(self) -> int:
        return self.data.train_y.max().item() + 1

    @property
    def raw_file_names(self) -> List[str]:
        return [
            f"{self.name}_stripped.nt.gz",
            "completeDataset.tsv",
            "trainingSet.tsv",
            "testSet.tsv",
        ]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        path = download_url(self.url.format(self.name), self.root)
        extract_tar(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        import gzip

        import pandas as pd
        import rdflib as rdf

        graph_file, task_file, train_file, test_file = self.raw_paths

        with hide_stdout():
            g = rdf.Graph()
            with gzip.open(graph_file, "rb") as f:
                g.parse(file=f, format="nt")

        freq = Counter(g.predicates())

        relations = sorted(set(g.predicates()), key=lambda p: -freq.get(p, 0))
        subjects = set(g.subjects())
        objects = set(g.objects())
        nodes = sorted(subjects.union(objects))  # is list(.) in original
        # sorted(.) for determinism

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

        if self.name == "am":
            label_header = "label_cateogory"
            nodes_header = "proxy"
        elif self.name == "aifb":
            label_header = "label_affiliation"
            nodes_header = "person"
        elif self.name == "mutag":
            label_header = "label_mutagenic"
            nodes_header = "bond"
        elif self.name == "bgs":
            label_header = "label_lithogenesis"
            nodes_header = "rock"

        labels_df = pd.read_csv(task_file, sep="\t")
        labels_set = set(labels_df[label_header].values.tolist())
        labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}
        nodes_dict = {np.unicode(key): val for key, val in nodes_dict.items()}

        train_labels_df = pd.read_csv(train_file, sep="\t")
        train_indices, train_labels = [], []
        for nod, lab in zip(
            train_labels_df[nodes_header].values, train_labels_df[label_header].values
        ):
            train_indices.append(nodes_dict[nod])
            train_labels.append(labels_dict[lab])

        train_idx = torch.tensor(train_indices, dtype=torch.long)
        train_y = torch.tensor(train_labels, dtype=torch.long)

        test_labels_df = pd.read_csv(test_file, sep="\t")
        test_indices, test_labels = [], []
        for nod, lab in zip(
            test_labels_df[nodes_header].values, test_labels_df[label_header].values
        ):
            test_indices.append(nodes_dict[nod])
            test_labels.append(labels_dict[lab])

        test_idx = torch.tensor(test_indices, dtype=torch.long)
        test_y = torch.tensor(test_labels, dtype=torch.long)

        data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            train_idx=train_idx,
            train_y=train_y,
            test_idx=test_idx,
            test_y=test_y,
            num_nodes=N,
        )
        data.relations_dict = relations_dict

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name.upper()}{self.__class__.__name__}()"


from pathlib import Path
import sys


class KGBench(InMemoryDataset):
    r"""http://kgbench.info

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
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
        os.system(f"git lfs clone {str(Path(root) / 'kgbench')}")

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


class NodeClassificationDataset(InMemoryDataset):
    def __init__(self, root, name):
        kgb_names = ["amplus", "dblp", "dmgfull", "dmg777k", "mdgenre"]
        ent_names = ["aifb", "am", "mutag", "bgs"]
        assert name in kgb_names + ent_names

        if name in kgb_names:
            self.__class__ = KGBench
            self.__init__(root, name)
        else:
            self.__class__ = SmallKG
            self.__init__(root, name)
