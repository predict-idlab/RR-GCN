import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from rrgcn import RRGCNEmbedder
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from dataset import NodeClassificationDataset
from dataset_location import root
import subprocess


def get_gpu_memory_map():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) * 2**20 for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


seed = 42
n_splits = 5
emb_seeds = list(range(n_splits))
ent_names = ["aifb", "am", "mutag", "bgs"]
kgb_names = ["amplus", "dblp", "dmgfull", "dmg777k", "mdgenre"]

embedding_sizes = [256, 512, 768, 1024]
layers = [1, 2, 3, 4, 5]
ppvs = [True, False]

results = []
results_df = None

results_file = Path("tuning_results.csv")
if results_file.is_file():
    results_df = pd.read_csv(str(results_file))
    results = results_df.to_dict("records")

device = "cuda" if torch.cuda.is_available() else "cpu"

for name in ent_names + kgb_names:
    dataset = NodeClassificationDataset(root, name)
    data = dataset.data.to(device)

    if hasattr(data, "val_idx"):
        folds = [
            ((data.train_idx, data.train_y), (data.val_idx, data.val_y))
        ] * n_splits
    else:
        kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
        folds = [
            (
                (data.train_idx[tr], data.train_y[tr]),
                (data.train_idx[te], data.train_y[te]),
            )
            for tr, te in kf.split(
                data.train_idx.detach().cpu().numpy(),
                data.train_y.detach().cpu().numpy(),
            )
        ]

    for i, ((train_idx, train_y), (val_idx, val_y)) in enumerate(folds):
        for e in embedding_sizes:
            for l in layers:
                for ppv in ppvs:
                    if device != "cpu":
                        torch.cuda.reset_peak_memory_stats(device=device)
                        before_mem = torch.cuda.max_memory_allocated(device=device)
                    else:
                        before_mem = 0
                    if results_df is not None:
                        already_calculated = results_df[
                            (results_df.dataset == name)
                            & (results_df.fold == i)
                            & (results_df.embedding_size == e)
                            & (results_df.num_layers == l)
                            & (results_df.ppv == ppv)
                        ].shape[0]

                        if already_calculated > 0:
                            continue

                    if device != "cpu":
                        free_mem = (
                            torch.cuda.get_device_properties(0).total_memory
                            - get_gpu_memory_map()[0]
                        )
                    else:
                        free_mem = 0

                    embedder = RRGCNEmbedder(
                        num_nodes=data.num_nodes,
                        num_relations=dataset.num_relations,
                        num_layers=l,
                        emb_size=e,
                        device=device,
                        seed=emb_seeds[i],
                        ppv=ppv,
                    )

                    kwargs = {
                        "edge_index": data.edge_index,
                        "edge_type": data.edge_type,
                        "idx": torch.hstack((train_idx, val_idx)),
                    }
                    estimated_mem_usage = embedder.estimated_peak_memory_usage(**kwargs)

                    print(f"Free mem: {free_mem * 1e-6}MB")
                    print(f"Estimated peak mem: {estimated_mem_usage * 1e-6}MB")

                    # In principle we could evaluate any setup for which
                    # "if estimated_mem_usage > free_mem:""
                    # succeeds. But limit to ~24GB to make it fit on
                    # contemporary consumer GPUs (e.g. RTX3090)
                    if estimated_mem_usage > 24 * 10**9:
                        results.append(
                            {
                                "dataset": name,
                                "embedding_size": e,
                                "num_layers": l,
                                "fold": i,
                                "loss": np.nan,
                                "f1": np.nan,
                                "acc": np.nan,
                                "lr": np.nan,
                                "it": np.nan,
                                "peak_mem": np.nan,
                                "seed": emb_seeds[i],
                                "ppv": ppv,
                            }
                        )
                    else:
                        embs = embedder.embeddings(**kwargs)
                        if device != "cpu":
                            peak_mem = (
                                torch.cuda.max_memory_allocated(device=device)
                                - before_mem
                            )
                        else:
                            peak_mem = 0
                        train_embs = embs[: len(train_idx)]
                        val_embs = embs[len(train_idx) :]

                        task_type = "GPU" if torch.cuda.is_available() else "CPU"
                        clf = CatBoostClassifier(
                            iterations=2_000,  # very suboptimal, but much faster
                            # and still valid for relative performance
                            early_stopping_rounds=100,  # idem ^
                            task_type=task_type,
                            random_seed=seed,
                            use_best_model=True,
                            verbose=0,
                            # auto_class_weights="Balanced",
                        )

                        clf = clf.fit(
                            train_embs.cpu().numpy(),
                            train_y.cpu().numpy(),
                            eval_set=(val_embs.cpu().numpy(), val_y.cpu().numpy()),
                        )

                        lr = clf.learning_rate_
                        it = clf.best_iteration_

                        pred = clf.predict(val_embs.cpu().numpy())
                        f1 = f1_score(val_y.cpu().numpy(), pred, average="macro")
                        acc = accuracy_score(val_y.cpu().numpy(), pred)
                        loss = log_loss(
                            val_y.cpu().numpy(),
                            clf.predict_proba(val_embs.cpu().numpy()),
                        )

                        results.append(
                            {
                                "dataset": name,
                                "embedding_size": e,
                                "num_layers": l,
                                "fold": i,
                                "loss": loss,
                                "f1": f1,
                                "acc": acc,
                                "lr": lr,
                                "it": it,
                                "peak_mem": peak_mem,
                                "seed": emb_seeds[i],
                                "ppv": ppv,
                            }
                        )

                        del embs
                        del train_embs
                        del val_embs
                        del clf

                    del embedder
                    gc.collect()
                    if device != "cpu":
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(str(results_file), index=False)
