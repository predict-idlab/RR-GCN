import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from rrgcn import RRGCNEmbedder
from sklearn.metrics import accuracy_score, f1_score, log_loss

from dataset import NodeClassificationDataset
from dataset_location import root

seed = 42
emb_seeds = list(range(10))
ent_names = ["aifb", "am", "bgs", "mutag"]
kgb_names = ["dmgfull", "dmg777k", "mdgenre", "amplus", "dblp"]

tuning_results_file = Path("tuning_results.csv")
if not tuning_results_file.is_file():
    exit("No tuning results to base hyper parameters on")

tuning_results_df = pd.read_csv(str(tuning_results_file))

results = []
results_df = None

results_file = Path("evaluation_results.csv")
if results_file.is_file():
    results_df = pd.read_csv(str(results_file))
    results = results_df.to_dict("records")

device = "cuda" if torch.cuda.is_available() else "cpu"

ppvs = [True, False]
for name in kgb_names + ent_names:
    dataset = NodeClassificationDataset(root, name)
    data = dataset.data.to(device)

    for ppv in ppvs:
        e, l = (
            tuning_results_df[
                (tuning_results_df.dataset == name) & (tuning_results_df.ppv == ppv)
            ]
            .groupby(["embedding_size", "num_layers"])
            .mean()["loss"]
            .idxmin()
        )
        e = int(e)
        l = int(l)

        lr, it = (
            tuning_results_df[
                (tuning_results_df.dataset == name)
                & (tuning_results_df.ppv == ppv)
                & (tuning_results_df.embedding_size == e)
                & (tuning_results_df.num_layers == l)
            ]
            .loc[:, ["lr", "it"]]
            .max()
            .values
        )

        # hyperparameter tuning was done with auto-set lr based on 2k iterations
        # for speed, this is however suboptimal and we want to scale this to a
        # lower learning rate
        # we also add a margin of 20% of the total iterations to the optimal iterations
        # as underfitting is easier than overfitting for catboost
        # and small kg datasets fit using subset of trainset during hyperparameter
        # tuning
        orig_it = 2000
        new_lr = 0.01
        it = int((int(it) + (0.2 * orig_it)) * (lr / new_lr))
        lr = new_lr

        for s in emb_seeds:
            # print(name, ppv, s)
            torch.cuda.reset_peak_memory_stats(device=device)
            before_mem = torch.cuda.max_memory_allocated(device=device)

            if results_df is not None:
                already_calculated = results_df[
                    (results_df.dataset == name)
                    & (results_df.ppv == ppv)
                    & (results_df.seed == s)
                ].shape[0]

                if already_calculated > 0:
                    continue

            embedder = RRGCNEmbedder(
                num_nodes=data.num_nodes,
                num_relations=dataset.num_relations,
                num_layers=l,
                emb_size=e,
                device=device,
                seed=s,
                ppv=ppv,
            )

            kwargs = {
                "edge_index": data.edge_index,
                "edge_type": data.edge_type,
                "idx": torch.hstack((data.train_idx, data.test_idx)),
            }
            estimated_mem_usage = embedder.estimated_peak_memory_usage(**kwargs)

            free_mem = torch.cuda.mem_get_info(torch.cuda.device(device))[0]
            if estimated_mem_usage > free_mem:
                results.append(
                    {
                        "dataset": name,
                        "embedding_size": e,
                        "num_layers": l,
                        "loss": np.nan,
                        "f1": np.nan,
                        "acc": np.nan,
                        "lr": lr,
                        "it": it,
                        "peak_mem": np.nan,
                        "ppv": ppv,
                        "seed": s,
                    }
                )
            else:
                embs = embedder.embeddings(**kwargs)
                peak_mem = torch.cuda.max_memory_allocated(device=device) - before_mem
                train_embs = embs[: len(data.train_idx)]
                test_embs = embs[len(data.train_idx) :]

                task_type = "GPU" if torch.cuda.is_available() else "CPU"
                clf = CatBoostClassifier(
                    iterations=it,
                    learning_rate=lr,
                    task_type=task_type,
                    random_seed=seed,
                    verbose=0,
                    # auto_class_weights="Balanced",
                )

                clf = clf.fit(train_embs.cpu().numpy(), data.train_y.cpu().numpy())

                pred = clf.predict(test_embs.cpu().numpy())
                f1 = f1_score(data.test_y.cpu().numpy(), pred, average="macro")
                acc = accuracy_score(data.test_y.cpu().numpy(), pred)
                loss = log_loss(
                    data.test_y.cpu().numpy(),
                    clf.predict_proba(test_embs.cpu().numpy()),
                )

                results.append(
                    {
                        "dataset": name,
                        "embedding_size": e,
                        "num_layers": l,
                        "loss": loss,
                        "f1": f1,
                        "acc": acc,
                        "lr": lr,
                        "it": it,
                        "peak_mem": peak_mem,
                        "ppv": ppv,
                        "seed": s,
                    }
                )

                del embs
                del train_embs
                del test_embs

            del embedder
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            results_df = pd.DataFrame(results)
            results_df.to_csv(str(results_file), index=False)
