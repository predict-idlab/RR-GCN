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
ent_names = ["am"]
kgb_names = []

relation_importances = {
    "am": {
        "http://purl.org/collections/nl/am/AHMTextsAuthor": 0.06303783,
        "http://purl.org/collections/nl/am/AHMTextsAuthorLref": 0.053174842,
        "http://purl.org/collections/nl/am/AHMTextsDate": 0.64511883,
        "http://purl.org/collections/nl/am/AHMTextsPubl": 0.12269362,
        "http://purl.org/collections/nl/am/AHMTextsReference": 0.05773992,
        "http://purl.org/collections/nl/am/AHMTextsTekst": 0.12725344,
        "http://purl.org/collections/nl/am/AHMTextsType": 0.22197823,
        "http://purl.org/collections/nl/am/acquisitionDate": 0.42431456,
        "http://purl.org/collections/nl/am/acquisitionMethod": 0.14483288,
        "http://purl.org/collections/nl/am/ahmteksten": 0.48855352,
        "http://purl.org/collections/nl/am/alternativeNumber": 0.29752237,
        "http://purl.org/collections/nl/am/alternativeNumberDate": 0.28878084,
        "http://purl.org/collections/nl/am/alternativeNumberInstitution": 0.13795996,
        "http://purl.org/collections/nl/am/alternativeNumberType": 0.044125266,
        "http://purl.org/collections/nl/am/alternativenumber": 0.11343717,
        "http://purl.org/collections/nl/am/associationPerson": 0.35776043,
        "http://purl.org/collections/nl/am/associationSubject": 0.40650496,
        "http://purl.org/collections/nl/am/biography": 0.4360712,
        "http://purl.org/collections/nl/am/birthDateEnd": 0.4457117,
        "http://purl.org/collections/nl/am/birthDatePrecision": 0.39759082,
        "http://purl.org/collections/nl/am/birthDateStart": 0.23738389,
        "http://purl.org/collections/nl/am/birthNotes": 0.07955016,
        "http://purl.org/collections/nl/am/birthPlace": 0.1856656,
        "http://purl.org/collections/nl/am/collection": 0.19865727,
        "http://purl.org/collections/nl/am/contentMotifGeneral": 0.29103184,
        "http://purl.org/collections/nl/am/contentPersonName": 0.41138935,
        "http://purl.org/collections/nl/am/contentSubject": 0.41133052,
        "http://purl.org/collections/nl/am/creator": 0.14630744,
        "http://purl.org/collections/nl/am/creatorQualifier": 0.028363675,
        "http://purl.org/collections/nl/am/creatorRole": 0.45782918,
        "http://purl.org/collections/nl/am/creditLine": 0.12638736,
        "http://purl.org/collections/nl/am/currentLocation": 0.37523505,
        "http://purl.org/collections/nl/am/currentLocationDateEnd": 0.22378257,
        "http://purl.org/collections/nl/am/currentLocationDateStart": 0.5980421,
        "http://purl.org/collections/nl/am/currentLocationFitness": 0.7327842,
        "http://purl.org/collections/nl/am/currentLocationLref": 0.16002963,
        "http://purl.org/collections/nl/am/currentLocationNotes": 0.5973673,
        "http://purl.org/collections/nl/am/currentLocationType": 0.2272382,
        "http://purl.org/collections/nl/am/deathDateEnd": 0.42766345,
        "http://purl.org/collections/nl/am/deathDatePrecision": 0.30504245,
        "http://purl.org/collections/nl/am/deathDateStart": 0.7499021,
        "http://purl.org/collections/nl/am/deathNotes": 0.03476212,
        "http://purl.org/collections/nl/am/deathPlace": 0.33196434,
        "http://purl.org/collections/nl/am/dimension": 0.39519337,
        "http://purl.org/collections/nl/am/dimensionNotes": 0.7246582,
        "http://purl.org/collections/nl/am/dimensionPart": 0.7982459,
        "http://purl.org/collections/nl/am/dimensionPrecision": 0.10186699,
        "http://purl.org/collections/nl/am/dimensionType": 0.12625578,
        "http://purl.org/collections/nl/am/dimensionUnit": 0.285496,
        "http://purl.org/collections/nl/am/dimensionValue": 0.4558276,
        "http://purl.org/collections/nl/am/documentation": 0.4619388,
        "http://purl.org/collections/nl/am/documentationAuthor": 0.42655697,
        "http://purl.org/collections/nl/am/documentationPageReference": 0.058117356,
        "http://purl.org/collections/nl/am/documentationShelfmark": 0.25491154,
        "http://purl.org/collections/nl/am/documentationSortyear": 0.22580686,
        "http://purl.org/collections/nl/am/documentationTitle": 0.28000435,
        "http://purl.org/collections/nl/am/documentationTitleArticle": 0.06999776,
        "http://purl.org/collections/nl/am/documentationTitleLref": 0.36292106,
        "http://purl.org/collections/nl/am/equivalentName": 0.25707427,
        "http://purl.org/collections/nl/am/exhibition": 0.56281066,
        "http://purl.org/collections/nl/am/exhibitionCatalogueNumber": 0.41291523,
        "http://purl.org/collections/nl/am/exhibitionCode": 0.036488824,
        "http://purl.org/collections/nl/am/exhibitionDateEnd": 0.40187964,
        "http://purl.org/collections/nl/am/exhibitionDateStart": 0.29703894,
        "http://purl.org/collections/nl/am/exhibitionLref": 0.68999124,
        "http://purl.org/collections/nl/am/exhibitionNotes": 0.00086111843,
        "http://purl.org/collections/nl/am/exhibitionObjectLocation": 0.03564944,
        "http://purl.org/collections/nl/am/exhibitionOrganiser": 0.22905755,
        "http://purl.org/collections/nl/am/exhibitionTitle": 0.2774571,
        "http://purl.org/collections/nl/am/exhibitionVenue": 0.52103615,
        "http://purl.org/collections/nl/am/locat": 0.2699622,
        "http://purl.org/collections/nl/am/maker": 0.41290924,
        "http://purl.org/collections/nl/am/name": 0.5375274,
        "http://purl.org/collections/nl/am/objectName": 0.7976162,
        "http://purl.org/collections/nl/am/objectNumber": 0.16461484,
        "http://purl.org/collections/nl/am/occupation": 0.100786716,
        "http://purl.org/collections/nl/am/ownershipHistoryFreeText": 0.27951413,
        "http://purl.org/collections/nl/am/partOfReference": 0.2720498,
        "http://purl.org/collections/nl/am/partsReference": 0.2559356,
        "http://purl.org/collections/nl/am/partsTitle": 0.004209239,
        "http://purl.org/collections/nl/am/priref": 0.23985225,
        "http://purl.org/collections/nl/am/productionDateEnd": 0.32115442,
        "http://purl.org/collections/nl/am/productionDateStart": 0.40604717,
        "http://purl.org/collections/nl/am/productionPeriod": 0.6566071,
        "http://purl.org/collections/nl/am/productionPlace": 0.22909126,
        "http://purl.org/collections/nl/am/relatedObjectAssociation": 0.046732314,
        "http://purl.org/collections/nl/am/relatedObjectNotes": 0.19242144,
        "http://purl.org/collections/nl/am/relatedObjectReference": 0.11984928,
        "http://purl.org/collections/nl/am/relatedObjectTitle": 0.20093197,
        "http://purl.org/collections/nl/am/reproduction": 0.4405587,
        "http://purl.org/collections/nl/am/reproductionCreator": 0.1542454,
        "http://purl.org/collections/nl/am/reproductionDate": 0.48199052,
        "http://purl.org/collections/nl/am/reproductionFormat": 0.085925125,
        "http://purl.org/collections/nl/am/reproductionIdentifierURL": 0.30243084,
        "http://purl.org/collections/nl/am/reproductionNotes": 0.0006066842,
        "http://purl.org/collections/nl/am/reproductionReference": 0.10846674,
        "http://purl.org/collections/nl/am/reproductionReferenceLref": 0.21556842,
        "http://purl.org/collections/nl/am/reproductionType": 0.2079444,
        "http://purl.org/collections/nl/am/scopeNote": 0.36327124,
        "http://purl.org/collections/nl/am/selected": 0.11709955,
        "http://purl.org/collections/nl/am/source": 0.8530823,
        "http://purl.org/collections/nl/am/technique": 0.7431707,
        "http://purl.org/collections/nl/am/termType": 0.51139283,
        "http://purl.org/collections/nl/am/title": 0.20479189,
        "http://purl.org/collections/nl/am/usedFor": 0.14009202,
        "http://purl.org/collections/nl/am/wasPresentAt": 0.2309511,
        "http://www.europeana.eu/schemas/edm/aggregatedCHO": 0.08731055,
        "http://www.europeana.eu/schemas/edm/landingPage": 0.059950467,
        "http://www.europeana.eu/schemas/edm/object": 0.262677,
        "http://www.openarchives.org/ore/terms/aggregates": 0.09818017,
        "http://www.openarchives.org/ore/terms/proxyFor": 0.15166244,
        "http://www.openarchives.org/ore/terms/proxyIn": 0.16704601,
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type": 0.20914431,
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#value": 0.43370706,
        "http://www.w3.org/2000/01/rdf-schema#label": 0.16497347,
        "http://www.w3.org/2004/02/skos/core#altLabel": 0.10515733,
        "http://www.w3.org/2004/02/skos/core#broader": 0.85931337,
        "http://www.w3.org/2004/02/skos/core#exactMatch": 0.1857888,
        "http://www.w3.org/2004/02/skos/core#inScheme": 0.32146212,
        "http://www.w3.org/2004/02/skos/core#narrower": 0.85274255,
        "http://www.w3.org/2004/02/skos/core#prefLabel": 0.43852648,
        "http://www.w3.org/2004/02/skos/core#related": 0.395839,
    }
}

tuning_results_file = Path("tuning_results.csv")
if not tuning_results_file.is_file():
    exit("No tuning results to base hyper parameters on")

tuning_results_df = pd.read_csv(str(tuning_results_file))

results = []
results_df = None

results_file = Path("evaluation_results_relation_cutting.csv")
if results_file.is_file():
    results_df = pd.read_csv(str(results_file))
    results = results_df.to_dict("records")

device = "cuda" if torch.cuda.is_available() else "cpu"

ppvs = [True]
for name in ent_names + kgb_names:
    dataset = NodeClassificationDataset(root, name)
    data = dataset.data.to(device)

    # --- remove unimportant relations
    kept_rels = []
    for rel, i in data.relations_dict.items():
        if (
            str(rel) in relation_importances[name]
            and (relation_importances[name][str(rel)] / max(list(relation_importances[name].values()))) > 0.6
        ):
            kept_rels.append(2 * i)
            kept_rels.append(2 * i + 1)
    kept_rels = torch.tensor(kept_rels).to(data.edge_type.device)
    mask = torch.isin(data.edge_type, kept_rels)
    data.edge_index = data.edge_index[:, mask]
    data.edge_type = data.edge_type[mask]
    # ---

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
                min_node_degree=5,
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
