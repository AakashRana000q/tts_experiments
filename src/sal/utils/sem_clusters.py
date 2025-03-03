import numpy as np
from sklearn.cluster import KMeans
import re
import torch
from sklearn.metrics import silhouette_score
import pandas as pd
from sal.config import Config
from scipy.cluster.hierarchy import linkage, fcluster

def clean_solutions(ls):
    cleaned_ls = []
    for solution in ls:
        cleaned_solution = solution.strip()
        cleaned_solution = re.sub(r'\n+', '\n', cleaned_solution)
        cleaned_solution = re.sub(r'\s+', ' ', cleaned_solution)
        cleaned_solution = re.sub(r'\s([.,!?])', r'\1', cleaned_solution)
        cleaned_ls.append(cleaned_solution)
    return cleaned_ls


def get_optimal_clusters(liss,em_model,em_batch_size):
    if(len(liss)==1):
        return 1
    embeddings = em_model.encode(liss, batch_size=128, convert_to_tensor=False)
    embeddings = np.array(embeddings)
    
    Z = linkage(embeddings, method='average', metric='cosine')
    clusters = fcluster(Z, 0.05, criterion='distance') - 1 # subtract to get 0-indexed labels
    K = len(np.unique(clusters))

    return K,clusters.tolist()


def get_semantic_indices(config:Config,em_model,active_beams,agg_scores):
    active_text = [b.next_texts[0] for b in active_beams]
    agg_scores = np.array(agg_scores).flatten()
    cleaned_ls = clean_solutions(active_text)

    num_clusters,labels = get_optimal_clusters(cleaned_ls,em_model,config.em_batch_size)

    num_select = (config.n // config.beam_width)

    df = pd.DataFrame({"index": range(len(active_text)), "score": agg_scores, "group": labels})
    df = df.sort_values(by="score", ascending=False)

    samples_per_group = num_select // num_clusters

    selected = df.groupby("group").head(samples_per_group)
    remaining_samples = num_select-len(selected)
    remaining = df[~df["index"].isin(selected["index"])].head(remaining_samples)
    final_selection = pd.concat([selected, remaining])

    ret_ind = final_selection["index"].tolist()
    return ret_ind







