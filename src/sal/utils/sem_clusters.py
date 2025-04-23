import numpy as np
from sklearn.cluster import KMeans
import re
import torch
from sklearn.metrics import silhouette_score
import pandas as pd
from sal.config import Config
from scipy.cluster.hierarchy import linkage, fcluster
import json
import os


def log_semantic_clusters(config, num_samples, num_clusters, agg_scores, iteration_number, problem_id,budget=None):
    """
    Log semantic clustering information to a JSON file, appending new entries.
    If the file doesn't exist, it starts with an empty list.
    """

    print("+"*20,f"Logging  problem_id {problem_id} at iteration {iteration_number}","+"*20)
    log_file = config.log_file
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                if not isinstance(log_data, list):
                    log_data = []  
        except json.JSONDecodeError:
            log_data = []  
    else:
        log_data = []

    if budget is not None:
        new_entry = {
            "num_samples": num_samples,
            "num_clusters": num_clusters,
            "agg_scores": agg_scores.tolist() if hasattr(agg_scores, 'tolist') else agg_scores,
            "iteration_number": iteration_number,
            "problem_id": problem_id,
            "budget":budget,
        }
    else:
        new_entry = {
            "num_samples": num_samples,
            "num_clusters": num_clusters,
            "agg_scores": agg_scores.tolist() if hasattr(agg_scores, 'tolist') else agg_scores,
            "iteration_number": iteration_number,
            "problem_id": problem_id
        }
    log_data.append(new_entry)

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)

    return log_file

def clean_solutions(ls):
    cleaned_ls = []
    for solution in ls:
        cleaned_solution = solution.strip()
        cleaned_solution = re.sub(r'\n+', '\n', cleaned_solution)
        cleaned_solution = re.sub(r'\s+', ' ', cleaned_solution)
        cleaned_solution = re.sub(r'\s([.,!?])', r'\1', cleaned_solution)
        cleaned_ls.append(cleaned_solution)
    return cleaned_ls

def get_embeddings(text,em_model):
    tokens = em_model.tokenizer.encode(text, add_special_tokens=False)
    embeds = []
    if(len(tokens)>256):
        for start in range(0, len(tokens), 128):
            end = min(start + 256, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = em_model.tokenizer.decode(chunk_tokens)  # Convert back to text
            chunk_embedding = em_model.encode(chunk_text,convert_to_tensor=False)  # Get embedding
            embeds.append(chunk_embedding)
            if end == len(tokens):
                break
        return np.mean(np.array(embeds), axis=0)
    return em_model.encode(text, convert_to_tensor=False)

def get_optimal_clusters(liss,em_model,em_batch_size):
    if(len(liss)==1):
        return 1,[0]
    embeddings = []
    for item in liss:
        embeddings.append(get_embeddings(item,em_model))
    embeddings = np.array(embeddings)
    
    Z = linkage(embeddings, method='average', metric='cosine')
    clusters = fcluster(Z, 0.05, criterion='distance') - 1 
    K = len(np.unique(clusters))

    return K,clusters.tolist()


def get_semantic_indices(config:Config,em_model,active_beams,agg_scores,is_non_dss = False, iteration_number=0, problem_id=0,budget=None):
    if not is_non_dss:
        active_text = [b.next_texts[0] for b in active_beams]
    else:
        active_text = active_beams
    agg_scores = np.array(agg_scores).flatten()
    cleaned_ls = clean_solutions(active_text)

    num_clusters,labels = get_optimal_clusters(cleaned_ls,em_model,config.em_batch_size)
    num_select = (config.n // config.beam_width)

    if is_non_dss:
        log_semantic_clusters(
            config,
            num_samples=len(active_text),
            num_clusters=num_clusters,
            agg_scores=agg_scores,
            iteration_number=iteration_number,
            problem_id=problem_id,
            budget=budget
        )
        return num_clusters

    df = pd.DataFrame({"index": range(len(active_text)), "score": agg_scores, "group": labels})
    df = df.sort_values(by="score", ascending=False)

    samples_per_group = num_select // num_clusters

    selected = df.groupby("group").head(samples_per_group)
    remaining_samples = num_select-len(selected)
    remaining = df[~df["index"].isin(selected["index"])].head(remaining_samples)
    final_selection = pd.concat([selected, remaining])

    ret_ind = final_selection["index"].tolist()
    return ret_ind

def get_diversity_budget(config:Config,beam,em_model):
    active_text = list(beam.next_texts)
    cleaned_ls = clean_solutions(active_text)
    num_clusters,_ = get_optimal_clusters(cleaned_ls,em_model,config.em_batch_size)
    ratio_uniq = (num_clusters/len(cleaned_ls))

    if(ratio_uniq<=0.125):   
        return 1
    elif(ratio_uniq<=0.375):
        return 2
    elif(ratio_uniq<=0.75):
        return 3
    return 4

def get_num_selects(target,budget):
    samples = []
    for cat in budget:
        if cat == 1: samples.append(1)
        elif cat == 2: samples.append(3)
        elif cat == 3: samples.append(5)
        elif cat == 4: samples.append(7)

    total = sum(samples)
    rem = [1,2,3]
            
    if total > target:
        surplus = total - target
        for cat in [2,3,4,2,3,4,2,3,4]:
            for i in [idx for idx, c in enumerate(budget) if c == cat]:
                remove = min(surplus, rem[cat-2])
                samples[i] -= remove
                surplus -= remove
                if surplus == 0: break
            if surplus == 0: break

        assert sum(samples) == target, f"Invalid total: {sum(samples)}"

    return samples

def num_selects_bpds(target,budget):
    samples = []
    for cat in budget:
        if cat == 1: samples.append(1)
        elif cat == 2: samples.append(3)
        elif cat == 3: samples.append(5)
        elif cat == 4: samples.append(7)

    total = sum(samples)
    rem = [1,2,3]

    if total < target:
        deficit = target - total
        for cat in [4, 3, 2, 1]:
            for i in [idx for idx, c in enumerate(budget) if c == cat]:
                available = 8 - samples[i]
                add = min(deficit, available)
                samples[i] += add
                deficit -= add
                if deficit == 0: break
            if deficit == 0: break
            
    elif total > target:
        surplus = total - target
        for cat in [2,3,4,2,3,4,2,3,4]:
            for i in [idx for idx, c in enumerate(budget) if c == cat]:
                remove = min(surplus, rem[cat-2])
                samples[i] -= remove
                surplus -= remove
                if surplus == 0: break
            if surplus == 0: break

    assert sum(samples) == target, f"Invalid total: {sum(samples)}"

    return samples
