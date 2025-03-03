import numpy as np
from sklearn.cluster import KMeans
import re
import torch
from sklearn.metrics import silhouette_score
import pandas as pd
from sal.config import Config

def clean_solutions(ls):
    cleaned_ls = []
    for solution in ls:
        cleaned_solution = solution.strip()
        cleaned_solution = re.sub(r'\n+', '\n', cleaned_solution)
        cleaned_solution = re.sub(r'\s+', ' ', cleaned_solution)
        cleaned_solution = re.sub(r'\s([.,!?])', r'\1', cleaned_solution)
        cleaned_ls.append(cleaned_solution)
    return cleaned_ls

def generate_embedding(sentences,em_model,em_tokenizer,batch_size):
    device = torch.device("cuda:0")
    em_model.eval()
    em_model.to(device)
    lens = len(sentences)
    all_embeds = []

    for i in range(0, lens, batch_size):
        batch = sentences[i:min(i+batch_size,lens)]

        inputs = em_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = em_model(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = outputs.last_hidden_state[:, 0, :]
        all_embeds.append(hidden_state.cpu().numpy())

    return np.vstack(all_embeds)

def optimal_clusters_silhouette(embeddings,config):
    best_k = 4
    best_score = -1
    total_samples = embeddings.shape[0]
    for k in [4,8,16,32,64,128,256,512]:  #add config
        if k>=total_samples:
            continue

        k = min(k, total_samples-1)
        kmeans = KMeans(n_clusters=k, random_state=config.seed, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)

        if score > best_score:
            best_k = k
            best_score = score

    return best_k

def get_semantic_indices(config:Config,em_model,em_tokenizer,active_beams,agg_scores,is_non_dss = False, iteration_number=0, problem_id=0):
    if not is_non_dss:
        active_text = [b.current_text for b in active_beams]
    else:
        active_text = active_beams

    agg_scores = np.array(agg_scores).flatten()
    cleaned_ls = clean_solutions(active_text)

    embeddings_array = generate_embedding(cleaned_ls,em_model,em_tokenizer,config.em_batch_size)

    num_clusters = optimal_clusters_silhouette(embeddings_array,config)

    kmeans = KMeans(n_clusters=num_clusters, random_state=config.seed)
    kmeans.fit(embeddings_array)
    labels = kmeans.labels_

    num_select = (config.n // config.beam_width)
    num_clusters = len(set(labels))

    // TODO: Add Logging for non dss;
    // Log num_samples , num_clusters, agg_scores, iteration_number, problem_id

    if is_non_dss:
        return num_clusters

    df = pd.DataFrame({"index": range(len(active_text)), "score": agg_scores, "group": labels})
    df = df.sort_values(by="score", ascending=False)

    samples_per_group = num_select // num_clusters

    selected = df.groupby("group").head(samples_per_group)
    remaining_samples = num_select-len(selected)
    remaining = df[~df["index"].isin(selected["index"])].head(remaining_samples)
    final_selection = pd.concat([selected, remaining])

    ret_ind = final_selection["index"].tolist()

    // TODO: Add Logging for dss;
    return ret_ind







