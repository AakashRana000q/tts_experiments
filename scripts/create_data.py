import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import torch
from vllm import LLM

from sal.config import Config
from datasets import Dataset,concatenate_datasets

import pandas as pd
import os
from sal.search import data_gen
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

APPROACHES = {
    "data_pb": data_gen,
}

def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()
    approach_fn = APPROACHES[config.approach]

    num_gpus = torch.cuda.device_count()
    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
    )
    em_model = SentenceTransformer(config.em_path,device = torch.device("cuda:0"))

    df = pd.read_json("/workspace/tts_experiments/combined_pb.json")
    df = df.groupby('fin_source', group_keys=False).apply(lambda x: x.sample(n=2, random_state=42))
    df['num_steps'] = df['steps'].apply(lambda x:len(x))
    dataset = Dataset.from_pandas(df)
    print("********************* Length = ",len(df),"*********************")
    
    if config.push_to_hub==False:
        os.makedirs(f"/workspace/tts_experiments/data/{config.model_path}", exist_ok=True)

    processed_splits = []
    for n in set(dataset["num_steps"]):
        bucket = dataset.filter(lambda ex: ex["num_steps"] == n)
        processed = bucket.map(
            approach_fn,           # your preprocessing function
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "llm": llm,"em_model":em_model},
            desc="Running search",
            load_from_cache_file=False,
        )
        processed_splits.append(processed)

    dataset = concatenate_datasets(processed_splits)

    save_dataset(dataset, config)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()



