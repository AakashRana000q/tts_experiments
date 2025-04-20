#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


import torch
from vllm import LLM

from sal.config import Config
from sal.models.reward_models import load_prm
from datasets import Dataset
import pandas as pd
import os
from sal.search import beam_search, best_of_n, dvts, dss, dis, bpds
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
    "dss":dss,
    "dis":dis,
    "bpds":bpds,
}


def main():
    start_time = time.time()

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
    prm = load_prm(config)
    em_model = SentenceTransformer(config.em_path,device = torch.device("cuda:0"))

    dataset = get_dataset(config)
    df = pd.DataFrame(dataset)
    df = df[(df['level']==4) | (df['level']==5)]
    df = df.groupby('level', group_keys=False).apply(lambda x: x.sample(n=50, random_state=42))
    dataset = Dataset.from_pandas(df)
    print("\n\n","********************* Length = ",len(df),"*********************","\n\n")
    print("\n\n","********************* Search Batch Size = ",config.search_batch_size,"*********************","\n\n")
    os.makedirs(config.log_dir, exist_ok=True)
    
    if config.push_to_hub==False:
        os.makedirs(f"/workspace/tts_experiments/data/{config.model_path}", exist_ok=True)
    print("********************* Agg strategy = ",config.agg_strategy,"*********************")
    
    if(config.approach=="bpds"):
        dataset = dataset.map(
            approach_fn,
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "llm": llm, "prm": prm,"em_model":em_model},
            desc="Running search",
            load_from_cache_file=False,
        )
    else:
        dataset = dataset.map(
            approach_fn,
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "llm": llm, "prm": prm},
            desc="Running search",
            load_from_cache_file=False,
        )

    dataset = score(dataset, config)

    save_dataset(dataset, config)
    logger.info("Done 🔥!")
    end_time = time.time()
    print(f"Search took {end_time - start_time:.2f} seconds to run.")


if __name__ == "__main__":
    main()
