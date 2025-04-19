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
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

from .utils import Beam, build_conv, generate_k_steps
from sal.utils.sem_clusters import get_diversity_budget
logger = logging.getLogger()


def _data_gen(batch_of_prompts: list[str],steps: list[list[str]], config: Config, llm: LLM, prm: PRM, em_model):
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=2048,
        top_p=config.top_p,
        stop=[
            "\n\n"
        ],  # we consider that a step in the problem is indicated by a double newline
        include_stop_str_in_output=True,
        n=1,
    )

    beams: list[Beam] = []
    max_steps = 1e9
    for idx,prompt in enumerate(batch_of_prompts):
        max_steps=min(max_steps,len(steps[idx]))
        beams.append(
            Beam(
                prompt=prompt,
                index=idx,
                current_text="",
                next_texts=None,
                lookahead_texts=None,
                best_scores=[0.0],
                all_scores=[],
                previous_text=None,
                pruned=False,
                stop_reasons=None,
                history=[],
                children=[],
                diversity=[],
                org_steps=steps[idx]
            )
        )
    
    for i in tqdm(range(max_steps), desc="Diversity Iterations"):
        # generation
        gen_beams = [b for b in beams if not b.pruned]
        if len(gen_beams) == 0:
            break

        if i == max_steps - 1:
            # last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=2048,
                top_p=config.top_p,
                n=1,
            )

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in gen_beams
        ]
        continue_final_message = i > 0
        add_generation_prompt = i == 0

        tokenizer = llm.get_tokenizer()
        # TODO: set the augmented template from a file
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        lookahead = 0 if i == max_steps - 1 else config.lookahead
        gen_results = generate_k_steps(
            templated_convs, lookahead, llm, sampling_params, config.beam_width
        )

        for beam, gen_result in zip(gen_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.diversity.append(get_diversity_budget(config,beam,em_model))
            beam.children.append(gen_result.next_texts)
            beam.current_text = beam.current_text+beam.org_steps[i]

            # if len(beam.next_texts) != config.beam_width:
            #     beam.pruned = True
            #     # rarely ~1/1000 the model will generate few beams than expected. #TODO: investigate why
            #     logger.warning(
            #         f"beam {beam.index} has {len(beam.next_texts)} completions"
            #     )

        # scoring and chose best generation per beam TODO: add option for selection across beams within the same prompt


    return beams



def data_gen(examples, config: Config, llm: LLM, prm: PRM,em_model):
    problems = examples["problem"]
    steps = examples['steps']
    beam_results = _data_gen(problems,steps,config, llm, prm,em_model)

    # group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"diversity": [], "childs": []}

    for p in problems:
        beams = grouped_results[p]
        results["diversity"].append(beams[0].diversity)
        results["pred"].append(beams[0].children)

    return results