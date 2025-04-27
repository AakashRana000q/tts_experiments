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
import random

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
import copy

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

from .utils import Beam, build_conv, generate_k_steps
from sal.utils.sem_clusters import get_semantic_indices,get_diversity_budget,get_num_selects


logger = logging.getLogger()

def _dis(batch_of_prompts: list[str], config: Config, llm: LLM, prm: PRM, em_model = None, problem_id = None):

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

    curr_beams: list[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(config.n):
            curr_beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    best_scores=[0.0],
                    all_scores=[],
                    previous_text=None,
                    pruned=False,
                    stop_reasons=None,
                    history=[],
                )
            )
    completed_beams: list[Beam] = []

    for i in tqdm(range(config.num_iterations), desc="DIS search iterations"):
        old_i = i

        if(i>0 and len(curr_beams)!=config.n_beams):
            repeats = (config.n_beams // len(curr_beams)) + 1

            extended_active_beams = [
                copy.deepcopy(b) for b in (curr_beams * repeats)[: config.n_beams]
            ]
            curr_beams = extended_active_beams
            if len(curr_beams) != config.n_beams:
                raise ValueError(
                    f"Expected {config.n_beams} active beams, but got {len(curr_beams)}"
                )
        if i == config.num_iterations - 1:
            # last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=2048,
                top_p=config.top_p,
                n=1,
            )
        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in curr_beams
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

        lookahead = 0 if i == config.num_iterations - 1 else config.lookahead
        if(i>0):
            gen_results = generate_k_steps(
                templated_convs, lookahead, llm, sampling_params, config.beam_width*2
            )
            budget = []
            for beam, gen_result in zip(curr_beams, gen_results, strict=True):
                beam.next_texts = gen_result.next_texts
                beam.stop_reasons = gen_result.stop_reasons
                beam.lookahead_texts = gen_result.lookahead_texts
                if len(beam.next_texts) != (config.beam_width*2):
                    beam.pruned = True
                    # rarely ~1/1000 the model will generate few beams than expected. #TODO: investigate why
                    logger.warning(
                        f"beam {beam.index} has {len(beam.next_texts)} completions"
                    )
                budget.append(get_diversity_budget(config,beam,em_model))
            num_selects = get_num_selects(config.n,budget)

        else:
            gen_results = generate_k_steps(
                templated_convs, lookahead, llm, sampling_params, 1
            )
            num_selects = [1]*config.n

        prompts, completions = [], []
        active_beams: list[Beam] = []

        for beam, num_bud in zip(curr_beams, num_selects): 
            num_buds = min(num_bud,len(beam.next_texts))
            indices = random.sample(range(len(beam.next_texts)),num_buds)
            for iter in indices:
                new_beam = Beam(
                        prompt=beam.prompt,
                        index=beam.index,
                        current_text=beam.current_text+beam.next_texts[iter],
                        next_texts=[beam.next_texts[iter]],
                        lookahead_texts=[beam.next_texts[iter]],
                        stop_reasons=[beam.stop_reasons[iter]],
                        best_scores=[],
                        all_scores=[],
                        previous_text=None,
                        pruned=False,
                        history=[],
                    )
                if (
                    new_beam.stop_reasons[0] == "EOS"
                    or new_beam.stop_reasons[0] == "length"
                    or new_beam.next_texts[0] == ""
                ):
                    new_beam.completed = True
                    completed_beams.append(new_beam)

                active_beams.append(new_beam)
                prompts.append(new_beam.prompt)
                completions.append([new_beam.current_text])
        del curr_beams

        scores = prm.score(prompts, completions)
        agg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in scores
        ]
        for beam, score in zip(active_beams, scores, strict=True):
            beam.all_scores = score[0]
        
        agg_scores = [
            agg_scores[i] for i, b in enumerate(active_beams) if not b.completed
        ]
        active_beams = [b for b in active_beams if not b.completed]

        if len(active_beams) == 0:
            break

        if len(completed_beams) >= config.n:
            break

        if config.filter_duplicates:
            # Create a dictionary to filter duplicates and retain order
            unique_beam_dict = {}
            for i, b in enumerate(active_beams):
                if b.current_text not in unique_beam_dict:
                    unique_beam_dict[b.current_text] = (
                        i  # Map the unique text to its index
                    )
            active_beams = [active_beams[i] for i in unique_beam_dict.values()]
            agg_scores = [agg_scores[i] for i in unique_beam_dict.values()]

        top_indices = np.argsort(np.array(agg_scores).flatten())[    # make sure it does not change agg_scores - nhi karta
            -(config.n // config.beam_width) :
        ]
        selected_scores = []
        selected_text = []
        curr_beams: list[Beam] = []
        for idx, beam in enumerate(active_beams):
            if idx in top_indices:
                curr_beams.append(beam)
                selected_scores.append(agg_scores[idx])
                selected_text.append(beam.current_text)
        
        get_semantic_indices(config, em_model , selected_text, selected_scores, is_non_dss=True, iteration_number=old_i, problem_id=problem_id,budget=budget)

    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]
    
    if len(completed_beams) != config.n:
        # If we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (config.n // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams with {repeats} repetitions to reach size {config.n}"
        )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams

    return completed_beams

def dis(examples, config: Config, llm: LLM, prm: PRM, em_model=None):
    problems = examples["problem"]
    beam_results = _dis(problems, config, llm, prm, em_model ,examples["unique_id"][0])

    # Group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

    for p in problems:
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        agg_scores = [
            aggregate_scores(b.all_scores, config.agg_strategy) for b in beams
        ]
        pred = completions[np.argmax(agg_scores)]
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["pred"].append(pred)
        results["completion_tokens"].append([b.completion_tokens for b in beams])

    return results