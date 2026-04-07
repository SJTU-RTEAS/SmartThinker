# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.utils.logger.aggregate_logger import print_rank_0, print_with_rank  

import re
from transformers import AutoTokenizer

import requests

class SmartThinkerRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine=1, compute_score=None, reward_fn_key="data_source", **kwargs) -> None:
        """
        Initialize the SmartThinkerRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        #self.train_batch_size = kwargs.get("train_batch_size", 64)
        self.num_generation = kwargs.get("num_generation", 8)

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        #assert len(data) == self.num_generation * self.train_batch_size, f"Implementation Error: the input data should have len={self.train_batch_size * self.num_generation}, where train_batch_size={self.train_batch_size}, num_generation={self.num_generation}, while len(data)={len(data)}"

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        step_sim_score_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        
        problem_ids = [data_item.non_tensor_batch['extra_info']['index'] for data_item in data]
        unique_problem_ids = []
        for pid in problem_ids:
            if pid not in unique_problem_ids:
                unique_problem_ids.append(pid)

        all_acc_score_list = [0.0] * len(data)
                
        print("====Starting StepPrune Reward Computation====")

        # Collect all response strings across all problem groups for batched processing
        all_response_str_list = []
        problem_response_counts = []  # To track how many responses per problem
        for problem_id in unique_problem_ids:
            # Find all instances of this problem in the batch
            indices = [i for i in range(len(data)) if data[i].non_tensor_batch['extra_info']['index'] == problem_id]

            # Should have num_generation repetitions of each problem
            assert len(indices) == self.num_generation, f"Expected {self.num_generation} repetitions for problem {problem_id}, got {len(indices)}"

            data_group = [data[i] for i in indices]

            response_str_list = []
            for i in range(len(data_group)):
                data_item = data_group[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                response_str_list.append(response_str)

            all_response_str_list.extend(response_str_list)
            problem_response_counts.append(len(response_str_list))

        # Batch compute all step-wise similarities
        #print("Computing all step-wise similarity scores in batch...")
        #all_sim_list, all_steps_span_list = self._step_sim(all_response_str_list)
        #print("All step-wise similarity computation done.")

        # Reset for per-problem processing
        sim_start_idx = 0

        for problem_idx, problem_id in enumerate(unique_problem_ids):
            # Find all instances of this problem in the batch
            indices = [i for i in range(len(data)) if data[i].non_tensor_batch['extra_info']['index'] == problem_id]

            data_group = [data[i] for i in indices]

            acc_score_list = []
            valid_prompt_length_list = []
            valid_response_length_list = []
            prompt_str_list = []
            response_str_list = []
            ground_truth_list = []

            for i in range(len(data_group)):
                data_item = data_group[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                response_ids = data_item.batch["responses"]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
                rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
                extra_info["num_turns"] = num_turns
                extra_info["rollout_reward_scores"] = rollout_reward_scores

                acc_score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

                acc_score_list.append(acc_score)
                valid_prompt_length_list.append(valid_prompt_length)
                valid_response_length_list.append(valid_response_length)
                prompt_str_list.append(prompt_str)
                response_str_list.append(response_str)
                ground_truth_list.append(ground_truth)


            for i in range(len(indices)):
                acc_score = acc_score_list[i]
                reward_tensor[indices[i], valid_response_length_list[i] - 1] = acc_score
            
            for i in range(len(indices)):
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", prompt_str_list[i][:100])
                    print("[response]", response_str_list[i][-100:])
                    print("[ground_truth]", ground_truth_list[i])
                    print("[valid_response_length]", valid_response_length_list[i])
                    print("[acc_score]", acc_score_list[i])

                all_acc_score_list[indices[i]] = acc_score_list[i]

            # Get the pre-computed similarities for this problem group
            num_responses = problem_response_counts[problem_idx]
            sim_start_idx += num_responses

        reward_extra_info['acc_score'] = all_acc_score_list

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor