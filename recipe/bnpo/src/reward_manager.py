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

import re
import torch

from verl import DataProto
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def accuracy_reward(model_output, gold_answer):
    """Reward function that checks if the completion is the same as the ground truth."""
    extracted_answer = parse(
        "\\boxed{" + gold_answer + "}",
        extraction_mode="first_match",
        fallback_mode="no_fallback",
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    basic_latex=True,
                    units=True,
                    malformed_operators=True,
                    nits=True,
                    boxed="all",
                ),
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=True,
            )
        ],
    )
    if len(extracted_answer)==0:
        # print(f"fail to extract gold answer {gold_answer}")
        reward=0.0
        extracted_predictions = [model_output]
        result = {
            "score": reward,
            "acc": reward,
            "pred": str(extracted_predictions[0])
        }
        return result
    
    extracted_predictions = parse(
        model_output[-500:],
        extraction_mode="first_match",
        fallback_mode="first_match",
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    basic_latex=True,
                    units=True,
                    malformed_operators=False,
                    nits=False,
                    boxed="all",
                ),
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
    )
    if len(extracted_predictions) == 0:
        # print(f"fail to extract: {model_output}")
        reward=0.0
        extracted_predictions = [model_output]
        result = {
            "score": reward,
            "acc": reward,
            "pred": str(extracted_predictions[0])
        }
        return result
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    try:
        reward = float(verify(extracted_answer, extracted_predictions))
    except Exception as e:
        print(f"verify failed: {e}, prediction: {extracted_predictions}, answer: {extracted_answer}")
        reward = 0.0
    result = {
        "score": reward,
        "acc": reward,
        "pred": str(extracted_predictions[0])
    }
    return result


def format_reward(model_output):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\s*.*?\s*</think><answer>\s*.*?\s*</answer>$"
    match = 1.0 if (
        re.fullmatch(pattern, model_output, flags=re.DOTALL)
        and model_output.count("<think>") == 1
        and model_output.count("</think>") == 1
        and model_output.count("<answer>") == 1
        and model_output.count("</answer>") == 1
    ) else 0.0
    return match


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, use_format_reward=False):
    if data_source in ["openai/gsm8k", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500", "math-ai/amc23", "math-ai/aime24", "math-ai/aime25"]:
        result = accuracy_reward(solution_str, ground_truth)
        if use_format_reward:
            f_reward = format_reward(solution_str)
            result["score"] = result["score"] + f_reward
            result["format"] = f_reward
        return result
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")


class BNPORewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, compute_score=None, reward_fn_key='data_source', use_format_reward=False) -> None:
        self.tokenizer = tokenizer
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.use_format_reward = use_format_reward


    def __call__(self, data: DataProto, return_dict: bool = False, num_examine: int=1):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        acc_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        format_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                use_format_reward=self.use_format_reward
            )

            if isinstance(score, dict):
                reward = score["score"]
                acc_reward = score["acc"]
                if self.use_format_reward:
                    format_reward = score["format"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward
            acc_reward_tensor[i, valid_response_length - 1] = acc_reward
            if self.use_format_reward:
                format_reward_tensor[i, valid_response_length - 1] = format_reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "acc_reward_tensor": acc_reward_tensor,
                "format_reward_tensor": format_reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
