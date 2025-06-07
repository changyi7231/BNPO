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

import os
import re
import pandas
import argparse
import datasets


qwen_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
r1_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string: str) -> str:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def extract_solution(solution_str: str) -> str:
    return remove_boxed(last_boxed_only_string(solution_str))


def repeat_rows_in_parquet(input_path: str, output_path: str, n: int) -> None:
    df = pandas.read_parquet(input_path)
    repeated_index = df.index.repeat(n)
    df_repeated = df.loc[repeated_index].reset_index(drop=True)
    
    # remove id column
    if 'id' in df_repeated.columns:
        df_repeated = df_repeated.drop(columns=['id'])
    df_repeated.to_parquet(output_path, index=False)

    df_repeated.to_parquet(output_path, index=False)
    print(f"Repeat dataset {n} times and save to: {output_path}")


def get_prompt(prompt_type, system_prompt, question):
    if prompt_type == "type1":
        prompt = [{
            "role": "system",
            "content": system_prompt
            }, {
            "role": "user",
            "content": question
            }]
    elif prompt_type == "type2":
        prompt = [{
            "role": "system",
            "content": "You are a helpful assistant."
            }, {
            "role": "user",
            "content": question + " " + system_prompt
            }]
    elif prompt_type == "type3":
        prompt = [{
            "role": "user",
            "content": question + " " + system_prompt
            }]
    elif prompt_type == "type4":
        prompt = [{
            "role": "user",
            "content": question
            }]
    else:
        raise ValueError(f"{prompt_type} is not supported")
    return prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/datasets")
    parser.add_argument('--save_dir', type=str, default="./save/data")
    parser.add_argument('--prompt_type', type=str, default="type1")
    parser.add_argument('--system_prompt', type=str, default=qwen_prompt)
    parser.add_argument('--n_repeat', type=int, default=1)
    parser.add_argument('--data_source', default="DigitalLearningGmbH/MATH-lighteval",
                        choices=["openai/gsm8k", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500", "math-ai/amc23", "math-ai/aime24", "math-ai/aime25"])

    args = parser.parse_args()

    if args.data_source == "openai/gsm8k":
        print(f"Loading the {args.data_source} dataset", flush=True)
        dataset = datasets.load_dataset(os.path.join(args.data_dir, args.data_source), "main", trust_remote_code=True)

        train_dataset = dataset['train']
        test_dataset = dataset['test']
        
        def extract_solution_gsm8k(solution_str):
            solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
            assert solution is not None
            final_solution = solution.group(0)
            final_solution = final_solution.split('#### ')[1].replace(',', '')
            return final_solution

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                question = example.pop('question')
                answer = example.pop('answer')
                solution = extract_solution_gsm8k(answer)
                prompt = get_prompt(args.prompt_type, args.system_prompt, question)
                data = {
                    "data_source": args.data_source,
                    "prompt": prompt,
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx
                    }
                }
                return data

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

        save_dir = os.path.join(args.save_dir, args.data_source)
        train_dataset.to_parquet(os.path.join(save_dir, 'train.parquet'))
        test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'))
        print(train_dataset[0])
        print(test_dataset[0])
        
        repeat_rows_in_parquet(os.path.join(save_dir, 'train.parquet'), os.path.join(save_dir, 'train_repeated.parquet'), args.n_repeat)
        repeat_rows_in_parquet(os.path.join(save_dir, 'test.parquet'), os.path.join(save_dir, 'test_repeated.parquet'), args.n_repeat)
        
    elif args.data_source == "DigitalLearningGmbH/MATH-lighteval":
        print(f"Loading the {args.data_source} dataset", flush=True)
        dataset = datasets.load_dataset(os.path.join(args.data_dir, args.data_source), trust_remote_code=True)

        train_dataset = dataset['train']
        test_dataset = dataset['test']

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                question = example.pop('problem')
                answer = example.pop('solution')
                solution = extract_solution(answer)
                prompt = get_prompt(args.prompt_type, args.system_prompt, question)
                data = {
                    "data_source": args.data_source,
                    "prompt": prompt,
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx
                    }
                }
                return data

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
        
        save_dir = os.path.join(args.save_dir, args.data_source)
        train_dataset.to_parquet(os.path.join(save_dir, 'train.parquet'))
        test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'))
        print(train_dataset[0])
        print(test_dataset[0])
        
        repeat_rows_in_parquet(os.path.join(save_dir, 'train.parquet'), os.path.join(save_dir, 'train_repeated.parquet'), args.n_repeat)
        repeat_rows_in_parquet(os.path.join(save_dir, 'test.parquet'), os.path.join(save_dir, 'test_repeated.parquet'), args.n_repeat)
        
    elif args.data_source == "HuggingFaceH4/MATH-500":
        print(f"Loading the {args.data_source} dataset", flush=True)
        dataset = datasets.load_dataset(os.path.join(args.data_dir, args.data_source), trust_remote_code=True)

        test_dataset = dataset['test']

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                question = example.pop('problem')
                answer = example.pop('answer')
                prompt = get_prompt(args.prompt_type, args.system_prompt, question)
                data = {
                    "data_source": args.data_source,
                    "prompt": prompt,
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": answer
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx
                    }
                }
                return data

            return process_fn

        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

        save_dir = os.path.join(args.save_dir, args.data_source)
        test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'))
        print(test_dataset[0])
        
        repeat_rows_in_parquet(os.path.join(save_dir, 'test.parquet'), os.path.join(save_dir, 'test_repeated.parquet'), args.n_repeat)
        
    elif args.data_source == "math-ai/amc23":
        print(f"Loading the {args.data_source} dataset", flush=True)
        dataset = datasets.load_dataset(os.path.join(args.data_dir, args.data_source), trust_remote_code=True)

        test_dataset = dataset['test']

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                question = example.pop('question')
                answer = example.pop('answer')
                prompt = get_prompt(args.prompt_type, args.system_prompt, question)
                data = {
                    "data_source": args.data_source,
                    "prompt": prompt,
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": answer
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx
                    }
                }
                return data

            return process_fn

        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

        save_dir = os.path.join(args.save_dir, args.data_source)
        test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'))
        print(test_dataset[0])
        
        repeat_rows_in_parquet(os.path.join(save_dir, 'test.parquet'), os.path.join(save_dir, 'test_repeated.parquet'), args.n_repeat)
        
    elif args.data_source == "math-ai/aime24":
        print(f"Loading the {args.data_source} dataset", flush=True)
        dataset = datasets.load_dataset(os.path.join(args.data_dir, args.data_source), trust_remote_code=True)

        test_dataset = dataset['test']

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                question = example.pop('problem')
                answer = example.pop('solution')
                prompt = get_prompt(args.prompt_type, args.system_prompt, question)
                data = {
                    "data_source": args.data_source,
                    "prompt": prompt,
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": remove_boxed(answer)
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx
                    }
                }
                return data

            return process_fn

        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

        save_dir = os.path.join(args.save_dir, args.data_source)
        test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'))
        print(test_dataset[0])
        
        repeat_rows_in_parquet(os.path.join(save_dir, 'test.parquet'), os.path.join(save_dir, 'test_repeated.parquet'), args.n_repeat)
        
    elif args.data_source == "math-ai/aime25":
        print(f"Loading the {args.data_source} dataset", flush=True)
        dataset = datasets.load_dataset(os.path.join(args.data_dir, args.data_source), trust_remote_code=True)

        test_dataset = dataset['test']

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                question = example.pop('problem')
                answer = example.pop('answer')
                prompt = get_prompt(args.prompt_type, args.system_prompt, question)
                data = {
                    "data_source": args.data_source,
                    "prompt": prompt,
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": answer
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx
                    }
                }
                return data

            return process_fn

        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

        save_dir = os.path.join(args.save_dir, args.data_source)
        test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'))
        print(test_dataset[0])
        
        repeat_rows_in_parquet(os.path.join(save_dir, 'test.parquet'), os.path.join(save_dir, 'test_repeated.parquet'), args.n_repeat)
    else:
        raise NotImplementedError
