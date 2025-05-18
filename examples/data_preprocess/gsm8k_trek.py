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
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/gsm8k_trek")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "Deema/GSM8K_Masked"

    dataset = datasets.load_dataset(data_source, "default")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = 'You are a patient math tutor. When given a masked problem with placeholders (e.g., [NUMBER], [UNIT], [VALUE]), return concise, step-by-step reasoning guidelines that let a student solve the problem for any values. Never compute or replace the placeholders; output only the guidelines.'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("masked_question")

            # question = instruction_following + "\nProblem:" + question_raw

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                # "prompt": [
                #     {
                #         "role": "system",
                #         "content": instruction_following,  
                #     },
                #     {
                #         "role": "user",
                #         "content": question_raw,
                #     }
                # ],
                "prompt": f"""You are a math tutor helping students understand how to solve word problems that include placeholders like [NUMBER], [UNIT], or [AMOUNT].\
Your job is to write clear, concise step-by-step reasoning guidelines that explain how to approach and solve the problem — **without replacing or computing any placeholders**.
Focus on explaining the logic behind each step, as if teaching a student how to think through it.  
**Never give a final answer. Just explain the reasoning.**

Problem:
Janet’s ducks lay [NUMBER] eggs per day. She eats [NUMBER] for breakfast every morning and bakes muffins for her friends every day with [NUMBER]. She sells the remainder at the farmers' market daily for $[AMOUNT] per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Reasoning guidelines:
1. Identify how many eggs Janet's ducks lay each day: [NUMBER].
2. Determine how many eggs she uses for breakfast each day: [NUMBER].
3. Determine how many eggs she uses to bake muffins: [NUMBER].
4. Add together the eggs used for breakfast and baking to find total eggs used.
5. Subtract the total eggs used from the number of eggs laid to find how many eggs are left to sell.
6. Multiply the number of eggs she sells by the price per egg: $[AMOUNT].
7. This product gives the amount she earns each day at the farmers' market.


You are a math tutor helping students understand how to solve word problems that include placeholders like [NUMBER], [UNIT], or [AMOUNT].\
Your job is to write clear, concise step-by-step reasoning guidelines that explain how to approach and solve the problem — **without replacing or computing any placeholders**.
Focus on explaining the logic behind each step, as if teaching a student how to think through it.  
**Never give a final answer. Just explain the reasoning.**

Problem:
{question_raw}

Reasoning guidelines:
""",
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "masked_question": question_raw,
                    "original_question": example.pop("original_question"),
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
