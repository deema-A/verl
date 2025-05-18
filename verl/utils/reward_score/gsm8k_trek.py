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

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

STUDENT_MODEL_NAME = "meta-llama/Llama-3.2-1B"


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


# ----------------------------------------------------------------------------
# Lazy model loader (keeps memory footprint low when the scorer is imported)
# ----------------------------------------------------------------------------

def _get_student_model():
    print("DEEEEMMMMAAAA, DEEMA , Deema #########**********###############")
    """Load the student model / tokenizer the first time we need them."""
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model

tokenizer, model = _get_student_model()
 
# ----------------------------------------------------------------------------
# Prompt construction helper
# ----------------------------------------------------------------------------

def build_prompt(
    guidelines: str,
    original_question: str
    ) -> str:
    """Create the prompt given to the student model."""
    return (
        # "You are a math student. Use the teacher's reasoning guidelines to solve the question. Add \"####\" yobefore your Final answer.\n"
        "Question: " + "Janet’s chicken lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh chicken egg. How much in dollars does she make every day at the farmers' market?\n" #original_question.strip() + "\n"
        "Reasoning Steps:" 
        #+ guidelines.strip() + "\n"
        "1. Identify how many eggs Janet's chicken lay each day: [NUMBER]."
        "2. Determine how many eggs she uses for breakfast each day: [NUMBER]."
        "3. Determine how many eggs she uses to bake muffins: [NUMBER]."
        "4. Add together the eggs used for breakfast and baking to find total eggs used."
        "5. Subtract the total eggs used from the number of eggs laid to find how many eggs are left to sell."
        "6. Multiply the number of eggs she sells by the price per egg: $[AMOUNT]."
        "7. This product gives the amount she earns each day at the farmers' market.\n"
        # "Let\'s think step by step and output the final answer after '####'.\n"
        "Read the question and follow the reasoning steps to understand how to solve it."
        # "Focus on the reasoning — not just the answer."
        "Write your final answer after ####.\n"
        "Solution:"
        "1. Identify how many eggs Janet's chicken lay each day: 16."
        "2. Determine how many eggs she uses for breakfast each day: 3." 
        "3. Determine how many eggs she uses to bake muffins: 4."
        "4. Add together the eggs used for breakfast and baking to find total eggs used: 3 + 4 = 7."
        "5. Subtract the total eggs used from the number of eggs laid to find how many eggs are left to sell: 16 - 7 = 9."
        "6. Multiply the number of eggs she sells by the price per egg: 9 × 2 = 18."
        "7. This product gives the amount she earns each day at the farmers' market.\n"
        "#### 18"
        "\n\n"
        "Question: " + original_question.strip() + "\n"
        "Reasoning Steps:" + guidelines.strip() + "\n"
        "Read the question and follow the reasoning steps to understand how to solve it."
        "Write your final answer after ####.\n"
        "Solution:"

    )# 'Let\'s think step by step and output the final answer after "####".'

# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def compute_score(
    data_source,
    solution_str,
    ground_truth,
    # tokenizer,
    # model,
    extra_info,
    method="strict",
    format_score=0.0,
    score=1.0,
    max_new_tokens=256,
) -> float:
    """Score the student model's answer using GSM8K criteria.

    Args:
        solution_str: reasoning guidelines for the student model (not the answer!)
        ground_truth: the correct answer string
        extra_info: dict that **must** contain "original_question"
        method: extraction method for `extract_solution` ('strict' or 'flexible')
        format_score: score if answer format is correct but value is wrong
        score: score if the value matches `ground_truth`
        max_new_tokens: decoding budget for the student LLM
    """
    if "original_question" not in extra_info:
        raise KeyError("extra_info must contain the key 'original_question'")

    # solution_str = solution_str[:solution_str.index("You are a math tutor helping students")]
    solution_str = solution_str.split("You are a math tutor helping students")[0] if "You are a math tutor helping students" in solution_str else solution_str

    prompt = build_prompt(solution_str, extra_info["original_question"])

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    """
    if "You are a math tutor helping students" in solution_str:
    solution_str = solution_str[:solution_str.index("You are a math tutor helping students")]
    """
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            top_p=None,
            min_new_tokens=1,
            return_dict_in_generate=True,
        )

    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    # # Keep only the part generated *after* the prompt
    student_answer = generated_text[len(prompt):].strip()
    # print("DEEEMA, prompt", prompt)
    # print("#######\n")
    # print("DEEEMA, solution_str", solution_str)
    # print("########\n")
    # # exit(0)
    # print("DEEEMA, original_question", extra_info["original_question"])
    # print("########\n")
    # print("DEEEMA, student_answer", student_answer)
    # # exit(0)
    # # Use existing util to grab the final numeric/text answer after '####'
    answer = extract_solution(solution_str=student_answer, method=method)
    # answer = 3
    if answer is None:
        return 0.0
    elif answer == ground_truth:
        return score
    else:
        return format_score
