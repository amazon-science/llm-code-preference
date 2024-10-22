# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: Apache-2.0

# This script is partially adapted from
# https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/pair-pm/process_pair_data.py
# which inherits their Apache-2.0 license.


import json
import random
from typing import Dict

from transformers import AutoTokenizer

from codefavor.template import pairwise_classification_template, pairwise_cot_template
from codefavor.utility import make_progress
from datasets import Dataset, DatasetDict


def make_judge_data(example: Dict, judge_type: str):
    codes = [example["improved_code"].strip(), example["naive_code"].strip()]

    data_in_both_order = []
    for chosen_position in [0, 1]:
        code_a = codes[chosen_position]
        code_b = codes[1 - chosen_position]

        if judge_type == "classification":
            prompt = pairwise_classification_template(
                example["instruction"], code_a, code_b, example["criteria"]
            )
            response = ["A", "B"][chosen_position]
        elif judge_type == "cot":
            prompt = pairwise_cot_template(
                example["instruction"], code_a, code_b, example["criteria"]
            )
            naive_code = ["[CODE_A]", "[CODE_B]"][1 - chosen_position]
            improved_code = ["[CODE_A]", "[CODE_B]"][chosen_position]
            assert "[RESULT]" not in example["feedback"], example["feedback"]
            response = (
                (
                    example["feedback"]
                    .replace("[NAIVE_CODE]", naive_code)
                    .replace("[IMPROVED_CODE]", improved_code)
                    .replace("initial attempt", naive_code)
                    .replace("improved code", improved_code)
                )
                + f"\n[RESULT]\nTherefore, {improved_code} is better than {naive_code} on [CRITERIA]."
            )

            if naive_code not in response and improved_code not in response:
                print("\n\n---")
                print("Feedback contains code indentifier:")
                print(example["feedback"])
                continue
        else:
            raise ValueError(f"Unknown judge type: {judge_type}")

        item = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        }
        if "seed_info" in example:
            item["commit"] = example["seed_info"]["commit"]
        data_in_both_order.append(item)
    return data_in_both_order


def main(
    decomposed_dataset: str,
    judge_type: str,
    both_order: bool = False,
    tokenizer: str = None,
):
    random.seed(66666)

    data = [json.loads(l) for l in open(decomposed_dataset, "r") if l.strip()]
    print("Received", len(data), "decomposed data")

    with make_progress() as p:
        data = [make_judge_data(d, judge_type=judge_type) for d in p.track(data)]

    hf_train_list = []
    for item in data:
        if not both_order:
            item = [random.choice(item)]
        hf_train_list.extend([{"messages": it["messages"]} for it in item])

    # tokenizer status
    if tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        num_tokens = sorted(
            [
                len(tokenizer.apply_chat_template(item["messages"]))
                for item in hf_train_list
            ]
        )
        print(
            f"Tokenization statistics: min@ {min(num_tokens)}, "
            f"max@ {max(num_tokens)}, "
            f"avg@ {sum(num_tokens) / len(num_tokens) :.1f}, "
            f"medium@ {num_tokens[len(num_tokens) // 2]}, "
            f"p80@ {num_tokens[int(len(num_tokens) * 0.8)]:.1f}, "
            f"p90@ {num_tokens[int(len(num_tokens) * 0.9)]:.1f}, "
            f"p95@ {num_tokens[int(len(num_tokens) * 0.95)]:.1f}, "
        )

    # shuffle the train set
    random.shuffle(hf_train_list)

    # Compose both to a HuggingFace dataset
    hf_dataset = DatasetDict({"train": Dataset.from_list(hf_train_list)})

    print(hf_dataset)

    target_path = decomposed_dataset.replace("decompose.jsonl", f"axolotl.{judge_type}")
    if both_order:
        target_path += ".both_order"
    hf_dataset.save_to_disk(target_path)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
