# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: CC-BY-NC-4.0

# This script is partially adapted from
# https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/pair-pm/process_pair_data.py
# which inherits their Apache-2.0 license.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(save_model_dir: str, model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.save_pretrained(save_model_dir)
    tokenizer.save_pretrained(save_model_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
