# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_cpu_fp32_model(path: str) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)


def merge_models(models: list, weights: Optional[List] = None):
    assert len(models) >= 2
    weights = weights or [1] * len(models)
    weights = [w / sum(weights) for w in weights]

    print(f"Merging weights {weights} for models: {models}")

    model: Optional[AutoModelForCausalLM] = None
    merged_state_dict = {}
    for weight, model in zip(weights, models):
        model = _load_cpu_fp32_model(model)
        for key, value in model.state_dict().items():
            if key not in merged_state_dict:
                merged_state_dict[key] = value * weight
            else:
                merged_state_dict[key] += value * weight

    print("Merged state dict:")
    print(merged_state_dict)

    model.load_state_dict(merged_state_dict)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    args = parser.parse_args()

    weights = args.weights or [1] * len(args.models)
    model = merge_models(args.models, weights)

    model.save_pretrained(args.output)
    # also save the tokenizer
    AutoTokenizer.from_pretrained(args.models[0]).save_pretrained(args.output)
