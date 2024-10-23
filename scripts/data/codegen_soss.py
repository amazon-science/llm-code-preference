# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import md5
from traceback import print_exc

from codefavor.provider.bedrock import BedrockCaller
from codefavor.provider.openai import OpenAICaller
from codefavor.utility import make_progress
from datasets import load_dataset


def codegen_soss(instruction, caller):
    caller = caller["caller_type"](caller["model_id"])
    response = caller.call(
        [{"role": "user", "content": instruction}], max_new_tokens=1024
    )
    return {"instruction": instruction, "response": response}


def main(model_id, model_type="openai"):
    if model_type == "bedrock":
        caller = {"caller_type": BedrockCaller, "model_id": model_id}
    elif model_type == "openai":
        caller = {"caller_type": OpenAICaller, "model_id": model_id}

    dataset = load_dataset(
        "bigcode/self-oss-instruct-sc2-exec-filter-50k", split="train"
    )

    done_instruction_md5 = set()
    model_id_sim = model_id.split("/")[-1].split(":")[0].split(".")[-1]
    target_path = f"{model_id_sim}-codegen_soss.jsonl"
    if os.path.isfile(target_path):
        with open(target_path, "r") as f:
            for line in f:
                item = json.loads(line)
                done_instruction_md5.add(md5(item["instruction"].encode()).hexdigest())

    dataset = dataset.filter(
        lambda x: md5(x["instruction"].encode()).hexdigest() not in done_instruction_md5
    )
    with open(target_path, "a") as f:
        with ThreadPoolExecutor(max_workers=256) as executor:
            print(f"Loading {len(dataset)} items")
            with make_progress() as pbar:
                futures = [
                    executor.submit(codegen_soss, item, caller)
                    for item in pbar.track(dataset["instruction"], total=len(dataset))
                ]
            print("Working progressing...")
            with make_progress() as pbar:
                for future in pbar.track(as_completed(futures), total=len(futures)):
                    try:
                        item = future.result()
                        f.write(json.dumps(item) + "\n")
                    except:
                        print_exc()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
