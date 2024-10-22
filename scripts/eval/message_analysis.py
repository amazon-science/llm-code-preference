# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import json
from statistics import mean
from typing import List

from transformers import AutoTokenizer


def main(args):
    print(f"Model ID: {args.tokenizer}")
    files: List[str] = args.files
    tokenizer: str = AutoTokenizer.from_pretrained(args.tokenizer)

    iword_cnt = []
    oword_cnt = []
    itken_cnt = []
    otken_cnt = []

    for file in files:
        with open(file, "r") as f:
            # jsonl
            data = [json.loads(line) for line in f]

        for item in data:
            conversations = item["conversations"]
            for conversation in conversations:
                if conversation["role"] == "user":
                    iword_cnt.append(len(conversation["content"].split()))
                    itken_cnt.append(
                        len(tokenizer(conversation["content"])["input_ids"])
                    )
                else:
                    oword_cnt.append(len(conversation["content"].split()))
                    otken_cnt.append(
                        len(tokenizer(conversation["content"])["input_ids"])
                    )

    print("=" * 16)
    print(f"Input words: {sum(iword_cnt)} = {mean(iword_cnt):.1f} * {len(iword_cnt)}")
    print(f"Output words: {sum(oword_cnt)} = {mean(oword_cnt):.1f} * {len(oword_cnt)}")
    print(f"Human cost: ${mean(iword_cnt) * 0.11 / 50:.2f} / Sample")
    print("=" * 16)
    print(f"Input tokens: {sum(itken_cnt)} = {mean(itken_cnt):.1f} * {len(itken_cnt)}")
    print(f"Output tokens: {sum(otken_cnt)} = {mean(otken_cnt):.1f} * {len(otken_cnt)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--tokenizer", type=str)

    args = parser.parse_args()
    main(args)
