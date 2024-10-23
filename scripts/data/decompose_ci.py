# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
from typing import Dict, List, Optional

from codefavor.utility import make_progress

# The order matters
CAPTURE_FIELDS = [
    "INSTRUCTION",
    "CRITERIA",
    "NAIVE_CODE",
    "IMPROVED_CODE",
    "FEEDBACK",
]


def data_filter(data):
    if data["note"] != "success":
        return False

    content = data["conversations"][-1]["content"].upper()
    return all(field in content for field in CAPTURE_FIELDS)


def parse_fields(raw_text) -> Optional[Dict[str, str]]:
    # data["conversations"][-1]["content"]
    lines = [l for l in raw_text.split("\n") if l.strip()]

    # capture splitter
    capture_line_ids = {}
    capture_idx = 0
    for i, line in enumerate(lines):
        if capture_idx >= len(CAPTURE_FIELDS):
            break
        prefix_cap = CAPTURE_FIELDS[capture_idx]
        if ("".join(char for char in line if char.isalpha())).startswith(
            prefix_cap.replace("_", "")
        ):
            capture_idx += 1
            capture_line_ids[prefix_cap] = i

    # return on incomplete parsing
    if len(capture_line_ids) != len(CAPTURE_FIELDS):
        return None

    capture2block = {}
    for i in range(len(CAPTURE_FIELDS)):
        capture_kw = CAPTURE_FIELDS[i]
        start = capture_line_ids[capture_kw]
        end = (
            capture_line_ids[CAPTURE_FIELDS[i + 1]]
            if i + 1 < len(CAPTURE_FIELDS)
            else None
        )
        clines: List[str] = lines[start:end]
        for kw in [
            f"**[{capture_kw}]**",
            f"[{capture_kw}]",
            f"**{capture_kw}**",
            capture_kw,
        ]:
            clines[0] = clines[0].lstrip(kw)
        capture2block[capture_kw] = "\n".join(clines).strip()

    return {k.lower(): v for k, v in capture2block.items()}


def pattern_replace(keyword: str, text: str):
    # keyword ~ "Improved Code"
    target_keyword = "[" + keyword.upper().replace(" ", "_") + "]"
    for k in [keyword, keyword.upper()]:
        for srcp in [
            f"[{k}]",
            f"**{k}**",
            f"**[{k}]**",
            f"**{k}:**",
        ]:
            text = text.replace(srcp, target_keyword)
    return text


def main(dataset: str):
    data = [json.loads(l) for l in open(dataset, "r") if l.strip()]
    data = list(filter(data_filter, data))
    target_path = dataset.replace(".jsonl", ".decompose.jsonl")
    print("Decomposing data to", target_path)
    with open(target_path, "w") as f:
        with make_progress() as p:
            for d in p.track(data):
                raw_text = d["conversations"][-1]["content"]
                raw_text = pattern_replace("Feedback", raw_text)
                raw_text = pattern_replace("Improved Code", raw_text)
                raw_text = pattern_replace("Naive Code", raw_text)
                raw_text = pattern_replace("Instruction", raw_text)
                raw_text = pattern_replace("Criteria", raw_text)
                raw_text = raw_text.replace("[Criteria]", "[CRITERIA]")  # fix typo
                fileds = parse_fields(raw_text)
                assert fileds, raw_text
                for k, v in fileds.items():
                    assert v, f"Empty field: {k}"
                f.write(
                    json.dumps(
                        {
                            **{
                                k: v
                                for k, v in d.items()
                                if k
                                in [
                                    "seed_info",
                                    "model_type",
                                    "model_id",
                                    "timestamp",
                                    "note",
                                ]
                            },
                            **fileds,
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
