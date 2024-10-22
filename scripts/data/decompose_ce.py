# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: Apache-2.0

# Decompose the raw response in Active Code Evolution
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor

from evalplus.data.utils import stream_jsonl
from evalplus.sanitize import sanitize
from rich.syntax import Syntax

from codefavor.prompt.critic_evol import FEWSHOT_SPLITTER, datagen_inference
from codefavor.utility import make_progress, remove_comments

query_sections = ["INSTRUCTION", "ATTEMPT_1"]
response_sections = ["REFLECTION", "CRITERIA", "ATTEMPT_2", "FEEDBACK"]


def compute_md5_str(inst: str) -> str:
    return hashlib.md5(inst.encode()).hexdigest()


def parse_string_to_dict(input_string, sections):
    result = {}
    for section in sections:
        pattern = f"-- BEGIN \\[{section}\\] --\n(.*?)-- END \\[{section}\\] --"
        match = re.search(pattern, input_string, re.DOTALL)
        if match:
            result[section.lower()] = match.group(1).strip()
    return result


def process_raw_response(
    example: dict, weak_item: dict = None, remove_comment: bool = False
) -> dict:
    if example["prompt"] is not None:
        query = example["prompt"].split(FEWSHOT_SPLITTER)[-1].strip()
    else:
        query = datagen_inference(
            weak_item["instruction"], sanitize(weak_item["response"])
        )

    query_dict = parse_string_to_dict(query, query_sections)
    if "instruction" not in query_dict or "attempt_1" not in query_dict:
        return None

    response = example["response"].strip()
    response_dict = parse_string_to_dict(response, response_sections)
    if not all(
        [
            k in response_dict
            for k in ["reflection", "criteria", "attempt_2", "feedback"]
        ]
    ):
        return None

    if (
        "feedback" in response_dict
        and "[ATTEMPT_1]" not in response_dict["feedback"]
        and "[ATTEMPT_2]" not in response_dict["feedback"]
    ):
        return None

    # response-1
    santized_code = sanitize(
        code=query_dict["attempt_1"].lstrip("```python").rstrip("```").strip()
    ).strip()
    if santized_code and remove_comment:
        try:
            santized_code = remove_comments(santized_code)
        except:
            pass
        else:
            response_dict["attempt_1"] = f"```python\n{santized_code}\n```"

    # response-2
    santized_code = sanitize(
        code=response_dict["attempt_2"].lstrip("```python").rstrip("```").strip()
    ).strip()
    if not santized_code:
        return None
    if remove_comment:
        try:
            santized_code = remove_comments(santized_code)
        except:
            return None
        else:
            response_dict["attempt_2"] = f"```python\n{santized_code}\n```"

    return {
        "inst_md5": example["inst_md5"],
        **query_dict,
        **response_dict,
    }


def viz(example):
    print(f"[bold]Instruction[/bold]:")
    print(example["instruction"])
    print(f"[bold]Attempt 1[/bold]:")
    # print as markdown
    print(Syntax(example["attempt_1"], "markdown"))
    print(f"[bold]Reflection[/bold]:")
    print(example["reflection"])
    if "criteria" in example:
        print(f"[bold]Criteria[/bold]:")
        print(example["criteria"])
        print(f"[bold]Attempt 2[/bold]:")
        # print as markdown
        print(Syntax(example["attempt_2"], "markdown"))
        print(f"[bold]Feedback[/bold]:")
        print(example["feedback"])


def main(
    raw_dataset: str,
    inspect: bool = False,
    weak_dataset: str = None,
    remove_comment: bool = False,
):
    assert raw_dataset.endswith(".jsonl")
    if remove_comment:
        target_path = raw_dataset.replace(".jsonl", ".decompose.jsonl")
    else:
        target_path = raw_dataset.replace(".jsonl", ".withcmt.decompose.jsonl")

    if weak_dataset:
        weak_dataset = {
            compute_md5_str(d["instruction"]): d for d in stream_jsonl(weak_dataset)
        }

    data = []
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = []
        for example in stream_jsonl(raw_dataset):
            weak_item = None
            if weak_dataset:
                weak_item = weak_dataset[example["inst_md5"]]

            futures.append(
                executor.submit(
                    process_raw_response, example, weak_item, remove_comment
                )
            )

        with make_progress() as pbar:
            for future in pbar.track(futures, total=len(futures)):
                item = future.result()
                if item:
                    data.append(item)

    print(f"Total {len(data)} examples")

    evol_data = data
    if inspect:
        from IPython import embed

        embed()

    with open(target_path, "w") as f:
        for eval_item in evol_data:
            # update key: "attempt_1" -> "naive_code"
            eval_item["naive_code"] = eval_item.pop("attempt_1")
            # update key: "attempt_2" -> "improved_code"
            eval_item["improved_code"] = eval_item.pop("attempt_2")
            eval_item["feedback"] = (
                eval_item["feedback"]
                .replace("[ATTEMPT_1]", "[NAIVE_CODE]")
                .replace("[ATTEMPT_2]", "[IMPROVED_CODE]")
            )
            f.write(json.dumps(eval_item) + "\n")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
