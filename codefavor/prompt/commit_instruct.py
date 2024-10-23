# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: CC-BY-NC-4.0


import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from traceback import print_exc
from typing import Any, Dict, List, Optional

from codefavor.data import prepare_commitpackft, prepare_editpackft
from codefavor.data.utility import diff
from codefavor.prompt.utility import ChatTermination
from codefavor.provider import BaseCaller
from codefavor.provider.bedrock import BedrockCaller
from codefavor.provider.openai import OpenAICaller
from codefavor.utility import make_progress


def _round1_invoke_reasoning(message, old_code, new_code, language=""):
    return f"""\
Given a code commit below, think about the code change:

Commit message: {message}

[OLD_CODE]
```{language.lower()}
{old_code}
```

[NEW_CODE]
```{language.lower()}
{new_code}
```

Please briefly explain the code change.
"""


def _round1_invoke_reasoning_diff(message, old_code, new_code, language=""):
    return f"""\
Given a code commit below, think about the code change:

Commit message: {message}

[OLD_CODE]
```{language.lower()}
{old_code}
```

[CODE_DIFF]
```
{diff(old_code, new_code)}
```

BRIEFLY reason about the pro & cons of the change:
* Is the code change educational? Does the code change bring CLEAR improvement on some good code aspects?
* Can the improvement be reasoned within the context?
"""


def _round2_quality_check():
    return r"""Directly answer [YES] or [NO]:
* If [YES], it clearly improves the [some good properties, e.g., functionality/performance/completeness/safety/...]
* If [NO], this minor change does not clearly make the code better.
"""


def _round3_data_construct():
    return """Inspired by the commit and explanation, please construct an instruction-following data with following components:

[INSTRUCTION]
A natural-language description to describe the goal and requirement of the code.

[CRITERIA]
A brief and focused criterion that the code should ideally meet, which are not necessarily implied in [INSTRUCTION].

[NAIVE_CODE]
```
A self-contained solution code that may NOT completely meet [CRITERIA].
```

[IMPROVED_CODE]
```
Improved code that better meets [CRITERIA] than [NAIVE_CODE], while still respecting [INSTRUCTION].
```

[FEEDBACK]
Briefly describe why the [IMPROVED_CODE] is better than [NAIVE_CODE]?
Refer to the codes using "[IMPROVED_CODE]" and "[NAIVE_CODE]" only.
"""


USER_QUERIES = [
    _round1_invoke_reasoning_diff,  # invoke reasoning
    _round2_quality_check,  # check improvement
    _round3_data_construct,  # data construction
]


def commit_instruct(
    history: List[Dict[str, str]], message, old_code, new_code, language
) -> Optional[List[Dict[str, str]]]:
    assert len(history) == 0 or history[-1]["role"] in ["assistant", "system"]
    if len(history) <= 1:
        return {
            "role": "user",
            "content": _round1_invoke_reasoning(message, old_code, new_code, language),
        }
    elif len(history) in [2, 3]:
        return {
            "role": "user",
            "content": _round2_quality_check(),
        }
    elif len(history) in [4, 5]:
        response = history[-1]["content"]
        # Check response validity
        if "YES" not in response and "NO" not in response:
            raise ChatTermination(
                reason=f"'YES' or 'NO' not found in response: {response}"
            )
        if "NO" in response:
            raise ChatTermination(
                reason=f"LLM thinks the response is not interesting: {response}"
            )
        return {
            "role": "user",
            "content": _round3_data_construct(),
        }
    else:
        return None  # None means done


SYSTEM_PROMPT = "You are a helpful code assistant good at analyzing and generating high-quality code."


def commit_instruct_loop(
    caller_type, caller_kwargs: Dict[str, Any], row: Dict[str, str]
) -> Dict[str, str]:
    required_fields = ["commit", "old_contents", "new_contents", "message", "lang"]
    row = {k: v for k, v in row.items() if k in required_fields}

    caller: BaseCaller = caller_type(**caller_kwargs)
    messages = []
    note = None

    try:
        while True:
            new_query = commit_instruct(
                messages,
                row["message"],
                row["old_contents"],
                row["new_contents"],
                row["lang"],
            )
            if new_query is None:
                note = "success"
                break  # Done
            messages.append(new_query)
            messages.append(
                {
                    "role": "assistant",
                    "content": caller(messages=messages, eos=["NO"]),
                }
            )
    except ChatTermination as e:
        note = e.reason

    # schema of return
    return {
        "seed_info": row,
        "timestamp": datetime.now().isoformat(),
        "model_id": caller.model_id,
        "model_type": caller.model_type,
        "conversations": messages,
        "note": note,  # success | fail-reason:...
    }


def main(
    dataset="editpackft",
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    model_type: str = "bedrock",
    model_url: Optional[str] = None,
    temperature: float = 0.8,
    concurrency: int = 8,
):
    target_path: str = (
        f"{dataset}-commit-instruct-raw-{model_id.replace('/', '--')}.jsonl"
    )
    if dataset == "editpackft":
        dataset = prepare_editpackft()
    elif dataset == "commitpackft":
        dataset = prepare_commitpackft()

    caller_type = None
    if model_type == "bedrock":
        caller_type = BedrockCaller
    elif model_type == "openai":
        caller_type = OpenAICaller
    else:
        raise RuntimeError(f"Unknown model type: {model_type}")
    caller_args = {
        "model_id": model_id,
        "model_url": model_url,
        "temperature": temperature,
    }

    done_commits = []
    if os.path.isfile(target_path):
        with open(target_path, "r") as f:
            # each line is a json
            done_commits = [
                json.loads(line)["seed_info"]["commit"]
                for line in f.readlines()
                if line
            ]
            print(f"Found {len(done_commits)} finished commits.")

    with open(target_path, "+a") as f:
        for lang, rows in dataset.items():
            print(f"Language: {lang}")
            print(f"Number of total samples: {len(rows)}")
            rows = [row for row in rows if row["commit"] not in done_commits]
            print(f"Number of unfinished samples: {len(rows)}")

            # emit futures
            with ProcessPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(commit_instruct_loop, caller_type, caller_args, row)
                    for row in rows
                ]
                with make_progress() as p:
                    for r in p.track(as_completed(futures), total=len(futures)):
                        try:
                            f.write(json.dumps(r.result()) + "\n")
                        except:
                            print_exc()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
