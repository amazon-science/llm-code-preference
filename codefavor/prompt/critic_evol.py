# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from traceback import print_exc

from evalplus.data.utils import stream_jsonl
from evalplus.sanitize import sanitize
from fire import Fire

from codefavor.provider.bedrock import BedrockCaller
from codefavor.provider.openai import OpenAICaller
from codefavor.utility import make_progress

# Some issues in the current versions
# -- bias in a way that the model tend to always improve the code (impacted by the few shot examples maybe)

LAST_INTRUCTION = """\
You are given [INSTRUCTION] and [ATTEMPT_1], in response you generate:

1. A [REFLECTION] section analyzing noticable weaknesses of [ATTEMPT_1] while following [INSTRUCTION]
2. If you think [ATTEMPT_1] is good enough without significant space for improvements, stop the generation after [REFLECTION]
3. Otherwise, keep going with [CRITERIA], [ATTEMPT_2], and [FEEDBACK]

Notes:
1. [CRITERIA] should focus on one significant code weakness over correctness/efficiency/security/conciseness -- DO NOT worry about trivial pitfalls like missing type hints, docstrings, input validation, etc.
2. In [FEEDBACK], only refer to the code attempts using [ATTEMPT_1] and [ATTEMPT_2]
"""

FEW_SHOT_EXAMPLES = [
    {
        "instruction": "Add two numbers and return the sum using Python.",
        "attempt1": """\
def add(x, y):

    return x + y""",
        "reflection": """`x + y` performs the behavior of "add two numbers"t. \
The straightforward implementation of single statement is good enough and does not need further improvements.""",
    },
    {
        "instruction": "Provide a Python function `square_root` to compute the square root of a number and throw a ValueError if the number is negative.",
        "attempt1": """\
def square_root(x: float) -> float:
    return math.sqrt(x)
""",
        "reflection": """[ATTEMPT_1] uses `math.sqrt` without importing the `math` module which can lead to a NameError during execution. \
The bug can be fixed by importing the `math` module.""",
        "criteria": "The function should precisely follow the instruction while being runnable and bug-free.",
        "attempt2": """\
import math

def square_root(x):
    return math.sqrt(x)""",
        "feedback": "[ATTEMPT_1] forgets to import a necessary module. [ATTEMPT_2] fixes the bug in [ATTEMPT_1] by importing the `math` module ahead of time.",
    },
    {
        "instruction": """\
Create a Python function to format a file size in bytes into a human-readable string representation, \
using 'bytes', 'KB' (kilobytes), 'MB' (megabytes), 'GB' (gigabytes), or 'TB' (terabytes) units. \
The output should be rounded to one decimal place and follow the format: "{X.Y}{Unit}", \
where "Unit" is the appropriate size unit and "X.Y" is the rounded size. \
For example, `format_size(1024 * 512)` should return `'512.0KB'`, and `format_size(1024**2 - 1)` should return `'1024.0KB'`.""",
        "attempt1": r"""def format_size(size: int) -> str:
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0 or unit == 'TB':
            return f"{size:.1f}{unit}"
        size /= 1024
""",
        "reflection": """\
[ATTEMPT_1] is correct. Meanwhile, it is possible to slightly optimize it by converting float division to integer division.
The efficiency can also be further improved by treating "bytes" and "TB" as special cases to avoid unnecessary division.""",
        "criteria": "The code with better execution efficiency is preferred.",
        "attempt2": r"""def format_size(size: int) -> str:
    if size < 1024:
        return f"{size}bytes"

    for unit in ['KB', 'MB', 'GB']:
        if size < 1048576:
            return f"{size/1024:.1f}{unit}"
        size >>= 10

    return f"{size/1024:.1f}TB"
""",
        "feedback": """[ATTEMPT_1] uses float divison and [ATTEMPT_2] uses integer division. \
In general, integer division is more efficient than float division. Specifically, integer division used in [ATTEMPT_2] \
uses a base of 1024 which provides sufficient precision to the desired format.""",
    },
    {
        "instruction": r"""Write a function that takes a dictionary as an argument and returns a formatted multi-line string, where each line contains a key-value pair in the format "key: value". The function should use string formatting
with concatenation to construct the formatted string, and it should traverse the dictionary using dictionary lookups. You can assume the dictionary only contains string keys and values.

For example, given the dictionary {"name": "Alice", "age": 25, "city": "New York"}, the function should return a string that looks like this:

```
assert dict_to_string({"name": "Alice", "age": 25, "city": "New York"}) == "name: Alice\nage: 25\ncity: New York"
```
""",
        "attempt1": r"""def dict_to_string(d: dict) -> str:
    result = ""
    for key, value in d.items():
        result += f"{key}: {value}\n"
    return result.strip()
""",
        "reflection": """[ATTEMPT_1] iterates key-value pairs in the dictionary. \
In each iteration, a pair is presented using "key: value" which is concatenated to the result string. \
From the example, newline should not be added in the end and [ATTEMPT_1] resolves it by using `strip()`. \
The code is correct. Yet, we can optimize it into a one-liner to be more Pythonic.""",
        "criteria": "The desired code should be correct, efficient, and Pythonic.",
        "attempt2": r"""def dict_to_string(d: dict) -> str:
    return "\n".join([f"{key}: {value}" for key, value in d.items()])
""",
        "feedback": """Both implementation follow the correct format and do not introduce dangling newliners. \
Yet, [ATTEMPT_2] is more Pythonic and concise than [ATTEMPT_1].""",
    },
    {
        "instruction": """\
Implement a Python function `parse_primary_data` that parses a string representation of a data and returns an object representing that data. \
The function should support common Python-native data types, including numerics, lists, dictionaries, and tuples. \
The syntax of the string representation follows the Python literal syntax.

For instance, given the string representation of a list of tuples, the function should return a list of tuples, as shown in the following example:

```python
>>> parse_primary_data('[(1, 2, 3), (4, 5, 6)]')
[
    (1, 2, 3),
    (4, 5, 6)
]
```
""",
        "attempt1": """\
def parse_primary_data(data_type_str: str):
    return eval(data_type_str)""",
        "reflection": """[ATTEMPT_1] uses `eval` to directly evaluate the string representation. Using `eval()` is dangerous especially there is no input validation. \
For security, it is better to use `ast.literal_eval()` which only parse literals and does not execute arbitrary code.
""",
        "criteria": "While implementing the instruction, the code should be as secure as possible.",
        "attempt2": """\
import ast

def parse_primary_data(data_str: str):
    return ast.literal_eval(data_str)""",
        "feedback": """While both codes implement the functionality of the instruction, [ATTEMPT_1] uses `eval()` without sufficient input validation, \
leading to the security vulnerability of arbitrary code execution (ACE). [ATTEMPT_2] uses `ast.literal_eval()` which is safer as it only tries to parse the literals, disallowing ACE.""",
    },
]


bos = lambda k: f"-- BEGIN {k} --"
eos = lambda k: f"-- END {k} --"


def datagen_inference(instruction: str, attempt1: str):
    assert attempt1 is not None
    return f"""\
Follow the following instruction to write a Python function:
{bos('[INSTRUCTION]')}
{instruction.strip()}
{eos('[INSTRUCTION]')}


Initial attempt to implement the function:
{bos('[ATTEMPT_1]')}
```python
{attempt1.strip()}
```
{eos('[ATTEMPT_1]')}
"""


def fewshot_template_improve(
    instruction: str,
    attempt1: str,
    reflection: str,
    criteria: str,
    attempt2: str,
    feedback: str,
):
    return f"""\
{datagen_inference(instruction, attempt1)}

Do you see APPARENT bugs, inefficiencies, security vulnerabilities, or inconciseness in [ATTEMPT_1] when following the [INSTRUCTION]?
{bos('[REFLECTION]')}
{reflection.strip()}
{eos('[REFLECTION]')}

A SIMPLE criteria where [ATTEMP_1] can be improved from [REFLECTION]:
{bos('[CRITERIA]')}
{criteria.strip()}
{eos('[CRITERIA]')}

The improved version of [ATTEMPT_1] based on the [CRITERIA] and [REFLECTION]:
{bos('[ATTEMPT_2]')}
```python
{attempt2.strip()}
```
{eos('[ATTEMPT_2]')}

How does [ATTEMPT_2] improve over [ATTEMPT_1]?
{bos('[FEEDBACK]')}
{feedback.strip()}
{eos('[FEEDBACK]')}
"""


def fewshot_template_no_improve(instruction: str, attempt1: str, reflection: str):
    return f"""\
{datagen_inference(instruction, attempt1)}

Do you see clear bugs, inefficiencies, security vulnerabilities, or inconciseness in [ATTEMPT_1] when following the [INSTRUCTION]?
{bos('[REFLECTION]')}
{reflection.strip()}
{eos('[REFLECTION]')}
"""


def fewshot_template(example):
    if "feedback" in example:
        return fewshot_template_improve(**example)
    return fewshot_template_no_improve(**example)


FEWSHOT_SPLITTER = "\n" + "-" * 8 + "\n"


def _critic_evol_fewshot(seed):
    random.seed(seed)
    assert len(FEW_SHOT_EXAMPLES) > 2
    return (
        """You are a great Python coding instructor good at judging code snippets, localizing code faults, and providing educational feedbacks.

Please follow the formats of these examples to provide necessary code feedbacks:"""
        + FEWSHOT_SPLITTER
        + "\n\n".join(
            fewshot_template(example)
            for example in random.sample(FEW_SHOT_EXAMPLES, len(FEW_SHOT_EXAMPLES))
        )
        + FEWSHOT_SPLITTER
        + LAST_INTRUCTION
    )


def construct_critic_evol_prompt(instruction, attempt1, seed: int = 0) -> str:
    return _critic_evol_fewshot(seed) + "\n" + datagen_inference(instruction, attempt1)


def make_one_call(
    input, model_id, model_url, seed: int = 0, no_save_prompt: bool = False
) -> dict:
    temperature = 0.5
    if "." in model_id:
        caller = BedrockCaller(model_id=model_id, temperature=temperature)
    else:
        caller = OpenAICaller(
            model_id=model_id, temperature=temperature, model_url=model_url
        )
    (instruction, attempt1), inst_md5 = input
    prompt = construct_critic_evol_prompt(instruction, attempt1, seed)
    return {
        "inst_md5": inst_md5,
        "prompt": None if no_save_prompt else prompt,
        "response": caller.call(
            [{"role": "user", "content": prompt}], max_new_tokens=2048
        ),
    }


def process_item(item):
    attempt1 = sanitize(item["response"])
    if not attempt1:
        return None
    return (item["instruction"], attempt1), hashlib.md5(
        item["instruction"].encode()
    ).hexdigest()


def _make_one_call(args):
    return make_one_call(*args)


def main(
    weak_dataset: str,
    model_id: str,
    concurrency: int = 256,
    model_url: str = None,
    no_save_prompt: bool = False,
):
    assert weak_dataset.endswith(".jsonl")
    model_id_sim = model_id.split("/")[-1].split(":")[0].split(".")[-1]
    target_path = weak_dataset.replace(
        ".jsonl", f".teacher.{model_id_sim}.critic_evol.jsonl"
    )
    print(f"Writing to {target_path}")
    done_inst_md5 = frozenset()
    if os.path.isfile(target_path):
        with open(target_path, "r") as f:
            done_inst_md5 = frozenset([json.loads(line)["inst_md5"] for line in f])

    # read jsonl
    print(f"Computing unfinished tasks from {weak_dataset}...")
    todo_items = [
        item
        for item in stream_jsonl(weak_dataset)
        if hashlib.md5(item["instruction"].encode()).hexdigest() not in done_inst_md5
    ]

    print(f"Code sanitization for {len(todo_items)} samples...")
    with ProcessPoolExecutor(max_workers=16) as executor:
        with make_progress() as pbar:
            inputs = [
                item
                for item in pbar.track(
                    executor.map(process_item, todo_items, chunksize=16),
                    total=len(todo_items),
                )
                if item is not None
            ]

    # load params
    inputs = [
        (input, model_id, model_url, i, no_save_prompt)
        for i, input in enumerate(inputs)
    ]
    print(f"Total {len(inputs)} samples left to process.")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        print(f"Loading {len(inputs)} items")
        with make_progress() as pbar:
            futures = [
                executor.submit(_make_one_call, input)
                for input in pbar.track(inputs, total=len(inputs))
            ]
        print("Working progressing...")
        with open(target_path, "a") as f:
            with make_progress() as pbar:
                for future in pbar.track(as_completed(futures), total=len(futures)):
                    try:
                        item = future.result()
                        f.write(json.dumps(item) + "\n")
                    except:
                        print_exc()


if __name__ == "__main__":
    Fire(main)
