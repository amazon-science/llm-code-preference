# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
This file evaluates LLMs judges for efficiency alignements.
"""

import hashlib
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from traceback import print_exc
from typing import Dict, List, Optional, Union

from evalplus.data.utils import stream_jsonl
from rich import print

from codefavor.provider import BaseCaller
from codefavor.provider.bedrock import BedrockCaller
from codefavor.provider.google import GoogleCaller
from codefavor.provider.hf import HuggingFaceCaller, PairwiseRMCaller
from codefavor.provider.openai import OpenAICaller
from codefavor.template import (
    CORRECTNESS_CRITERIA,
    EFFICIENCY_CRITIERIA,
    HUMAN_PREF_CRITERIA,
    SECURITY_CRITERIA,
    pairwise_classification_template,
    pairwise_cot_template,
)
from codefavor.utility import make_progress, remove_comments


def _skip_broken_evalperf_code(code):
    """Via human inspection, some code evalperf code is not well processed so we skip them"""
    patterns = [
        # x2
        "flattened_list = list(flatten(lst_of_lst))",
        # x5
        "def count_Primes_nums(n):\n    primes = [2]\n    x = 3\n    if n < 2:\n        return 0\n    while x <= n:\n        for y in range(3, x, 2):\n            if x % y == 0:\n                x += 2\n                break\n        else:\n            primes.append(x)\n            x += 2\n    None\n    return len(primes)",
        # x3
        "heapify(min_heap, len(min_heap), 0)",  # the dependency is not parsed by "sanitize" so we skip
    ]
    for p in patterns:
        if p in code:
            return True
    return False


def _code_wrapper(code, lang="python"):
    return f"```{lang}\n{code}\n```"


def construct_judge_prompt(task, criteria, caller_type):
    caller = pairwise_cot_template
    if caller_type == "pair-rm":
        caller = pairwise_classification_template
    # instruct / code1 / code2 / criteria
    return caller(
        instruction=task["instruction"],
        code1=_code_wrapper(task["choices"][0]),
        code2=_code_wrapper(task["choices"][1]),
        criteria=criteria,
    )


def response_parser(response: str) -> int:
    # split by comment prefix word: RESULT | Result |
    sentences = re.split("|".join(map(re.escape, ["RESULT", "Result"])), response)[
        -1
    ].split(".")

    best_choice = None
    confidence_level = 0

    # reverse traverse
    for sentence in reversed(sentences):
        # remove non-alphanumeric characters
        sentence = "".join([x for x in sentence.upper() if x.isalnum()])
        # check neither
        if "NEITHER" in sentence:
            continue

        idx1 = idx2 = -1
        if "CODE1" in sentence or "CODE2" in sentence:
            idx1 = sentence.rfind("CODE1")
            idx2 = sentence.rfind("CODE2")
        elif "CODEA" in sentence or "CODEB" in sentence:
            idx1 = sentence.rfind("CODEA")
            idx2 = sentence.rfind("CODEB")

        if idx1 == -1 and idx2 == -1:  # meanless sentence
            continue

        if "BETTER" in sentence:
            idx_better = sentence.rfind("BETTER")
            # "X" is better than "Y" --> chooise X
            if idx1 == -1:
                cur_confidence_level = 1
                cur_decision = 1
            elif idx2 == -1:
                cur_confidence_level = 1
                cur_decision = 0
            else:
                if idx1 < idx_better < idx2:
                    cur_confidence_level = 2
                    cur_decision = 0
                elif idx2 < idx_better < idx1:
                    cur_confidence_level = 2
                    cur_decision = 1
                else:
                    cur_confidence_level = 1
                    cur_decision = 0 if idx1 < idx2 else 1

            if cur_confidence_level > confidence_level:
                confidence_level = cur_confidence_level
                best_choice = cur_decision

    # if best_choice is None:
    #     print(f"Undecidable response:\n{response}")
    return best_choice


def task_md5(task: Dict):
    return hashlib.md5(json.dumps(task, sort_keys=True).encode()).hexdigest()


def evaluate_messages(gt_choice: int, messages: List[Dict], caller_type: str):
    if caller_type == "pair-rm":
        # "A" -> 0, "B" -> 1
        decision = int(messages[-1]["content"] == "B")
    else:
        decision = response_parser(messages[-1]["content"])
    # schema of return
    return {
        "decision": decision,
        "classification": decision == gt_choice,
    }


def task_try_to_remove_comment(task, verbose=False):
    try:
        task["choices"] = [remove_comments(c) for c in task["choices"]]
    except:
        if verbose:
            print(f"Failed to remove comments in task: {task['task_id']}")
    return task


def task_eval(
    task: Dict,
    caller: Union[Dict, BaseCaller],
    criteria: Optional[str] = None,
    remove_comment=False,
) -> Dict[str, str]:
    if not isinstance(caller, BaseCaller):
        caller: BaseCaller = caller["caller_type"](**caller["caller_kwargs"])

    if criteria is None and "criteria" in task:
        criteria = task["criteria"]

    original_task_md5 = task_md5(task)  # use the original task to compute md5
    if remove_comment:
        task = task_try_to_remove_comment(task)

    messages = [
        {
            "role": "user",
            "content": construct_judge_prompt(task, criteria, caller.model_type),
        }
    ]
    messages.append(
        {
            "role": "assistant",
            "content": caller(messages=messages),
        }
    )

    result = {
        "task_md5": original_task_md5,
        "model_id": caller.model_id,
        "model_type": caller.model_type,
        "gt_choice": task["gt_choice"],
        "conversations": messages,
    }
    if "task_id" in task:
        result["task_id"] = task["task_id"]
    result.update(evaluate_messages(task["gt_choice"], messages, caller.model_type))
    return result


def length_bias_eval(
    task: Dict, preference: str, remove_comment=False
) -> Dict[str, str]:
    assert preference in ["shorter", "longer"]

    original_task_md5 = task_md5(task)  # use the original task to compute md5
    if remove_comment:
        task = task_try_to_remove_comment(task)

    if preference == "shorter":
        choose0 = len(task["choices"][0]) < len(task["choices"][1])
    else:
        choose0 = len(task["choices"][0]) > len(task["choices"][1])
    chosen_idx = int(1 - choose0)
    return {
        "task_md5": original_task_md5,
        "model_id": "length_bias",
        "model_type": "length_bias",
        "gt_choice": task["gt_choice"],
        "decision": chosen_idx,
        "classification": chosen_idx == task["gt_choice"],
    }


def evaluate_one_aspect(
    caller,
    tasks,
    criteria,
    model_type,
    model_id,
    concurrency,
    target_path,
    remove_comment: bool = True,
    reeval: bool = False,
):
    results = []
    done_tasks = defaultdict(int)
    # check not devnull
    if target_path != os.devnull and os.path.isfile(target_path):
        rewrite = []
        for data in stream_jsonl(target_path):
            if reeval:
                data.update(
                    evaluate_messages(
                        data["gt_choice"], data["conversations"], model_type
                    )
                )
                rewrite.append(data)
            results.append({"task_md5": data["task_md5"], "decision": data["decision"]})
            done_tasks[data["task_md5"]] += 1
        if rewrite:
            with open(target_path, "w") as f:
                for data in rewrite:
                    f.write(json.dumps(data) + "\n")

    # Running evaluation
    nsample = 1
    md52gt = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for task in tasks:
            md5 = task_md5(task)
            md52gt[md5] = task["gt_choice"]
            for _ in range(nsample - done_tasks[md5]):
                if model_type == "length_bias":
                    futures.append(
                        executor.submit(
                            length_bias_eval,
                            task,
                            preference=model_id,
                            remove_comment=remove_comment,
                        )
                    )
                else:
                    futures.append(
                        executor.submit(
                            task_eval,
                            task,
                            caller=caller,
                            criteria=criteria,
                            remove_comment=remove_comment,
                        )
                    )

        with make_progress() as p:
            with open(target_path, "+a") as f:
                for r in p.track(as_completed(futures), total=len(futures)):
                    try:
                        result = r.result()
                        results.append(
                            {
                                "task_md5": result["task_md5"],
                                "decision": result["decision"],
                            }
                        )
                        f.write(json.dumps(result) + "\n")
                    except:
                        print_exc()

    return results, md52gt


def prepare_eval_dataset(aspect):
    ##########################
    # Evaluation preparation #
    dataset_dir = os.path.join("datasets", "eval")
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, f"{aspect}-alignment.jsonl.gz")
    print("Dataset path:", dataset_path)

    dataset = None
    criteria = None  # non-fixed criteria
    if aspect == "efficiency":
        criteria = EFFICIENCY_CRITIERIA
    elif aspect == "correctness":
        criteria = CORRECTNESS_CRITERIA
    elif aspect == "security":
        criteria = SECURITY_CRITERIA
    elif aspect == "human-preference":
        criteria = HUMAN_PREF_CRITERIA
    else:
        raise RuntimeError(f"Unknown aspect: {aspect}")

    dataset = dataset or stream_jsonl(dataset_path)
    tasks = [t for t in dataset]

    return tasks, criteria


def evaluate(
    caller,
    aspect,
    model_type,
    model_id,
    evalroot,
    remove_comment: Optional[bool] = None,
    reeval: bool = False,
    concurrency: int = 8,
    overwrite: bool = False,
    dont_write: bool = False,
):
    # Make directory for evaluation results
    evaldir = os.path.join(evalroot, f"{aspect}-alignment")
    os.makedirs(evaldir, exist_ok=True)

    print("=" * 16)
    # Prepare the evaluation dataset
    tasks, criteria = prepare_eval_dataset(aspect)

    # Evaluation result path
    target_path = os.path.join(
        evaldir, f"{model_id.replace('/', '--').lstrip('.').strip('--')}.jsonl"
    )

    if remove_comment is None:
        remove_comment = "human-preference" not in aspect
    if remove_comment:
        target_path = target_path.replace(".jsonl", ".remove_comment.jsonl")

    if dont_write:
        print(f"Writing to {target_path} is disabled; redirecting to /dev/null")
        target_path = os.devnull
    elif overwrite and os.path.isfile(target_path):
        os.remove(target_path)

    # Resume the eval
    results, md52gt = evaluate_one_aspect(
        caller=caller,
        tasks=tasks,
        criteria=criteria,
        model_type=model_type,
        model_id=model_id,
        concurrency=concurrency,
        target_path=target_path,
        remove_comment=remove_comment,
        reeval=reeval,
    )

    # Compute accuracy
    ncorrect = 0
    undecided = 0
    ntotal = 0
    for item in results:
        md5, decision = item["task_md5"], item["decision"]
        if md5 not in md52gt:
            print(f"Unknown task cached: {md5}")
            continue
        ntotal += 1
        if decision is None:
            undecided += 1
        else:
            ncorrect += decision == md52gt[md5]

    rate_undecided = undecided / ntotal
    rate_sim_correct = (ncorrect + undecided / 2) / ntotal
    print(f"Aspect: {aspect}")
    print(
        f"Accuracy: {100 * rate_sim_correct:.1f}% +- {50 * rate_undecided:.1f}% -- this assumes half of the undecided are correct"
    )
    print(f"Results available at {target_path}")
    print("=" * 16)
    return rate_sim_correct


def main(
    model_id: str,
    model_type: str,
    # aspect: str,
    model_url: Optional[str] = None,
    evalroot: str = "evalroot",
    concurrency: int = 8,
    reeval: bool = False,
    remove_comment: Optional[bool] = None,
    overwrite: bool = False,
    dont_write: bool = False,
):
    # Initialize the caller
    caller_args = {
        "model_id": model_id,
        "model_url": model_url,
        "temperature": 0.0,
    }
    if model_type == "bedrock":
        caller = {"caller_type": BedrockCaller, "caller_kwargs": caller_args}
    elif model_type == "openai":
        caller = {"caller_type": OpenAICaller, "caller_kwargs": caller_args}
    elif model_type == "google":
        caller = {"caller_type": GoogleCaller, "caller_kwargs": caller_args}
    elif model_type == "pair-rm":
        caller = PairwiseRMCaller(**caller_args)
        concurrency = 1
    elif model_type == "huggingface":
        caller = HuggingFaceCaller(**caller_args)
        concurrency = 1
    elif model_type == "length_bias":
        assert model_id in ["shorter", "longer"]
        caller = None
        concurrency = 1
    else:
        raise RuntimeError(f"Unknown model type: {model_type}")

    tstart = time.time()
    os.makedirs(evalroot, exist_ok=True)
    scores = {}
    for aspect in ["correctness", "efficiency", "security", "human-preference"]:
        score = evaluate(
            caller,
            aspect,
            model_type,
            model_id,
            evalroot,
            remove_comment=remove_comment,
            reeval=reeval,
            concurrency=concurrency,
            overwrite=overwrite,
            dont_write=dont_write,
        )
        scores[aspect] = score
    print(
        f"Avg. score w/o human-pref: {100 * (sum(scores.values()) - scores.get('human-preference', 0)) / (len(scores) - int('human-preference' in scores)):.1f}%"
    )
    print(f"Avg. score: {100 * sum(scores.values()) / len(scores):.1f}%")
    print(f"Total time: {time.time() - tstart:.1f}s")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
