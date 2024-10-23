# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: CC-BY-NC-4.0

# Inspect decomposed data
import json
import random
from typing import List

from IPython import embed
from rich import print
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table


def viz(example):
    print(f"[bold]Instruction[/bold]:")
    print(example["instruction"])
    print(f"[bold]Attempt 1[/bold]:")
    # print as markdown
    print(Syntax(example["naive_code"], "markdown"))
    print(f"[bold]Reflection[/bold]:")
    print(example["reflection"])
    if "criteria" in example:
        print(f"[bold]Criteria[/bold]:")
        print(example["criteria"])
        print(f"[bold]Attempt 2[/bold]:")
        # print as markdown
        print(Syntax(example["improved_code"], "markdown"))
        print(f"[bold]Feedback[/bold]:")
        print(example["feedback"])


def make_response_panels(example):
    ret = []
    if "criteria" in example:
        ret = [
            Panel(f"[green]{example['criteria'].strip()}", title="Criteria"),
            Panel(Syntax(example["improved_code"], "markdown"), title="Attempt 2"),
            Panel(f"[green]{example['feedback'].strip()}", title="Feedback"),
        ]
    else:
        ret = [
            Panel("No criteria", title="Criteria"),
            Panel("No attempt 2", title="Attempt 2"),
            Panel("No feedback", title="Feedback"),
        ]
    if "reflection" in example:
        ret.insert(0, Panel(example["reflection"], title="Reflection"))
    return ret


def make_basic_panels(example):
    return [
        f"[yellow][bold]Instruction[/bold]\n{example['instruction']}\n",
        Syntax(example["naive_code"], "markdown"),
    ]


def viz_md5(md5, datasets: dict):
    paths = list(datasets.keys())
    datasets = list(datasets.values())
    [print(p) for p in make_basic_panels(datasets[0][md5])]

    panel_sets = [make_response_panels(dataset[md5]) for dataset in datasets]

    table = Table()
    [table.add_column(path) for path in paths]
    [table.add_row(*panels) for panels in zip(*panel_sets)]
    print(table)


def main(datasets: List[str]):
    print(datasets)
    parsed_datasets = {}
    inst_md5 = []
    for path in datasets:
        assert path.endswith(".decompose.jsonl")
        with open(path, "r") as f:
            data = [json.loads(line) for line in f.readlines() if line]
            inst_md5.append([d["inst_md5"] for d in data])
            parsed_datasets[path] = {d["inst_md5"]: d for d in data}
            print(f"{path}: # {len(data)} data")
    datasets = parsed_datasets

    inst_md5 = list(set(inst_md5[0]).intersection(*inst_md5[1:]))
    inst_md5.sort()

    def sample():
        ri = len(inst_md5) - 1
        print(f"viz_md5(inst_md5[{ri}], datasets)")
        viz_md5(inst_md5[random.randint(0, ri)], datasets)

    print("Sample results by: sample()")
    embed()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+")
    args = parser.parse_args()

    main(args.datasets)
