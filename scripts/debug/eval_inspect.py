# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import json
from collections import defaultdict

import rich
from IPython import embed
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


def viz(row):
    if "task_id" in row:
        print(f"Selected Data: {row['task_id']}")

    for item in row["conversations"]:
        print("-------", item["role"], "-------")
        print(item["content"])


def viz_rows(rows):
    table = Table()
    panels = []
    layers = list(zip(*[row["conversations"] for row in rows]))
    # copy easy mode
    print(layers[0][0]["content"])
    # syntax
    panels.append([Syntax(row["content"], "markdown") for row in layers[0]])
    # text
    panels.append([Panel(Text(row["content"]), title="bot") for row in layers[1]])

    decisions = []
    for i in range(len(rows)):
        data = layers[1][i]
        if "decision" not in rows[i]:
            print(data)
            assert (
                "codeA" in data["content"]
                or "codeB" in data["content"]
                or "Tie" in data["content"]
            )
            if "codeA" in data["content"]:
                decisions.append("A")
            elif "codeB" in data["content"]:
                decisions.append("B")
            else:
                decisions.append("Tie")
        else:
            decisions.append("B" if rows[i]["decision"] else "A")

    panels.append([Panel(Text(d), title="bot") for d in decisions])

    [table.add_column(f"{i+1}") for i in range(len(panels))]
    [table.add_row(*row) for row in (panels)]
    rich.print(table)

    print("Ground-truth answer:", "CODE_B" if rows[0]["gt_choice"] else "CODE_A")


def load_dataset(path: str):
    # .jsonl
    with open(path, "r") as f:
        data = [json.loads(line) for line in f.readlines() if line]

    print("=" * 16)
    print("=" * 16)
    print(f"{data[0]['model_type']} :: {data[0]['model_id']}")

    yes_data = []
    no_data = []
    for row in data:
        if row["classification"]:
            yes_data.append(row)
        else:
            no_data.append(row)

    # Check the overall rate of HQ commits
    rich.print(f"# Total = {len(data)}")
    rich.print(f"# Passing = {len(yes_data)} :: {len(yes_data)/len(data)*100:.1f}%")
    rich.print(f"# Failing = {len(no_data)}:: {len(no_data)/len(data)*100:.1f}%")
    print("-" * 16)

    return data, yes_data, no_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+")
    args = parser.parse_args()

    results = defaultdict(list)
    for dataset in args.datasets:
        data, _, _ = load_dataset(dataset)
        for d in data:
            results[d["task_md5"]].append(d)

    disagreements = []
    for md5, rows in results.items():
        # assert the same gt_choices
        assert len(set(r["gt_choice"] for r in rows)) == 1, f"GT choice mismatch: {md5}"
        if len(set(r["classification"] for r in rows)) > 1:
            disagreements.append(md5)

    def viz_md5(md5):
        rows = results[md5]
        viz_rows(rows)

    rich.print(f"{len(disagreements)}/{len(results)} disagreements: {disagreements}")
    rich.print(f"Example: viz_md5('{disagreements[0]}')")
    embed()


if __name__ == "__main__":
    main()
