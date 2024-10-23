# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import json

from IPython import embed

from codefavor.data.utility import diff
from codefavor.prompt import *


def viz(row):
    print(f"Selected Data: {row['note']}")

    for item in row["conversations"]:
        print("-------", item["role"], "-------")
        print(item["content"])


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
        if row["note"] == "success":
            yes_data.append(row)
        else:
            no_data.append(row)

    # Check the overall rate of HQ commits
    print(f"# Total = {len(data)}")
    print(f"# Yes = {len(yes_data)} :: {len(yes_data)/len(data)*100:.1f}%")
    print(f"# No = {len(no_data)}:: {len(no_data)/len(data)*100:.1f}%")
    print("-" * 16)

    # Check if there's data duplicates
    all_commits = [row["seed_info"]["commit"] for row in data]
    unique_commits = set(all_commits)
    print(f"#All commits: {len(all_commits)}")
    print(f"#Unique commits: {len(unique_commits)}")
    print(f"#Duplicated commits: {len(all_commits) - len(unique_commits)}")

    commit2yes = {r["seed_info"]["commit"]: r for r in yes_data}
    commit2no = {r["seed_info"]["commit"]: r for r in no_data}

    return commit2yes, commit2no


def viz_both(yes, no):
    print("=== PASSING ==")
    viz(yes)
    print("=== FAILING ==")
    viz(no)


def diff_commits(lset, rset) -> List:
    # find out commits where a commit is included (passes) in lset but not in rset
    # and make it deterministic
    diffset = []
    for c in lset:
        if c not in rset:
            diffset.append(c)
    return diffset


def main(dataset: str, compare: str = None):
    commit2yes, commit2no = load_dataset(dataset)

    if compare:
        comp_commit2yes, comp_commit2no = load_dataset(compare)
        base_yes_ex = diff_commits(commit2yes.keys(), comp_commit2yes.keys())
        base_no_ex = diff_commits(commit2no.keys(), comp_commit2no.keys())
    embed()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
