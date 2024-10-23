# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rc("text", usetex=True)


def main(dataset_type: str):
    data = {}
    for train_type in ["naive", "improve"]:
        for eval_type in ["bad", "good"]:
            path = os.path.join(
                "storage",
                "raw_data",
                "contamination_analysis",
                f"{dataset_type}_{train_type}_{eval_type}_pair_results.jsonl",
            )
            train_note = f"Train$^{ '-' if train_type == 'naive' else '+' }$"
            eval_note = f"Eval$^{ '-' if eval_type == 'bad' else '+' }$"
            with open(path) as f:
                data[(train_note, eval_note)] = [
                    json.loads(line)["score"] for line in f
                ]

    plt.figure(figsize=(3.5, 2), constrained_layout=True)

    # plot cdf
    colors = [
        "orange",
        "mediumorchid",
        "royalblue",
        "yellowgreen",
    ]

    for i, (train_type, eval_type) in enumerate(data.keys()):
        scores = data[(train_type, eval_type)]
        plt.ecdf(
            scores,
            label=f"{eval_type} $\sim$ {train_type}",
            linewidth=2,
            color=colors[i],
        )

        # how many % of scores > CUTOFF_SCORE?
        CUTOFF_SCORE = 80
        percentile = np.sum(np.array(scores) <= CUTOFF_SCORE) / len(scores)
        print(
            f"{eval_type} {train_type} {percentile*100:.1f}% <= {CUTOFF_SCORE} similarity"
        )

        # Find the CDF value at score 90
        if (train_type, eval_type) == ("Train$^+$", "Eval$^-$"):
            plt.axhline(
                y=percentile,
                color="red",
                linestyle="dotted",
                linewidth=1,
            )
            plt.axvline(
                x=CUTOFF_SCORE,
                color="red",
                linestyle="dotted",
                linewidth=1,
            )
            # point to
            plt.text(
                CUTOFF_SCORE - 1,
                percentile * 1.03,
                f"\\textbf{{{(1 - percentile)*100:.1f}\%}} test code has similarity score $>$ {CUTOFF_SCORE}",
                verticalalignment="bottom",
                horizontalalignment="right",
                color="firebrick",
            )

    plt.legend()
    plt.xlabel("Top-1 Similarity Score")
    plt.ylim(0, 1.15)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks([30, 40, 50, 60, 70, 80, 90, 100])
    plt.xlim(40, 85)
    plt.grid(alpha=0.5, ls="--")

    plt.savefig(
        f"{dataset_type}_contamination.png", dpi=150, bbox_inches="tight", pad_inches=0
    )
    plt.savefig(f"{dataset_type}_contamination.pdf", bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
