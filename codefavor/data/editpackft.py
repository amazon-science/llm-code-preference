# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from codefavor.data.utility import LANGUAGES, get_filters
from datasets import load_dataset


# languages not fixed
def prepare_editpackft(languages=LANGUAGES):
    dataset = load_dataset("nuprl/EditPackFT-Multi", split="train").filter(
        lambda x: all(f(x) for f in get_filters())
    )
    lang2data = {}
    for row in dataset:
        lang = row["lang"].lower()
        if lang.startswith("html"):  # normalize html
            lang = "html"
            row["lang"] = "html"
        if lang not in languages:
            continue
        lang2data.setdefault(lang, []).append(row)
    return lang2data
