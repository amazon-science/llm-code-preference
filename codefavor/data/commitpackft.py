# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from codefavor.data.utility import LANGUAGES, get_filters
from datasets import load_dataset


def prepare_commitpackft(languages=LANGUAGES):
    return {
        lang: load_dataset("bigcode/commitpackft", lang, split="train").filter(
            lambda x: all(f(x) for f in get_filters())
        )
        for lang in languages
    }
