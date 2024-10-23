# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from abc import ABC, abstractmethod
from typing import Any

SYSTEM_PROMPT = "You are a helpful code assistant good at analyzing and generating high-quality code."


class BaseCaller(ABC):
    def __init__(self, model_id, model_url, temperature=0.8) -> None:
        super().__init__()
        self.model_id = model_id
        self.model_url = model_url
        self.temperature = temperature

    @property
    @abstractmethod
    def model_type(self):
        pass

    @abstractmethod
    def call(self, messages, max_new_tokens=2048, eos=None) -> str:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> str:
        return self.call(*args, **kwds)


# This function is adapted from
# https://github.com/evalplus/repoqa/blob/main/repoqa/provider/request/__init__.py
# which inherits their Apache-2.0 license.
def hacky_assistant_stop_seq(tokenizer) -> str:
    _magic_string_ = "&== [HACKY SPLITTER] ==&"
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": _magic_string_},
        ],
        tokenize=False,
    ).split(_magic_string_)[-1]
