# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import os

import transformers
from huggingface_hub import repo_exists
from openai import AsyncClient, Client
from transformers import AutoTokenizer

from codefavor.provider.base import BaseCaller, hacky_assistant_stop_seq

transformers.logging.set_verbosity_error()


class OpenAICaller(BaseCaller):
    def __init__(self, model_id, model_url, temperature=0.8) -> None:
        super().__init__(model_id, model_url, temperature)
        self.client = Client(
            api_key=os.getenv("OPENAI_API_KEY", "none"), base_url=model_url
        )
        self.base_eos = []
        if not os.getenv("NO_BASE_EOS", model_id.count("/") > 1) and repo_exists(
            model_id
        ):
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.chat_template:
                self.base_eos.append(hacky_assistant_stop_seq(tokenizer))

    @property
    def model_type(self):
        return "openai"

    def call(self, messages, max_new_tokens=2048, eos=None):
        eos = eos or []
        eos += self.base_eos
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=self.temperature,
            n=1,
        )
        return response.choices[0].message.content


class OpenAIAsyncCaller(BaseCaller):
    def __init__(self, model_id, model_url, temperature=0.8) -> None:
        super().__init__(model_id, model_url, temperature)
        self.client = AsyncClient(
            api_key=os.getenv("OPENAI_API_KEY", "none"), base_url=model_url
        )
        self.base_eos = []
        if not os.getenv("NO_BASE_EOS", model_id.count("/") > 1) and repo_exists(
            model_id
        ):
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.chat_template:
                self.base_eos.append(hacky_assistant_stop_seq(tokenizer))

    @property
    def model_type(self):
        return "openai"

    async def call(self, messages, max_new_tokens=2048, eos=None):
        eos = eos or []
        eos += self.base_eos
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=self.temperature,
            n=1,
        )
        return response.choices[0].message.content
