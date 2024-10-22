# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List

import numpy as np
import torch
from stop_sequencer import StopSequencer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from codefavor.provider.base import BaseCaller, hacky_assistant_stop_seq
from codefavor.template import pairwise_classification_template
from codefavor.utility import remove_comments


class HuggingFaceCaller(BaseCaller):
    def __init__(self, model_id, model_url, temperature=0.8) -> None:
        super().__init__(model_id, model_url, temperature)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation=(
                "flash_attention_2" if os.getenv("DISABLE_FA", False) else None
            ),
        ).cuda()
        assert self.tokenizer.chat_template is not None
        seq_split_eos = hacky_assistant_stop_seq(self.tokenizer)
        self.extra_eos = [seq_split_eos] if seq_split_eos else []
        self.stop_sequencer = StopSequencer(
            self.model,
            model_type="causal",  # or seq2seq
            tokenizer=self.tokenizer,
        )

    @property
    def model_type(self):
        return "huggingface"

    @torch.no_grad()
    def call(self, messages, max_new_tokens=2048, eos=None) -> str:
        # swap the order to mitigate positional bias
        input_tokens = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).cuda()
        input_length = input_tokens.size(-1)

        eos = eos or []
        if eos:
            hf_model = self.stop_sequencer.register_stop_texts(
                stop_texts=eos, input_length=input_length
            )
        else:
            hf_model = self.model

        kwargs = {"do_sample": False}
        if self.temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = self.temperature

        output_text = hf_model.generate(
            input_tokens,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        return self.tokenizer.decode(
            output_text[0][input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


def try_remove_comments(code: str):
    try:
        return remove_comments(code)
    except Exception:
        return code


class PairwiseRMCaller(BaseCaller):
    def __init__(self, model_id, model_url, temperature=0.8) -> None:
        super().__init__(model_id, model_url, temperature)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).cuda()
        self.token_id_a = self.tokenizer.encode("A", add_special_tokens=False)
        self.token_id_b = self.tokenizer.encode("B", add_special_tokens=False)
        assert len(self.token_id_a) == 1 and len(self.token_id_b) == 1
        self.token_id_a = self.token_id_a[0]
        self.token_id_b = self.token_id_b[0]

    @property
    def model_type(self):
        return "pair-rm"

    @torch.no_grad()
    def call(self, messages, max_new_tokens=2048, eos=None) -> str:
        # swap the order to mitigate positional bias
        input_ids = self.tokenizer.encode(
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ).replace(self.tokenizer.bos_token, ""),
            return_tensors="pt",
            add_special_tokens=False,
        ).cuda()
        output = self.model(input_ids)
        logit_A = output.logits[0, -1, self.token_id_a].item()
        logit_B = output.logits[0, -1, self.token_id_b].item()

        if logit_A > logit_B:
            return "A"
        return "B"

    def apply_prompt_template(self, instruction, left, right, criteria):
        if self.model_id == "RLHFlow/pair-preference-model-LLaMA3-8B":
            prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"
            prompt = prompt_template.format(
                context=self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}], tokenize=False
                ),
                response_A=left,
                response_B=right,
            )
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False
            ).replace(self.tokenizer.bos_token, "")
        return self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": pairwise_classification_template(
                        instruction, left, right, criteria
                    ),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        ).replace(self.tokenizer.bos_token, "")

    @torch.no_grad()
    def batch_infer(
        self, instruction, left_samples, right_samples, criteria, max_batch=None
    ) -> List[str]:
        assert criteria is not None
        assert len(left_samples) == len(right_samples)

        left_samples = [try_remove_comments(s) for s in left_samples]
        right_samples = [try_remove_comments(s) for s in right_samples]

        input_strings = [
            self.apply_prompt_template(instruction, l, r, criteria)
            for l, r in zip(left_samples, right_samples)
        ]

        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            return_attention_mask=True,
        ).to("cuda")

        inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        logits = []
        if not max_batch:
            for logit in self.model(**inputs).logits:
                logits.append(
                    {
                        "A": logit[-1, self.token_id_a].item(),
                        "B": logit[-1, self.token_id_b].item(),
                    }
                )
        else:
            for i in range(0, len(inputs["input_ids"]), max_batch):
                for logit in self.model(
                    **{k: v[i : i + max_batch] for k, v in inputs.items()}
                ).logits:
                    logits.append(
                        {
                            "A": logit[-1, self.token_id_a].item(),
                            "B": logit[-1, self.token_id_b].item(),
                        }
                    )

        results = []

        for logit in logits:
            Z = np.exp(logit["A"] / self.temperature) + np.exp(
                logit["B"] / self.temperature
            )
            results.append(
                {
                    "A": np.exp(logit["A"] / self.temperature) / Z,
                    "B": np.exp(logit["B"] / self.temperature) / Z,
                }
            )

        return results


class ScoreRM(BaseCaller):
    def __init__(self, model_id, temperature=0.8) -> None:
        super().__init__(model_id, None, temperature)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            num_labels=1,
        ).cuda()

    def call(self, messages, max_new_tokens=2048, eos=None) -> str:
        raise NotImplementedError

    @property
    def model_type(self):
        return "score-rm"

    @torch.no_grad()
    def batch_infer(self, instruction, samples, max_batch=None) -> List[str]:
        samples = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": s},
                ],
                tokenize=False,
                add_generation_prompt=False,
            ).replace(self.tokenizer.bos_token, "")
            for s in samples
        ]
        inputs = self.tokenizer(
            samples,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            return_attention_mask=True,
        ).to("cuda")

        if not max_batch:
            return self.model(**inputs).logits.float().tolist()
        else:
            scores = []
            for i in range(0, len(inputs["input_ids"]), max_batch):
                output = self.model(
                    **{k: v[i : i + max_batch] for k, v in inputs.items()}
                )
                scores.extend(output.logits.float().tolist())
            return scores
