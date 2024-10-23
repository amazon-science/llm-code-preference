# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import time

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted

from codefavor.provider.base import BaseCaller


def make_request(
    client: genai.GenerativeModel, temperature, messages, max_new_tokens=2048, eos=None
) -> genai.types.GenerateContentResponse:
    messages = [{"role": m["role"], "parts": [m["content"]]} for m in messages]
    response = client.generate_content(
        messages,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=max_new_tokens,
            temperature=temperature,
            # stop_sequences=eos, # Gemini would trim the eos unfortunately
        ),
        safety_settings=[
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        ],
    )

    return response.text


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def make_auto_request(*args, **kwargs) -> genai.types.GenerateContentResponse:
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        except ResourceExhausted as e:
            print("Rate limit exceeded. Waiting...", e.message)
            time.sleep(10)
        except GoogleAPICallError as e:
            print(e.message)
            time.sleep(1)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret


class GoogleCaller(BaseCaller):
    def __init__(self, model_id, model_url, temperature=0.8) -> None:
        super().__init__(model_id, model_url, temperature)
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(self.model_id)
        self.base_eos = []

    @property
    def model_type(self):
        return "google"

    def call(self, messages, max_new_tokens=2048, eos=None):
        eos = eos or []
        eos += self.base_eos

        return make_auto_request(
            self.client, self.temperature, messages, max_new_tokens, eos
        )
