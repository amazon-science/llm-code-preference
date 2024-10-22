# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import os

import boto3
from botocore.config import Config
from dotenv import load_dotenv

from codefavor.provider.base import SYSTEM_PROMPT, BaseCaller

# load the environment variables from the .env file
load_dotenv()
AWS_ROLE_ARN = os.getenv("AWS_ROLE_ARN", None)
REGION = os.getenv("REGION", "us-east-1")

BEDROCK_CONFIG = Config(retries={"max_attempts": 100, "mode": "standard"})


class BedrockClientWithAutoRefresh:
    def __init__(self, role_arn, session_name):
        self.role_arn = role_arn
        self.session_name = session_name
        self.session = boto3.Session()
        self.sts_client = self.session.client("sts", region_name=REGION)
        self.bedrock_client = boto3.client(
            "bedrock-runtime",
            config=BEDROCK_CONFIG,
            region_name=REGION,
        )
        self.expiration = None
        self.refresh_credentials()

    def refresh_credentials(self):
        if AWS_ROLE_ARN is None:
            return
        assumed_role = self.sts_client.assume_role(
            RoleArn=self.role_arn,
            RoleSessionName=self.session_name,
            DurationSeconds=12 * 60 * 60,
        )
        credentials = assumed_role["Credentials"]
        self.bedrock_client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
            region_name=REGION,
            config=BEDROCK_CONFIG,
        )
        self.expiration = credentials["Expiration"]

    def _refresh_guard(self):
        if self.expiration is None or datetime.datetime.now(
            datetime.timezone.utc
        ) > self.expiration - datetime.timedelta(minutes=10):
            self.refresh_credentials()

    def converse(self, *arg, **kwargs):
        self._refresh_guard()
        return self.bedrock_client.converse(*arg, **kwargs)


class BedrockCaller(BaseCaller):
    def __init__(self, model_id, model_url=None, temperature=0.8) -> None:
        super().__init__(model_id, model_url, temperature)
        self.client = BedrockClientWithAutoRefresh(
            role_arn=AWS_ROLE_ARN, session_name="BedrockSession"
        )

    def call(self, messages, max_new_tokens=2048, eos=None) -> str:
        eos = eos or []

        messages = [
            {"role": message["role"], "content": [{"text": message["content"]}]}
            for message in messages
        ]

        response = self.client.converse(
            modelId=self.model_id,
            messages=messages,
            system=[{"text": SYSTEM_PROMPT}],
            inferenceConfig={
                "maxTokens": max_new_tokens,
                "stopSequences": eos,
                "temperature": self.temperature,  # self.temperature,
            },
        )
        response_text = response["output"]["message"]["content"][0]["text"]

        return response_text

    @property
    def model_type(self):
        return "bedrock"
