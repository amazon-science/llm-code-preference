# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: Apache-2.0


class ChatTermination(Exception):
    def __init__(self, reason) -> None:
        super().__init__(f"Chat terminated. Reason: {reason}")
        self.reason: str = reason
