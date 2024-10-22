# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from difflib import unified_diff

LANGUAGES = ["python"]


def diff(old_contents, new_contents):
    return "\n".join(
        unified_diff(
            old_contents.split("\n"),
            new_contents.split("\n"),
            fromfile="old",
            tofile="new",
            lineterm="",
        )
    )


def license_filter(commit):
    return commit["license"] in [
        "apache-2.0",
        "mit",
        "bsd-3-clause",
        "bsd-2-clause",
        # Python EditPackFT does not use licenses below, but they are permissive anyways :P
        # "cc0-1.0",
        # "isc",
        # "artistic-2.0",
        # "epl-1.0",
    ]


def content_filter(commit):
    """Filter out those with empty old/new contents"""
    return commit["old_contents"].strip() and commit["new_contents"].strip()


def get_filters():
    return [license_filter, content_filter]
