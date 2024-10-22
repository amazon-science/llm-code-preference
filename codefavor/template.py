# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: Apache-2.0

CORRECTNESS_CRITERIA = "The ideal code should be directly runnable, correct, and bug-free to the given instruction."
EFFICIENCY_CRITIERIA = "The ideal code should be efficient in execution time (primary) and memory usage (secondary)."
SECURITY_CRITERIA = "The ideal code should be robust against security attacks and meet common security properties."
HUMAN_PREF_CRITERIA = "Select the code that a human developer would generally prefer for the given instruction"


def pairwise_cot_template(instruction, code1, code2, criteria) -> str:
    return f"""\
Given an [INSTRUCTION] and two response candidates of [CODE_A] and [CODE_B], \
please judge which of them better meets [CRITERIA] while following [INSTRUCTION]:

---
[INSTRUCTION]
{instruction}

[CODE_A]
{code1}

[CODE_B]
{code2}

[CRITERIA]
{criteria}
---

1. Please FIRST provide a brief [FEEDBACK] section regarding if the code meets [CRITERIA]
2. THEN conclude with a [RESULT] section suggesting the conclusion in the format of \
"[CODE_?] is better than [CODE_?] on the mentioned criteria".
"""


def pairwise_classification_template(instruction, code1, code2, criteria) -> str:
    assert criteria is not None
    assert isinstance(criteria, str)
    assert isinstance(code1, str)
    assert isinstance(code2, str)
    return f"""\
[INSTRUCTION] {instruction} \
[CODE_A] {code1} \
[CODE_B] {code2} \
[CRITERIA] {criteria}
"""


# General criteria for controlled experiments
# "The ideal code should be of high quality"
