# Learning Code Preference via Synthetic Evolution

[![arXiv](https://img.shields.io/badge/arXiv-2410.03837-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2410.03837)

<p align="center">
    <a href="#-tldr">📰 TL;DR</a> •
    <a href="#-evaluation">🔎 Evaluation</a> •
    <a href="#-training">🧪 Training</a> •
    <a href="#-synthetic-data-generation">🔮 Synthetic Data Generation</a> •
    <a href="#-citation">📜 Citation</a> •
    <a href="#-acknowledgement">🙏 Acknowledgement</a>
</p>

## 📰 TL;DR

How to *effectively* and *efficiently* obtain code preferences and judgements is an important yet under-studied topic! 

To this end, our work provides:

* **CodeFavor**: an open recipe to train code preference models with from-scratch data!
  * *Commit-Instruct*: code commits -> code preference
  * *Critic-Evol*: code critique & revising -> code preference
* **CodePrefBench**: 1364 code preference tasks covering both verifiable and human objectives:
  * *Code Correctness*
  * *Code Efficiency*
  * *Code Security*
  * *Human Preference*
* **Study**: our paper provides comprehensive studies!
  * *Human studies*: quantifying the cost and performance of human preference based 18 developers
  * *Case studies*: our Appendix case-studies model preferences over code correctness, efficiency, and security
  * *Controlled experiments*: impact of data, comment, criteria, modeling, etc. on training preference models

![](./assets/codefavor.png)

## 🔎 Evaluation

### Environment

* Python requirements: 3.10 or higher.

```bash
conda create -n codefavor python=3.10
conda activate codefavor
pip install -r requirements.txt
```

### CodePrefBench

```bash
# OpenAI server
python codefavor/evaluate.py --model-id "gpt-4o-2024-05-13" --model-type openai --concurrency 80
# Other OpenAI-compatible servers (vLLM, DeepSeek APIs, etc.)
python codefavor/evaluate.py --model-id "google/gemma-2-27b-it" --model-type openai --concurrency 80 --model-url http://localhost:8000/v1
# Claude models at Bedrock
python codefavor/evaluate.py --model-id "anthropic.claude-3-sonnet-20240229-v1:0" --model-type bedrock --concurrency 10
# Pairwise RM
python codefavor/evaluate.py --model-id ./models/mix-cls-mistral-7b-it_bs32_ep1_lr5e-6-l3-70b/checkpoint-688 --model-type pair-rm
```

* Supported `--model-type`: `huggingface`, `openai`, `bedrock`, `pair-rm`, and `google`

## 🧪 Training

### Environment

```bash
git clone https://github.com/axolotl-ai-cloud/axolotl.git axolotl-dep
cd axolotl-dep

pip install torch==2.3.0
pip install packaging ninja wandb
pip install -e '.[flash-attn,deepspeed]'
```

### Use existing dataset

```bash
python scripts/axolotl/prepare_data.py \
    --decomposed-dataset datasets/train/editpackft-Llama-3-70B-Instruct.commit_instruct.decompose.jsonl \
    --judge-type classification --both-order
python scripts/axolotl/prepare_data.py \
    --decomposed-dataset datasets/train/Llama-3-8B-Instruct-SOSS.teacher.Llama-3-70B-Instruct.critic_evol.decompose.jsonl \
    --judge-type classification --both-order
```

### Train models using Axolotl

```bash
accelerate launch -m axolotl.cli.train \
    scripts/axolotl/recipe/gemma/cls-commit-instruct-from-llama3-70b.yaml \
    --deepspeed scripts/axolotl/zero3.json
# or use `torchrun` if your `accelerate` is complaining
torchrun --nproc_per_node 8 -m axolotl.cli.train \
    scripts/axolotl/recipe/gemma/cls-commit-instruct-from-llama3-70b.yaml \
    --deepspeed scripts/axolotl/zero3.json
```

## 🔮 Synthetic Data Generation

### Commit-Instruct from Scratch

```bash
# Support OpenAI and Bedrock interface
# OAI interface
python codefavor/prompt/commit_instruct.py --model-id "deepseek-chat" --model-type "openai" --concurrency 256 --dataset editpackft --model-url "https://api.deepseek.com/v1"
# Bedrock interface
python codefavor/prompt/commit_instruct.py --model-id "meta.llama3-1-405b-instruct-v1:0" --model-type "bedrock" --concurrency 10 --dataset editpackft
```

### Critic-Evol from Scratch

```bash
python codefavor/prompt/critic_evol.py --weak-dataset ./datasets/train/Llama-3-8B-Instruct-SOSS.jsonl \
                                     --model-id "deepseek-coder" --model-url "https://api.deepseek.com/v1"
python codefavor/prompt/critic_evol.py --weak-dataset ./datasets/train/Llama-3-8B-Instruct-SOSS.jsonl \
                                     --model-id "meta.llama3-1-405b-instruct-v1:0" --concurrency 10
```

* Pairwise training code is partially adopted from https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/pair-pm

## 📜 Citation

```bibtex
@article{codefavor,
  title = {Learning Code Preference via Synthetic Evolution},
  author = {Liu, Jiawei and Nguyen, Thanh and Shang, Mingyue and Ding, Hantian and Li, Xiaopeng and Yu, Yu and Kumar, Varun and Wang, Zijian},
  journal = {arXiv preprint arXiv:2410.03837},
  year = {2024},
}
```

## 🙏 Acknowledgement

* Our training code is partially adapted from [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling).
* Our evaluation code is partially adapted from [RepoQA](https://github.com/evalplus/repoqa).
* The seed corpus used in this paper comes from [EditPackFT](https://huggingface.co/datasets/nuprl/EditPackFT) and [Self-OSS-Instruct](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k).

## 🎓 Research Use Only
This source code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper, but interested parties are encouraged to open an issue requesting open source community development.
