# Bias Mitigation & Evaluation

This repository contains a **reproducible pipeline** for generating data, fine‑tuning language models with LoRA, and evaluating model bias using a suite of custom metrics.

It accompanies the research paper  *“Promoting Fairness in LLMs: Detection and Mitigation of Gender Bias”*

> **Status** : The codebase is functional but  **under active refactor** . Expect cleaner interfaces, additional documentation, and unit tests in upcoming commits.

---

## 📂 Repository Structure

| File / Folder                   | Purpose                                                                                                                     |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `custom_dataset_templates.py` | Generates a*synthetic training corpus* following a controlled prompt‑response schema (see*Dataset Generation*below).   |
| `unbiased_dataset.json`       | Ready‑to‑use dataset produced by `custom_dataset_templates.py` (balanced across gender, occupation, country, trait).    |
| `gpt3.py`                     | End‑to‑end LoRA fine‑tuning on GPT‑2. Handles data loading, tokenizer setup (with special tokens), and training loop. |
| `training_model.py`           | Low‑level helpers used by `gpt3.py` (model/optimizer config, checkpoints, etc.).                                         |
| `running_finetuned_model.py`  | Loads the locally saved LoRA checkpoints and runs offline inference  for quick testing.                                   |
| `Metrics_Eval.ipynb`          | Jupyter notebook with code cells that compute DI, ICS, TCS, IAT, and ZSC on model outputs and visualize results.          |
| `ics_testing.py`              | Stand‑alone script to run Idea Consistency Score evaluation and auto‑generate comparison plots.                         |
| `zero_shot_testing.py`        | Evaluates Zero‑Shot Classification (ZSC) scores and renders bias heat‑maps.                                               |
| `train_llama3b.py`            | Meta Llama 3B model                                                                                                         |
|                                 |                                                                                                                             |
|                                 |                                                                                                                             |
|                                 |                                                                                                                             |
|                                 |                                                                                                                             |
|                                 |                                                                                                                             |
|                                 |                                                                                                                             |
|                                 |                                                                                                                             |
|                                 |                                                                                                                             |
|                                 |                                                                                                                             |
