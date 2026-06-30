# CoDvsCoT_verbosity_SFT_math_kangaroo

## Verbose and Concise Teacher Supervision for Mathematical Reasoning: A Comparison of CoT- and CoD-Style Prompts in SFT

This repository packages the main data, evaluation artifacts, scripts, and
training metadata used in the study. It is organized as a reproducibility
archive for the reported Qwen3-8B supervised fine-tuning experiments on German
Kangaroo mathematics multiple-choice problems.

## Public Archive

A public Kaggle copy of the SFT package downloaded from runpod is available at:

- `https://www.kaggle.com/datasets/tolipp/codvscot-sft-math-repro-public`

The Kaggle archive is a shareable distribution layer for the reproducibility
package. Directory contents such as `kaggle_outputs/`, `scripts/`,
`metadata/`, `environment/`, and `training_artifacts/` are stored there as tar
archives because of the Kaggle upload format used for this release.

Suggested citation for the Kaggle archive:

- T. Lippuner, *CoDvsCoT SFT Math Repro Public*, Kaggle dataset, 2026.
  `https://www.kaggle.com/datasets/tolipp/codvscot-sft-math-repro-public`

## Repository Overview

- `generated_files/`: teacher annotation sources,  teacher annotation
  files, and condition-specific SFT data packages
- `kaggle_outputs/`: preserved 2025 and 2026 evaluation outputs, plus archived
  OpenRouter baseline runs
- `training_artifacts/`: preserved final LoRA adapters, training-loss histories,
  and checkpoint-selection records
- `scripts/`: rebuild and analysis scripts kept in the public package
- `metadata/`: compact manifests for training data, training logs, checkpoints,
  and run metadata
- `environment/`: dependency snapshots for analysis and training environments

## Primary Training Snapshot

The main training package for the reported Qwen3-8B student runs is:

- `generated_files/kaggle_qwen3_32b_5_conditions_2021_2023/data/`

Current row counts in that package:

- `train_official_human_2021_2023.jsonl`: `186`
- `train_qwen3_32b_verbose_2021_2023.jsonl`: `186`
- `train_qwen3_32b_concise_2021_2023.jsonl`: `186`
- `train_qwen3_32b_concise_rewrite_2021_2023.jsonl`: `186`
- `train_qwen3_32b_verbose_rewrite_2021_2023.jsonl`: `186`
- `valid_2024.jsonl`: `67`
- `val_2a.jsonl`: `67`
- `val_3a.jsonl`: `63`
- `val_4a.jsonl`: `67`
- `val_5a.jsonl`: `67`
- `test_2025.jsonl`: `71`

An additional official-human variant is also preserved:

- `train_official_human_2021_2023_seed42_full_186.jsonl`: `186`

## Preserved Student Artifacts

The repository preserves:

- final LoRA adapters for all 15 student runs in `training_artifacts/final_adapters/`
- extracted training-loss histories in `training_artifacts/training_logs/`
- checkpoint-selection summaries in `training_artifacts/checkpoint_selection/`
- full 2025 seed-by-condition evaluation artifacts in `kaggle_outputs/kaggle_kernel_outputs/`
- 2026 seed-by-condition evaluation artifacts in `kaggle_outputs/qwen3_8b_eval_2026/`
- unchanged-model OpenRouter baseline outputs in `kaggle_outputs/openrouter_qwen3_8b_pretrain_baseline_max4096_2025_2026/`

## Condition Mapping

Preferred condition names:

- official human solution
- verbose solve
- concise solve
- concise rewrite
- verbose rewrite

Historical artifact labels:

- `1` = official human solution
- `2a` = verbose solve
- `3a` = concise solve
- `4a` = concise rewrite
- `5a` = verbose rewrite

## Core Study Setup

- student model: `Qwen/Qwen3-8B`
- teacher model: `qwen/qwen3-32b`
- task: German Kangaroo mathematics multiple-choice problems
- source corpus file: `kaenguru_2021_2025_no_diagram_finetune.json`
- source split sizes:
  - train years `2021–2023`: `186`
  - validation year `2024`: `67`
  - held-out test year `2025`: `71`

## Public Scripts

The public script set currently includes:

- `python3 scripts/build_qwen3_32b_5_conditions_sft.py`
- `python3 scripts/build_release_qwen3_32b_package.py`
- `python3 scripts/validate_refreshed_package.py`
- `python3 scripts/build_qwen3_8b_final_results.py`
- `python3 scripts/build_qwen3_8b_deep_analysis.py`
- `python3 scripts/build_verbosity_analysis.py`

These scripts are intended to build or analyze the packaged artifacts rather
than to expose every intermediate local helper used during the project.

## Metadata

The compact metadata layer consists of:

- `metadata/training_data_manifest.csv`
- `metadata/training_log_manifest.csv`
- `metadata/checkpoint_selection_manifest.csv`
- `metadata/training_run_manifest.csv`

## Rights

The underlying Math Kangaroo materials remain the intellectual property of Math
Kangaroo Switzerland (Mathe Känguru Schweiz). This repository is intended for
scientific and academic use and must not be used for personal profit or other
commercial exploitation.
