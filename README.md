# CoDvsCoT_verbosity_SFT_math_kangaroo

## Verbose and Concise Teacher Supervision for Mathematical Reasoning: A Comparison of CoT- and CoD-Style Prompts in SFT

This repository packages the main data, evaluation artifacts, scripts, and
training metadata used in the study. It is organized as a reproducibility
archive for the reported Qwen3-8B supervised fine-tuning experiments on German
Kangaroo mathematics multiple-choice problems.

## Public Archive

A public Kaggle copy of the large artifact bundle is available at:

- `https://www.kaggle.com/datasets/tolipp/codvscot-sft-math-repro`

The Kaggle archive is a shareable distribution layer for the reproducibility
package. Directory contents such as `kaggle_outputs/`, `scripts/`,
`metadata/`, `environment/`, and `training_artifacts/` are stored there as tar
archives because of the limited storage on GitHub. The GitHub repository is the
readable, lightweight reproducibility layer. The Kaggle archive also stores the
larger artifacts that are impractical to keep directly in GitHub, including the
condition-specific `generated_files/` packages and the final LoRA adapter
artifacts where available.

Suggested citation for the Kaggle archive:

- T. Lippuner, *CoDvsCoT SFT Math Repro*, Kaggle dataset, 2026.
  `https://www.kaggle.com/datasets/tolipp/codvscot-sft-math-repro`

## Repository Overview

- source corpus files:
  - `kaenguru_2021_2025_no_diagram_finetune.json`
  - `kaenguru_2021_2023_no_diagram_finetune.json`
  - `test_2025.jsonl`
- `kaggle_outputs/`: preserved 2025 and 2026 evaluation outputs, plus archived
  OpenRouter baseline runs
- `training_artifacts/`: preserved training-loss histories and
  checkpoint-selection records
- `scripts/`: rebuild and analysis scripts kept in the public package
- `metadata/`: compact manifests for training data, training logs, checkpoints,
  and run metadata
- `environment/`: dependency snapshots for analysis and training environments

Large artifact layers stored in the Kaggle archive rather than directly in
GitHub include:

- `generated_files/`: teacher annotation sources, teacher annotation files, and
  condition-specific SFT data packages
- final LoRA adapter artifacts for the student runs, where preserved
- larger packaged bundles used for Kaggle/RunPod transfer

## Preserved Student Artifacts

The repository preserves:

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

## Public Surface Note

This public repository is an artifact-backed reproducibility archive. Its
strongest preserved layer consists of the source corpus files, the row-level
2025 and 2026 evaluation outputs, the OpenRouter baseline outputs, the training
log and checkpoint manifests, and the analysis scripts that rebuild the reported
tables from those artifacts. The large Kaggle archive complements this GitHub
repository with bulk artifacts such as `generated_files/` and final adapter
packages. Neither archive should be read as a guarantee of bit-identical
retraining, because exact reproduction also depends on external model hosting,
hardware, library versions, and GPU nondeterminism.

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
