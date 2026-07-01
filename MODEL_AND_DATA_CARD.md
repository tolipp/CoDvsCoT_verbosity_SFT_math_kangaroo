# Model and Data Card

## Dataset Scope

- source corpus: `kaenguru_2021_2025_no_diagram_finetune.json`
- language: German
- task: five-option Kangaroo mathematics multiple-choice problems
- curated corpus size: `324`
- split sizes:
  - train years `2021–2023`: `186`
  - validation year `2024`: `67`
  - held-out test year `2025`: `71`

## Public Artifact Layer

The current public repository surface preserves:

- source corpus files:
  - `kaenguru_2021_2025_no_diagram_finetune.json`
  - `kaenguru_2021_2023_no_diagram_finetune.json`
  - `test_2025.jsonl`
- 2025 row-level evaluation outputs in `kaggle_outputs/kaggle_kernel_outputs/`
- 2026 row-level evaluation outputs in `kaggle_outputs/qwen3_8b_eval_2026/`
- unchanged-model OpenRouter baseline outputs in
  `kaggle_outputs/openrouter_qwen3_8b_pretrain_baseline_max4096_2025_2026/`
- preserved training-loss histories in `training_artifacts/training_logs/`
- preserved checkpoint-selection summaries in
  `training_artifacts/checkpoint_selection/`
- compact manifests in `metadata/`

This means the public archive is strongest as an artifact-backed reproducibility
package for the reported analyses, not as a complete export of every
intermediate local training package used during development.

The accompanying Kaggle archive stores larger artifacts that are impractical to
keep directly in GitHub, including bulk `generated_files/` packages,
condition-specific SFT data packages, and final LoRA adapter artifacts where
available.

## Supervision Conditions

- official human solution: official human-written solution text
- verbose solve: direct qwen3-32b teacher solve in a verbose CoT-style format
- concise solve: direct qwen3-32b teacher solve in a concise CoD-style format
- concise rewrite: qwen3-32b concise rewrite of the official solution
- verbose rewrite: qwen3-32b verbose rewrite of the official solution

Historical labels still used in artifacts:

- `1` official human solution
- `2a` verbose solve
- `3a` concise solve
- `4a` concise rewrite
- `5a` verbose rewrite

## Original vs Derived Material

Original or externally sourced:

- Kangaroo problem statements and official-solution material
- pretrained model families:
  - `Qwen/Qwen3-8B`
  - `qwen/qwen3-32b`

Derived in this project:

- the text-only curated corpus JSON
- qwen3-32b teacher annotation bundles, stored in the large Kaggle artifact
  archive where not present in the lightweight GitHub tree
- qwen3-32b rewrite bundles, stored in the large Kaggle artifact archive where
  not present in the lightweight GitHub tree
- condition-specific SFT JSONLs, stored in the large Kaggle artifact archive
  where not present in the lightweight GitHub tree
- evaluation outputs and summaries
- failure-analysis tables and tags
- fine-tuned LoRA adapters, stored in the large Kaggle artifact archive where
  available
- thesis text, figures, and tables

## Preserved Training Artifacts

The repository preserves:

- extracted training-loss histories in `training_artifacts/training_logs/`
- checkpoint-selection summaries in `training_artifacts/checkpoint_selection/`
- consolidated manifests in:
  - `metadata/training_run_manifest.csv`
  - `metadata/training_log_manifest.csv`
  - `metadata/checkpoint_selection_manifest.csv`
  - `metadata/training_data_manifest.csv`

## Methodological Limits

- The study does not isolate verbosity as a single factor; verbosity varies alongside source type and supervision format.
- Only three student seeds are preserved.
- The held-out test set contains `71` problems.
- The current public repository surface does not expose every intermediate local training package or helper script referenced during development.
- Large generated training packages and final adapter artifacts are documented as
  Kaggle archive artifacts rather than ordinary GitHub files.
- Evaluation output length is not identical to supervised verbosity; some long student generations come from repetition or max-token behavior rather than useful reasoning.

## Redistribution Note

This repository contains Kangaroo-derived texts and derived JSONL files. Public scientific redistribution is permitted, while the underlying competition content remains the intellectual property of Math Kangaroo Switzerland (Mathe Känguru Schweiz). The repository and its data should not be used for personal profit or other commercial exploitation.
