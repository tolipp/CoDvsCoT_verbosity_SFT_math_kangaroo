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

## Primary Training Snapshot

The main reproduction layer for the thesis-result runs is:

- `generated_files/kaggle_qwen3_32b_5_conditions_2021_2023/data/`

Training rows:

- official human solution: `185`
- verbose solve: `183`
- concise solve: `186`
- concise rewrite: `186`
- verbose rewrite: `186`

Validation / held-out files:

- `valid_2024.jsonl`: `67`
- `val_2a.jsonl`: `46`
- `val_3a.jsonl`: `63`
- `val_4a.jsonl`: `67`
- `val_5a.jsonl`: `67`
- `test_2025.jsonl`: `71`

Additional preserved variant:

- `train_official_human_2021_2023_seed42_full_186.jsonl`: `186`

The recovery record for this layer is in:

- `metadata/RUNPOD_ALIGNED_PACKAGE.json`
- `metadata/RUNPOD_ALIGNED_PACKAGE.md`

## Secondary Canonical Release

The repository also keeps a stricter source-rebuilt package at:

- `generated_files/kaggle_qwen3_32b_5_conditions_2021_2023_canonical_refreshed/data/`

That layer is useful for inspection and clean public release, but it is smaller than the preserved RunPod-aligned training snapshot.

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
- qwen3-32b teacher annotation bundles
- qwen3-32b rewrite bundles
- condition-specific SFT JSONLs
- evaluation outputs and summaries
- failure-analysis tables and tags
- fine-tuned LoRA adapters where preserved locally
- thesis text, figures, and tables

## Preserved Training Artifacts

The repository preserves:

- final LoRA adapters for all 15 student runs in `training_artifacts/final_adapters/`
- `training_meta.json` and `trainer_state.json` alongside those adapters
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
- The RunPod-aligned package is a best-effort local reconstruction of the JSONLs referenced by the preserved student-run metadata.
- Some rows in the historical training layer remain terse or answer-dominant because that is how the saved teacher-side artifacts survived.
- Evaluation output length is not identical to supervised verbosity; some long student generations come from repetition or max-token behavior rather than useful reasoning.

## Redistribution Note

This repository contains Kangaroo-derived texts and derived JSONL files. Public scientific redistribution is permitted, while the underlying competition content remains the intellectual property of Math Kangaroo Switzerland (Mathe Känguru Schweiz). The repository and its data should not be used for personal profit or other commercial exploitation.
