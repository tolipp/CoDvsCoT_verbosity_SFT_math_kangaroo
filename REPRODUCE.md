# Reproduction Guide

This public repository supports reproduction of the reported analyses from the
preserved artifact layer that is currently included in the archive.

The strongest preserved public layer consists of:

- source corpus files:
  - `kaenguru_2021_2025_no_diagram_finetune.json`
  - `kaenguru_2021_2023_no_diagram_finetune.json`
  - `test_2025.jsonl`
- row-level 2025 evaluation outputs in `kaggle_outputs/kaggle_kernel_outputs/`
- row-level 2026 evaluation outputs in `kaggle_outputs/qwen3_8b_eval_2026/`
- OpenRouter unchanged-model baseline outputs in
  `kaggle_outputs/openrouter_qwen3_8b_pretrain_baseline_max4096_2025_2026/`
- preserved training-loss histories in `training_artifacts/training_logs/`
- preserved checkpoint-selection summaries in
  `training_artifacts/checkpoint_selection/`
- compact manifests in `metadata/`
- analysis scripts in `scripts/`

This repository should therefore be read as an artifact-backed reproducibility
archive. It is suitable for rebuilding the final reported summaries from the
preserved evaluation outputs and manifests. It is not a full public export of
every intermediate local training package used during development.

The accompanying Kaggle archive stores the larger artifact bundle that is
impractical to keep directly in GitHub. In particular, the Kaggle archive is the
place to look for bulk `generated_files/` packages, condition-specific SFT data
packages, and final LoRA adapter artifacts where available.

## 1. Rebuild the final aggregate result tables

From the repository root:

```bash
python3 scripts/build_qwen3_8b_final_results.py
```

This script rebuilds the final result summaries from the preserved 2025 Kaggle
evaluation outputs. If the non-public supervision-target JSONLs are absent, the
teacher-length rows in the answer-length comparison are left as `n/a`.

## 2. Rebuild the deep analysis

```bash
python3 scripts/build_qwen3_8b_deep_analysis.py
```

This script rebuilds the post-hoc condition summaries and failure analyses from
the preserved public artifact layer. The full
teacher-student linkage tables require the additional teacher-side annotation
bundles that are not shipped in the public clone.

## 3. Rebuild the verbosity analysis

```bash
python3 scripts/build_verbosity_analysis.py
```

This script rebuilds the verbosity-focused summaries from:

- the preserved 2025 evaluation outputs
- the preserved 2026 evaluation outputs
- the optional supervision-target layer, if those JSONLs are available locally

## 4. Inspect the preserved training metadata

Compact training metadata are available in:

- `metadata/training_data_manifest.csv`
- `metadata/training_log_manifest.csv`
- `metadata/checkpoint_selection_manifest.csv`
- `metadata/training_run_manifest.csv`

Per-run histories and checkpoint summaries are available in:

- `training_artifacts/training_logs/`
- `training_artifacts/checkpoint_selection/`

## 5. Inspect the preserved evaluation artifacts directly

The full 2025 seed-by-condition Kaggle evaluation outputs are in:

- `kaggle_outputs/kaggle_kernel_outputs/`

The full 2026 seed-by-condition evaluation outputs are in:

- `kaggle_outputs/qwen3_8b_eval_2026/`

The unchanged-model OpenRouter baseline outputs are in:

- `kaggle_outputs/openrouter_qwen3_8b_pretrain_baseline_max4096_2025_2026/`

## 6. Rebuild the thesis PDF

```bash
cd reasoning-verbosity-sft-math
pdflatex -interaction=nonstopmode main.tex
biber main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Public Kaggle mirror

A public Kaggle copy of the large artifact bundle is available at:

- `https://www.kaggle.com/datasets/tolipp/codvscot-sft-math-repro`

Use the GitHub repository for the readable metadata, scripts, source corpus
files, and row-level evaluation outputs. Use the Kaggle archive for larger
packaged artifacts such as `generated_files/` and final adapter bundles.
