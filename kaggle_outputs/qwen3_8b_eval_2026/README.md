# 2026 Evaluation Artifacts

This directory contains the merged Kaggle evaluation outputs for the `2026` held-out set.

Contents:

- `eval_*_test2026/`: merged per-seed evaluation outputs for all 15 runs
- `support/test_2026_prompt_answer.jsonl`: the evaluation prompt file used in the Kaggle runs
- `support/test_2026_prompt_answer_clean_source.jsonl`: optional source-cleaned variant
- `support/test_2026_source_records.jsonl`: source-side supporting records

Each run directory contains:

- `evaluation_results_changed.jsonl`
- `comparison_summary.json`
- `progress_changed.json`

The artifacts were copied from the downloaded Kaggle outputs so that the reproducibility package contains both the `2025` and `2026` evaluation layers.
