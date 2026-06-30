# Reproduction Guide

## 1. Rebuild the RunPod-aligned training snapshot

From the repository root:

```bash
python3 scripts/build_runpod_aligned_qwen3_32b_package.py
```

This refreshes:

- `generated_files/kaggle_qwen3_32b_5_conditions_2021_2023/data/`
- `metadata/RUNPOD_ALIGNED_PACKAGE.json`
- `metadata/RUNPOD_ALIGNED_PACKAGE.md`

Use this layer when reproducing the preserved thesis-result runs.

**Teacher / recovery policy.** All five condition files are built from the
qwen3-32b teacher annotations only. Rows missing from the preserved snapshot are
recovered exclusively from same-condition qwen3-32b traces in
`generated_files/dataset_answers_qwen_qwen3_32b*`; `repair_sft_with_openrouter.py`
(gpt-5.1) is **not** used for this build, so the rebuild is deterministic and
needs no API access. The training files retain the teacher's answer-only rows
(e.g. bare `Antwort: D`). Reported reasoning-length medians
(259 / 517 / 186 / 383 training tokens for concise solve / verbose solve /
concise rewrite / verbose rewrite) are computed over the non-answer-only traces;
including the answer-only rows the medians are 252 / 511 / 122 / 358.

## 2. Optional: rebuild the stricter canonical release layer

```bash
python3 scripts/build_release_qwen3_32b_package.py
python3 scripts/validate_refreshed_package.py
```

This refreshes:

- `generated_files/kaggle_qwen3_32b_5_conditions_2021_2023_canonical_refreshed/`
- `metadata/AUTHORITATIVE_RELEASE_MANIFEST.json`
- `metadata/REFRESHED_PACKAGE_VALIDATION.md`
- `metadata/REFRESHED_CONDITION_LEAKAGE_AUDIT.md`

## 3. Rebuild the final aggregate thesis tables

```bash
python3 scripts/build_qwen3_8b_final_results.py
```

Expected outputs include:

- `results_so_far/qwen3_8b_eval_snapshot_2026-05-24_final/tables/post_sft_results_complete.csv`
- `results_so_far/qwen3_8b_eval_snapshot_2026-05-24_final/tables/post_sft_condition_aggregates.csv`
- `results_so_far/qwen3_8b_eval_snapshot_2026-05-24_final/tables/pairwise_deltas_vs_condition1.csv`
- `results_so_far/qwen3_8b_eval_snapshot_2026-05-24_final/tables/answer_length_comparison.csv`

## 4. Rebuild the deep analysis

```bash
python3 scripts/build_qwen3_8b_deep_analysis.py
```

Expected outputs include:

- `results_so_far/qwen3_8b_deep_analysis_2026-05-24/tables/student_condition_behavior_summary.csv`
- `results_so_far/qwen3_8b_deep_analysis_2026-05-24/tables/teacher_annotation_summary.csv`
- `results_so_far/qwen3_8b_deep_analysis_2026-05-24/tables/failure_tag_counts.csv`
- `results_so_far/qwen3_8b_deep_analysis_2026-05-24/tables/student_teacher_all_tag_linkage.csv`

## 5. Rebuild the verbosity analysis

```bash
python3 scripts/build_verbosity_analysis.py
```

Expected outputs include:

- `results_so_far/verbosity_analysis_2026-06-27/tables/verbosity_metrics.csv`
- `results_so_far/verbosity_analysis_2026-06-27/tables/verbosity_record_level.csv`
- `results_so_far/verbosity_analysis_2026-06-27/README.md`

This package includes:

- `2025` evaluation outputs
- refreshed `2026` evaluation outputs
- the five SFT target files

Included metrics:

- `max_token_hit_rate`
- `mean_word_tokens`
- `TTR`
- `steps`
- `tokens_per_step`

`MATTR50` is intentionally excluded from the final reproducibility package.

## 6. Rebuild the thesis PDF

```bash
cd reasoning-verbosity-sft-math
pdflatex -interaction=nonstopmode main.tex
biber main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## 7. Inspect preserved training artifacts

Available preserved final training artifacts:

- `training_artifacts/final_adapters/`

Summaries:

- `metadata/training_run_manifest.csv`
- `metadata/training_log_manifest.csv`
- `metadata/checkpoint_selection_manifest.csv`
- `metadata/training_data_manifest.csv`

Per-run extracted loss histories:

- `training_artifacts/training_logs/`

Checkpoint-step summaries:

- `training_artifacts/checkpoint_selection/`

## 8. Inspect evaluation artifacts directly

The full seed-by-condition evaluation artifacts are in:

- `kaggle_outputs/kaggle_kernel_outputs/`

The `2026` seed-by-condition evaluation artifacts are in:

- `kaggle_outputs/qwen3_8b_eval_2026/`

The unchanged-model OpenRouter baseline outputs are in:

- `kaggle_outputs/openrouter_qwen3_8b_pretrain_baselines/`

The Kaggle student-evaluation implementation is in:

- `generated_files/kaggle_qwen3_8b_sft_design_1_3a_3b/scripts/evaluate_base_vs_finetuned_zip.py`

That evaluator documents the deterministic decoding setup, including `do_sample=False` and the stopping criterion that halts once an `Antwort: X` line is produced.
