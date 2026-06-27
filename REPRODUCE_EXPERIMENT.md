# Reproduction Guide

1. Rebuild the RunPod-aligned training snapshot.

From the repository root:

bash python3 scripts/build_runpod_aligned_qwen3_32b_package.py


This refreshes:

- generated_files/kaggle_qwen3_32b_5_conditions_2021_2023/data/
- metadata/RUNPOD_ALIGNED_PACKAGE.json
- metadata/RUNPOD_ALIGNED_PACKAGE.md

Use this layer when reproducing the preserved thesis-result runs.

Teacher/recovery policy: all five condition files are built from the qwen3-32b teacher annotations only. Missing rows from the preserved snapshot are recovered exclusively from the same condition qwen3-32b traces in the 'generated_files/dataset_answers_qwen_qwen3_32b*' folder. 'repair_sft_with_openrouter.py' (gpt-5.1) is not used for this build, so the rebuild is deterministic and needs no API access. The training files retain the teacher's answer-only rows (e.g. bare 'Antwort: D'). The reported reasoning-length medians (259/517/186/383 training tokens for concise solve/verbose solve/concise rewrite/verbose rewrite) are computed over the non-answer-only traces. Including the answer-only rows, the medians are 252/511/122/358.

## 2. Optional: Rebuild the stricter canonical release layer.

Run the following commands:

bash
python3 scripts/build_release_qwen3_32b_package.py
python3 scripts/validate_refreshed_package.py


This refreshes:

- generated_files/kaggle_qwen3_32b_5_conditions_2021_2023_canonical_refreshed/
- metadata/AUTHORITATIVE_RELEASE_MANIFEST.json
- metadata/REFRESHED_PACKAGE_VALIDATION.md
- metadata/REFRESHED_CONDITION_LEAKAGE_AUDIT.md

## 3. Rebuild the final aggregate thesis tables.

bash python3 scripts/build_qwen3_8b_final_results.py

Expected outputs include:

- results_so_far/qwen3_8b_eval_snapshot_2026-05-24_final/tables/post_sft_condition_aggregates.csv
- results_so_far/qwen3_8b_eval_snapshot_2026-05-24_final/tables/post_sft_condition_aggregates.csv
- results_so_far/qwen3_8b_eval_snapshot_2026-05-24_final/tables/pairwise_deltas_vs_condition1.csv
- re

This refreshes:

- generated_files/kaggle_qwen3_32b_5_conditions_2021_2023_canonical_refreshed/
- metadata/AUTHORITATIVE_RELEASE_MANIFEST.json
- metadata/REFRESHED_PACKAGE_VALIDATION.md
- metadata/REFRESHED_CONDITION_LEAKAGE_AUDIT.md

Expected outputs include:



- results_so_far/qwen3_8b_eval_snapshot_2026-05-24_final/tables/post_sft_results_complete.csv

- results_so_far/qwen3_8b_eval_snapshot_2026-05-24_final/tables/post_sft_condition_aggregates.csv

- results_so_far/qwen3_8b_eval_snapshot_2026-05-24_final/tables/pairwise_deltas_vs_condition1.csv

- results_so_far/qwen3_8b_eval_snapshot_2026-05-24_final/tables/answer_length_comparison.csv



4. Rebuild the deep analysis



bash

python3 scripts/build_qwen3_8b_deep_analysis.py


Expected outputs include:



- results_so_far/qwen3_8b_deep_analysis_2026-05-24/tables/student_condition_behaviour_summary.csv

- results_so_far/qwen3_8b_deep_analysis_2026-05-24/tables/teacher_annotation_summary.csv

- results_so_far/qwen3_8b_deep_analysis_2026-05-24/tables/failure_tag_counts.csv

- results_so_far/qwen3_8b_deep_analysis_2026-05-24/tables/student_teacher_all_tag_linkage.csv



5. Rebuild the thesis PDF.



bash

cd reasoning-verbosity-sft-math

pdflatex -interaction=nonstopmode main.tex

biber main.

pdflatex -interaction=nonstopmode main.tex

pdflatex -interaction=nonstopmode main.tex

## 6.



6. Inspect the preserved training artefacts.



Available preserved final training artefacts:



- training_artifacts/final_adapters/



Summaries:



- metadata/training_run_manifest.csv

- training_log_manifest.csv

- `metadata/checkpoint_selection_manifest.csv`

- metadata/training_data_manifest.csv



Per-run extracted loss histories:



- training_artifacts/training_logs/



Checkpoint-step summaries:



- training_artifacts/checkpoint_selection/



7. Inspect the evaluation artefacts directly.



The full seed-by-condition evaluation artefacts are in:



- kaggle_outputs/kaggle_kernel_outputs/.



The unchanged model OpenRouter baseline outputs are in:



- kaggle_outputs/openrouter_qwen3_8b_pretrain_baselines/.



The Kaggle student evaluation implementation is in:





- generated_files/kaggle_qwen3_8b_sft_design_1_3a_3b/scripts/evaluate_base_vs_finetuned.zip.py



This evaluator documents the deterministic decoding setup, including the setting of do_sample to False and the stopping criterion that halts once an answer is found. X' line is produced.


