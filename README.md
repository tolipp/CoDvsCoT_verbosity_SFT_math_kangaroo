# CoDvsCoT_verbosity_SFT_math_kangaroo
## The Effect of Reasoning Verbosity during Supervised Fine-Tuning on Mathematical Problem-Solving Performance

#ReasoningVerbositySFTMathReproducibility
Reproducibility repository for the bachelor's thesis:
The Effect of Reasoning Verbosity during Supervised Fine-Tuning on Mathematical Problem-Solving Performance
The repository now exposes two Qwen3-32B supervision layers.
- a RunPod-aligned training snapshot that matches the preserved student-run metadata as closely as possible given the saved local artefacts.
- a stricter canonical rebuild that retains only the source-recoverable refreshed rows.
Primary Thesis Reproduction Layer
The default training snapshot for reproducing the reported Qwen3-8B student runs is:
generated_files/kaggle_qwen3_32b_5_conditions_2021_2023/data/.
Its current row counts are:
- train_official_human_2021_2023.jsonl: 185
- train_qwen3_32b_verbose_2021_2023.jsonl: 183
- train_qwen3_32b_concise_2021_2023.jsonl: 186
- train_qwen3_32b_concise_rewrite_2021_2023.jsonl: 186
- train_qwen3_32b_verbose_rewrite_2021_2023.jsonl: 186
- valid_2024.jsonl: 67
- val_2a.jsonl: 46
- val_3a.jsonl: 63
- val_4a.jsonl: 67
- val_5a.jsonl: 67
- test_2025.jsonl: 71
An additional official-human variant is included for the later seed-42 rerun.
train_official_human_2021_2023_seed42_full_186.jsonl: 186
The recovery summary for this layer can be found in:
- metadata/RUNPOD_ALIGNED_PACKAGE.json
- metadata/RUNPOD_ALIGNED_PACKAGE.md
- metadata/training_run_manifest.csv
Secondary Clean Release Layer
The stricter source-rebuilt package is kept separately at:
- generated_files/kaggle_qwen3_32b_5_conditions_2021_2023_canonical_refreshed/.
This layer is useful when a cleaner public release package is required. However, it is not the closest match to the preserved RunPod training metadata. Its validation files are documented in:
- metadata/AUTHORITATIVE_RELEASE_MANIFEST.json
- metadata/REFRESHED_PACKAGE_VALIDATION.md
- metadata/REFRESHED_CONDITION_LEAKAGE_AUDIT.md
Preserved Student Artifacts
The repository preserves:
- final LoRA adapter
- for all 15 student runs can be found in the 'training_artifacts/final_adapters' directory.
- extracted training loss histories in training_artifacts/training_logs/.
- Checkpoint selection summaries in training_artifacts/checkpoint_selection/.
- Full seed-by-condition evaluation artefacts in kaggle_outputs/kaggle_kernel_outputs/.
Rebuilt thesis result tables in 'results_so_far/qwen3_8b_eval_snapshot_2026-05-24_final'.
- Deep analysis tables in results_so_far/qwen3_8b_deep_analysis_2026-05-24/.
Condition mapping
Preferred condition names:
- Official Human Solution
- verbose solution
- Concise Solve
- Concise rewrite
- verbose rewrite
Historical artefact labels:
- 1 = official human solution
- 2a = verbose solution
- 3a = concise solution
- `4a` = concise rewrite
- 5a = verbose rewrite
## Core study setup
Student model: Qwen/Qwen3-8B
- Teacher model: Qwen/Qwen3-32B
Task: German Kangaroo mathematics multiple-choice problems
Source corpus file: kaenguru_2021_2025_no_diagram_finetune.json
Source split sizes:
  - Training years: 2021–2023: 186
  - validation year 2024: 67
  - Held-out test year: 2025: 71
Rebuild scripts:
RunPod-aligned snapshot:
  python3 scripts/build_runpod_aligned_qwen3_32b_package.py
- Canonical release:
  - Run `python3 scripts/build_release_qwen3_32b_package.py`.
  - `python3 scripts/validate_refreshed_package.py`
- Thesis result tables:
  - `python3 scripts/build_qwen3_8b_final_results.py`
- Deep analysis:
  - python3 scripts/build_qwen3_8b_deep_analysis.py
Rights: The underlying Math Kangaroo materials remain the intellectual property of Math Kangaroo Switzerland (Mathe Känguru Schweiz). This repository is intended for scientific and academic use only and must not be used for personal profit or other commercial use.
