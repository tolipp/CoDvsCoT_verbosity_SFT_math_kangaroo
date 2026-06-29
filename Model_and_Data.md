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


The main reproduction layer for the thesis results runs is:

- generated_files/kaggle_qwen3_32b_5_conditions_2021_2023/data/.

## Training rows:

- Official human solution: 185
- verbose solution: 186
- Concise solution: 186
- Concise rewrite: 186
Verbosely rewritten: 186

Validation/held-out files:

- valid_2024.jsonl: 67
- val_2a.jsonl: 65
- val_3a.jsonl: 67
- val_4a.jsonl: 67
- val_5a.jsonl: 67
- test_2025.jsonl: 71

  
## Supervision Conditions


- Official human solution: official human-written solution text
- verbose solution: a direct Qwen3-32b teacher solution in a verbose CoT-style format
- Concise Solve: Direct Qwen3-32B Teacher Solve in a Concise CoD-Style Format
- Concise rewrite: QWEN-32B concise rewrite of the official solution
- Verbose Rewrite: Qwen3-32b verbose rewrite of the official solution

Historical labels still used in artefacts:

- `1` Official human solution
- 2a: verbose solution
- 3a: Concise solve
- `4a` Concise rewrite
- 5a: verbose rewrite

## External vs. original material:

Externally sourced:

- Kangaroo problem statements and official solution material
- Pretrained model families:
  - Qwen/Qwen3-8B
  - Qwen/Qwen3-32B

Derived in this project:

- the text-only curated corpus JSON
- Qwen3-32B teacher annotation bundles
- Qwen3-32b rewrite bundles
- condition-specific SFT JSONLs
- evaluation outputs and summaries
- Failure analysis tables and tags
Fine-tuned LoRA adapters preserved locally
- Thesis text, figures and tables



The repository preserves:

The final LoRA adapters for all 15 student runs can be found in the 'training_artifacts/final_adapters/' folder
with 'training_meta.json' and 'trainer_state.json' as metadata and 
logs in the 'training_artifacts/training_logs' folder.
'Training Artifacts' section has also 'Checkpoint Selection summaries'.
- Combined lists in:
  The file is called 'metadata/training_run_manifest.csv'.
  The file is called 'metadata/training_log_manifest.csv'.
  The file is called 'metadata/checkpoint_selection_manifest.csv'.
  The file is called 'metadata/training_data_manifest.csv'.

## Methodological Limits

The study does not look at verbosity as a single factor; it depends on the source type and supervision format.
Only three student seeds are kept.
The test set that was left out has 71 problems in it.
The RunPod-aligned package tries to rebuild the JSONLs that are mentioned in the student-run metadata.
The length of the evaluation output is not the same as the supervised verbosity. Some long student generations come from repetition or max-token behaviour, rather than useful reasoning.

## Note about redistribution:

This repository contains texts and JSONL files created using Kangaroo. You can redistribute the public scientific content, but the competition content is still the intellectual property of Math Kangaroo Switzerland (Mathe Känguru Schweiz). The repository and its data should not be used to make money or for other commercial purposes.
