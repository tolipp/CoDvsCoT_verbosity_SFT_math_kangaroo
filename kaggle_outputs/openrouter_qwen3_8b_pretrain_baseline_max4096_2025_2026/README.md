# OpenRouter Qwen3-8B Baseline, 2025 and 2026

This folder contains the complete local artifacts for the OpenRouter baseline runs of `qwen/qwen3-8b` on the 2025 and 2026 Kaenguru evaluation sets.

These runs use the comparable baseline setup discussed for the thesis:

- API: OpenRouter chat completions
- Model: `qwen/qwen3-8b`
- Runs per year: 3
- Seeds: `42`, `1337`, `2024`
- Maximum generation length: `max_tokens=4096`
- Decoding: `temperature=0`, `top_p=1`
- Reasoning mode: disabled with `reasoning.enabled=false`
- Workers: `4`

This folder is intentionally separate from older OpenRouter baseline outputs that used a shorter token budget. Do not mix those older 1024-token results with this `max4096` matrix.

## Input Files

- 2025: `generated_files/kaggle_qwen3_8b_eval_seed42_flat/test_2025_prompt_answer.jsonl`
- 2026: `kaggle_outputs/qwen3_8b_eval_2026/support/test_2026_prompt_answer.jsonl`

The 2025 file contains 71 problems. The 2026 file contains 69 problems.

## Result Summary

| Year | Seed | Correct | Total | Strict accuracy | Non-extractable answers |
|---:|---:|---:|---:|---:|---:|
| 2025 | 42 | 29 | 71 | 40.8% | 0 |
| 2025 | 1337 | 33 | 71 | 46.5% | 1 |
| 2025 | 2024 | 35 | 71 | 49.3% | 0 |
| 2026 | 42 | 27 | 69 | 39.1% | 2 |
| 2026 | 1337 | 34 | 69 | 49.3% | 0 |
| 2026 | 2024 | 28 | 69 | 40.6% | 1 |

Aggregate strict accuracy:

| Year | Mean | SD | Min | Max |
|---:|---:|---:|---:|---:|
| 2025 | 45.5% | 4.3 | 40.8% | 49.3% |
| 2026 | 43.0% | 5.5 | 39.1% | 49.3% |

## Files

- `condition_seed*_openrouter_qwen3_8b_test_*_max4096.jsonl`: row-level generations, predictions, gold answers, correctness flags, finish reasons, and usage metadata.
- `condition_seed*_openrouter_qwen3_8b_test_*_max4096.summary.json`: one summary per run.
- `run_matrix_summary.csv` and `run_matrix_summary.json`: derived per-run overview.
- `year_aggregate_summary.csv` and `year_aggregate_summary.json`: derived aggregate statistics by year.
- `artifact_manifest.json`: reproducibility manifest with settings, inputs, outputs, and run IDs.
- `run_matrix.log`: shell-level run log.

## Reproduction Note

The runs were made through the OpenRouter API, not through the local Hugging Face/Kaggle evaluator. They are useful as a comparable API baseline across 2025 and 2026, but they are not byte-identical to local Kaggle inference because OpenRouter wraps the request as a chat completion.

To reproduce, set `OPENROUTER_API_KEY` in the environment and run `scripts/run_openrouter_qwen3_8b_pretrain_baseline.py` with the same input files, seeds, and settings listed above.
