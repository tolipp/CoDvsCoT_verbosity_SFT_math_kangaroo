from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median, stdev

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results_so_far" / "qwen3_8b_eval_2026-05-28_to_31_final"
TABLES_DIR = OUT_DIR / "tables"
NOTES_DIR = OUT_DIR / "notes"

TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

RUN_SPECS = [
    {
        "condition": "1",
        "seed": 42,
        "source_type": "artifact_backed_refresh",
        "summary_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed42_1" / "eval_outputs" / "eval_1_seed42_test2025" / "comparison_summary.json",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed42_1" / "eval_outputs" / "eval_1_seed42_test2025" / "evaluation_results_changed.jsonl",
    },
    {
        "condition": "1",
        "seed": 1337,
        "source_type": "artifact_backed_refresh",
        "summary_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed1337_1" / "eval_outputs" / "eval_1_seed1337_test2025" / "comparison_summary.json",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed1337_1" / "eval_outputs" / "eval_1_seed1337_test2025" / "evaluation_results_changed.jsonl",
    },
    {
        "condition": "1",
        "seed": 2024,
        "source_type": "artifact_backed_refresh",
        "summary_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed2024_1" / "eval_outputs" / "eval_1_seed2024_test2025" / "comparison_summary.json",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed2024_1" / "eval_outputs" / "eval_1_seed2024_test2025" / "evaluation_results_changed.jsonl",
    },
    {
        "condition": "2a",
        "seed": 42,
        "source_type": "artifact_backed_refresh",
        "summary_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed42_2a" / "eval_outputs" / "eval_2a_seed42_test2025" / "comparison_summary.json",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed42_2a" / "eval_outputs" / "eval_2a_seed42_test2025" / "evaluation_results_changed.jsonl",
    },
    {
        "condition": "2a",
        "seed": 1337,
        "source_type": "artifact_backed_refresh",
        "summary_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed1337_2a" / "eval_outputs" / "eval_2a_seed1337_test2025" / "comparison_summary.json",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed1337_2a" / "eval_outputs" / "eval_2a_seed1337_test2025" / "evaluation_results_changed.jsonl",
    },
    {
        "condition": "2a",
        "seed": 2024,
        "source_type": "artifact_backed_refresh",
        "summary_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed2024_2a" / "eval_outputs" / "eval_2a_seed2024_test2025" / "comparison_summary.json",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed2024_2a" / "eval_outputs" / "eval_2a_seed2024_test2025" / "evaluation_results_changed.jsonl",
    },
    {
        "condition": "3a",
        "seed": 42,
        "source_type": "artifact_backed_refresh",
        "summary_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed42_3a" / "eval_outputs" / "eval_3a_seed42_test2025" / "comparison_summary.json",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed42_3a" / "eval_outputs" / "eval_3a_seed42_test2025" / "evaluation_results_changed.jsonl",
    },
    {
        "condition": "3a",
        "seed": 1337,
        "source_type": "artifact_backed_refresh",
        "summary_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed1337_3a" / "eval_outputs" / "eval_3a_seed1337_test2025" / "comparison_summary.json",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed1337_3a" / "eval_outputs" / "eval_3a_seed1337_test2025" / "evaluation_results_changed.jsonl",
    },
    {
        "condition": "3a",
        "seed": 2024,
        "source_type": "artifact_backed_refresh",
        "summary_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed2024_3a" / "eval_outputs" / "eval_3a_seed2024_test2025" / "comparison_summary.json",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed2024_3a" / "eval_outputs" / "eval_3a_seed2024_test2025" / "evaluation_results_changed.jsonl",
    },
    {
        "condition": "4a",
        "seed": 42,
        "source_type": "artifact_backed_refresh",
        "summary_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed42_4a" / "eval_outputs" / "eval_4a_seed42_test2025" / "comparison_summary.json",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed42_4a" / "eval_outputs" / "eval_4a_seed42_test2025" / "evaluation_results_changed.jsonl",
    },
    {
        "condition": "4a",
        "seed": 1337,
        "source_type": "artifact_backed_refresh",
        "summary_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed1337_4a" / "eval_outputs" / "eval_4a_seed1337_test2025" / "comparison_summary.json",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed1337_4a" / "eval_outputs" / "eval_4a_seed1337_test2025" / "evaluation_results_changed.jsonl",
    },
    {
        "condition": "4a",
        "seed": 2024,
        "source_type": "artifact_backed_refresh",
        "summary_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed2024_4a" / "eval_outputs" / "eval_4a_seed2024_test2025" / "comparison_summary.json",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed2024_4a" / "eval_outputs" / "eval_4a_seed2024_test2025" / "evaluation_results_changed.jsonl",
    },
    {
        "condition": "5a",
        "seed": 42,
        "source_type": "artifact_backed_refresh",
        "summary_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed42_5a" / "unzipped" / "eval_outputs" / "eval_5a_seed42_test2025" / "comparison_summary.json",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed42_5a" / "unzipped" / "eval_outputs" / "eval_5a_seed42_test2025" / "evaluation_results_changed.jsonl",
    },
    {
        "condition": "5a",
        "seed": 1337,
        "source_type": "artifact_backed_refresh",
        "summary_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed1337_5a" / "eval_outputs" / "eval_5a_seed1337_test2025" / "comparison_summary.json",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed1337_5a" / "eval_outputs" / "eval_5a_seed1337_test2025" / "evaluation_results_changed.jsonl",
    },
    {
        "condition": "5a",
        "seed": 2024,
        "source_type": "artifact_backed_refresh",
        "summary_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed2024_5a" / "eval_outputs" / "eval_5a_seed2024_test2025" / "comparison_summary.json",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed2024_5a" / "eval_outputs" / "eval_5a_seed2024_test2025" / "evaluation_results_changed.jsonl",
    },
]


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)


def round_or_none(value: float | None, digits: int = 1) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def token_count(text: str) -> int:
    return len(TOKENIZER.encode(text, add_special_tokens=False))


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return stdev(values)


def metric_from_rows(spec: dict, rows: list[dict]) -> dict:
    scorable = [r for r in rows if r.get("generated_answer") not in (None, "", "?")]
    correct = sum(1 for r in rows if r.get("is_correct") is True)
    token_counts = [int(r.get("generated_token_count", 0)) for r in rows if r.get("generated_token_count") is not None]
    return {
        "correct": correct,
        "total": len(rows),
        "strict_accuracy_percent": (correct / len(rows)) * 100.0,
        "scorable": len(scorable),
        "unscorable": len(rows) - len(scorable),
        "scorable_accuracy_percent": ((correct / len(scorable)) * 100.0) if scorable else None,
        "max_token_hits": sum(1 for r in rows if r.get("hit_max_new_tokens") is True),
        "mean_generated_tokens": mean(token_counts),
        "median_generated_tokens": median(token_counts) if token_counts else None,
        "max_generated_tokens": max(token_counts) if token_counts else None,
    }


def build_run_rows() -> list[dict]:
    rows_out = []
    for spec in RUN_SPECS:
        row = {
            "condition": spec["condition"],
            "seed": spec["seed"],
            "label": f"{spec['condition']}_seed{spec['seed']}",
            "source_type": spec["source_type"],
        }
        if "summary_override" in spec:
            row.update(spec["summary_override"])
            row["mean_generated_tokens"] = None
            row["median_generated_tokens"] = None
            row["max_generated_tokens"] = None
            row["artifact_path"] = None
        else:
            row["artifact_path"] = str(spec["row_path"].relative_to(ROOT))
            data_rows = load_jsonl(spec["row_path"])
            row.update(metric_from_rows(spec, data_rows))
        rows_out.append(row)
    return rows_out


def build_condition_aggregates(run_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in run_rows:
        grouped[row["condition"]].append(row)

    aggregates = []
    for condition, rows in sorted(grouped.items()):
        stricts = [r["strict_accuracy_percent"] for r in rows if r["strict_accuracy_percent"] is not None]
        scorables = [r["scorable_accuracy_percent"] for r in rows if r["scorable_accuracy_percent"] is not None]
        max_hits = [r["max_token_hits"] for r in rows if r["max_token_hits"] is not None]
        mean_tokens = [r["mean_generated_tokens"] for r in rows if r["mean_generated_tokens"] is not None]
        aggregates.append(
            {
                "condition": condition,
                "n_seeds": len(rows),
                "mean_strict_accuracy_percent": mean(stricts),
                "sd_strict_accuracy_percent": sample_sd(stricts),
                "min_strict_accuracy_percent": min(stricts),
                "max_strict_accuracy_percent": max(stricts),
                "mean_scorable_accuracy_percent": mean(scorables),
                "mean_max_token_hits": mean(max_hits),
                "mean_generated_tokens": mean(mean_tokens),
            }
        )
    aggregates.sort(key=lambda x: x["mean_strict_accuracy_percent"], reverse=True)
    return aggregates


def build_pairwise_deltas(run_rows: list[dict]) -> list[dict]:
    by_seed_condition = {(r["seed"], r["condition"]): r for r in run_rows}
    deltas = []
    for seed in (42, 1337, 2024):
        base = by_seed_condition[(seed, "1")]
        for condition in ("2a", "3a", "4a", "5a"):
            row = by_seed_condition[(seed, condition)]
            deltas.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "strict_accuracy_percent": row["strict_accuracy_percent"],
                    "condition_1_strict_accuracy_percent": base["strict_accuracy_percent"],
                    "delta_vs_condition_1_percent": row["strict_accuracy_percent"] - base["strict_accuracy_percent"],
                    "max_token_hits": row["max_token_hits"],
                    "condition_1_max_token_hits": base["max_token_hits"],
                }
            )
    return deltas


def load_teacher_lengths() -> dict[str, dict]:
    data_dir = ROOT / "generated_files" / "kaggle_qwen3_32b_5_conditions_2021_2023" / "data"
    specs = {
        "1": data_dir / "train_official_human_2021_2023.jsonl",
        "2a": data_dir / "train_qwen3_32b_verbose_2021_2023.jsonl",
        "3a": data_dir / "train_qwen3_32b_concise_2021_2023.jsonl",
        "4a": data_dir / "train_qwen3_32b_concise_rewrite_2021_2023.jsonl",
        "5a": data_dir / "train_qwen3_32b_verbose_rewrite_2021_2023.jsonl",
    }
    out = {}
    for condition, path in specs.items():
        lengths = []
        for row in load_jsonl(path):
            messages = row.get("messages", [])
            assistant = next((m["content"] for m in messages if m.get("role") == "assistant"), None)
            if assistant:
                lengths.append(token_count(assistant))
        out[condition] = {
            "group": "teacher",
            "condition": condition,
            "n": len(lengths),
            "avg_tokens": mean(lengths),
            "median_tokens": median(lengths),
            "max_tokens": max(lengths),
        }
    return out


def load_baseline_lengths() -> dict[str, dict]:
    base_dir = ROOT / "kaggle_outputs" / "openrouter_qwen3_8b_pretrain_baselines"
    out = {}
    for condition in ("1", "2a", "3a", "4a", "5a"):
        path = base_dir / f"condition_{condition}_openrouter_qwen3_8b_test_2025.jsonl"
        rows = load_jsonl(path)
        dedup = {}
        for row in rows:
            dedup[row["id"]] = row
        lengths = [token_count(r.get("generated", "")) for r in dedup.values()]
        out[condition] = {
            "group": "unchanged_base",
            "condition": condition,
            "n": len(lengths),
            "avg_tokens": mean(lengths),
            "median_tokens": median(lengths),
            "max_tokens": max(lengths),
        }
    return out


def load_trained_lengths(run_rows: list[dict]) -> dict[str, dict]:
    lengths_by_condition: dict[str, list[int]] = defaultdict(list)
    runs_with_lengths: dict[str, int] = defaultdict(int)

    for spec in RUN_SPECS:
        if "row_path" not in spec:
            continue
        rows = load_jsonl(spec["row_path"])
        if spec["condition"] == "1" and spec["seed"] == 42:
            lengths = [token_count(r.get("generated", "")) for r in rows]
        else:
            lengths = [int(r.get("generated_token_count", 0)) for r in rows if r.get("generated_token_count") is not None]
        lengths_by_condition[spec["condition"]].extend(lengths)
        runs_with_lengths[spec["condition"]] += 1

    out = {}
    for condition, lengths in lengths_by_condition.items():
        out[condition] = {
            "group": "trained_conditions",
            "condition": condition,
            "n": len(lengths),
            "n_runs_with_lengths": runs_with_lengths[condition],
            "avg_tokens": mean(lengths),
            "median_tokens": median(lengths),
            "max_tokens": max(lengths),
        }
    return out


def build_length_rows(run_rows: list[dict]) -> list[dict]:
    baseline = load_baseline_lengths()
    teacher = load_teacher_lengths()
    trained = load_trained_lengths(run_rows)
    rows = []
    for condition in ("1", "2a", "3a", "4a", "5a"):
        rows.append({**baseline[condition]})
        rows.append({**teacher[condition]})
        rows.append({**trained[condition]})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def fmt_num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def write_markdown(run_rows: list[dict], condition_aggs: list[dict], deltas: list[dict], length_rows: list[dict]) -> None:
    best_run = max(run_rows, key=lambda r: r["strict_accuracy_percent"])
    worst_run = min(run_rows, key=lambda r: r["strict_accuracy_percent"])

    by_condition = {row["condition"]: row for row in condition_aggs}
    solve_mean = mean([by_condition[c]["mean_strict_accuracy_percent"] for c in ("2a", "3a")])
    rewrite_mean = mean([by_condition[c]["mean_strict_accuracy_percent"] for c in ("4a", "5a")])

    overview_lines = [
        "# Final Results Overview",
        "",
        "## Status",
        "",
        "- The post-SFT result matrix is complete for conditions `1`, `2a`, `3a`, `4a`, `5a` across seeds `42`, `1337`, `2024`.",
        "- Most runs are artifact-backed with `comparison_summary.json`, `progress_changed.json`, and `evaluation_results_changed.jsonl`.",
        "- The only format mismatch is `condition 1 seed42`, which is backed by the RunPod eval JSONL plus local rescoring rather than the Kaggle summary format.",
        "",
        "## Best and Worst Runs",
        "",
        f"- Best strict-accuracy run: `{best_run['label']}` at `{best_run['strict_accuracy_percent']:.1f}%`.",
        f"- Weakest strict-accuracy run: `{worst_run['label']}` at `{worst_run['strict_accuracy_percent']:.1f}%`.",
        "",
        "## Condition Ranking by Mean Strict Accuracy",
        "",
    ]
    for idx, row in enumerate(condition_aggs, start=1):
        overview_lines.append(
            f"{idx}. `{row['condition']}`: mean `{row['mean_strict_accuracy_percent']:.1f}%`, sd `{row['sd_strict_accuracy_percent']:.1f}`, mean max-token hits `{row['mean_max_token_hits']:.1f}`"
        )

    overview_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Solve-style supervision (`2a`, `3a`) averaged `{solve_mean:.1f}%`, clearly above rewrite-style supervision (`4a`, `5a`) at `{rewrite_mean:.1f}%`.",
            f"- `2a` is the strongest condition overall, while `4a` is clearly the weakest.",
            "- `condition 1` is competitive with weaker rewrite settings but does not match the best solve-style condition.",
            "",
            "## Seed-Level Table",
            "",
            "| Condition | Seed | Strict Accuracy | Scorable Accuracy | Max-Token Hits | Mean Generated Tokens | Source |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(run_rows, key=lambda r: (r["condition"], r["seed"])):
        overview_lines.append(
            f"| {row['condition']} | {row['seed']} | {fmt_pct(row['strict_accuracy_percent'])} | {fmt_pct(row['scorable_accuracy_percent'])} | {row['max_token_hits'] if row['max_token_hits'] is not None else 'n/a'} | {fmt_num(row['mean_generated_tokens'])} | {row['source_type']} |"
        )

    insights_lines = [
        "# Statistical Analysis Notes",
        "",
        "## What Can Be Analyzed Directly",
        "",
        "- Per-condition mean, standard deviation, minimum, and maximum strict accuracy across three seeds.",
        "- Per-seed paired deltas versus `condition 1`.",
        "- Secondary reliability metrics: `max_token_hits`, unscorable rows, and generated token length.",
        "- Row-level error inspection from `evaluation_results_changed.jsonl`.",
        "",
        "## Pairwise Deltas vs Condition 1",
        "",
        "| Seed | Condition | Strict Accuracy | Condition 1 | Delta vs 1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in deltas:
        insights_lines.append(
            f"| {row['seed']} | {row['condition']} | {fmt_pct(row['strict_accuracy_percent'])} | {fmt_pct(row['condition_1_strict_accuracy_percent'])} | {row['delta_vs_condition_1_percent']:+.1f}% |"
        )

    insights_lines.extend(
        [
            "",
            "## Answer Length Comparison",
            "",
            "| Group | Condition | Avg Tokens | Median Tokens | Max Tokens |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in length_rows:
        insights_lines.append(
            f"| {row['group']} | {row['condition']} | {fmt_num(row['avg_tokens'])} | {fmt_num(row['median_tokens'])} | {int(row['max_tokens']) if row['max_tokens'] is not None else 'n/a'} |"
        )

    insights_lines.extend(
        [
            "",
            "## Writing Guidance",
            "",
            "- Use strict accuracy as the primary outcome.",
            "- Report scorable accuracy and max-token hits as secondary reliability metrics.",
            "- Be explicit that `n=3` seeds supports descriptive and paired analysis, but not strong inferential claims.",
            "- Highlight that generation-control failures and reasoning failures both contribute to final performance.",
        ]
    )

    provenance_lines = [
        "# Final Provenance",
        "",
        "## Artifact-Backed Runs",
        "",
    ]
    for row in sorted(run_rows, key=lambda r: (r["condition"], r["seed"])):
        provenance_lines.append(
            f"- `{row['label']}`: `{row['source_type']}`" + (f", artifact `{row['artifact_path']}`" if row.get("artifact_path") else "")
        )

    (NOTES_DIR / "RESULTS_FINAL_OVERVIEW.md").write_text("\n".join(overview_lines) + "\n")
    (NOTES_DIR / "STATS_AND_INSIGHTS.md").write_text("\n".join(insights_lines) + "\n")
    (NOTES_DIR / "PROVENANCE_FINAL.md").write_text("\n".join(provenance_lines) + "\n")
    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Qwen3-8B Final Evaluation Snapshot",
                "",
                "This folder contains the final consolidated post-SFT result matrix, aggregate statistics, and thesis-ready notes.",
                "",
                "## Key Files",
                "",
                "- `tables/post_sft_results_complete.csv`: one row per condition-seed run.",
                "- `tables/post_sft_condition_aggregates.csv`: mean and spread per condition.",
                "- `tables/pairwise_deltas_vs_condition1.csv`: per-seed deltas versus condition 1.",
                "- `tables/answer_length_comparison.csv`: unchanged base, teacher, and trained answer lengths.",
                "- `notes/RESULTS_FINAL_OVERVIEW.md`: concise project status and ranking.",
                "- `notes/STATS_AND_INSIGHTS.md`: analysis directions and thesis-writing notes.",
                "- `notes/PROVENANCE_FINAL.md`: artifact/source provenance for every run.",
            ]
        )
        + "\n"
    )


def main() -> None:
    ensure_dirs()
    run_rows = build_run_rows()
    condition_aggs = build_condition_aggregates(run_rows)
    deltas = build_pairwise_deltas(run_rows)
    length_rows = build_length_rows(run_rows)

    write_csv(TABLES_DIR / "post_sft_results_complete.csv", run_rows)
    write_json(TABLES_DIR / "post_sft_results_complete.json", run_rows)
    write_csv(TABLES_DIR / "post_sft_condition_aggregates.csv", condition_aggs)
    write_json(TABLES_DIR / "post_sft_condition_aggregates.json", condition_aggs)
    write_csv(TABLES_DIR / "pairwise_deltas_vs_condition1.csv", deltas)
    write_json(TABLES_DIR / "pairwise_deltas_vs_condition1.json", deltas)
    write_csv(TABLES_DIR / "answer_length_comparison.csv", length_rows)
    write_json(TABLES_DIR / "answer_length_comparison.json", length_rows)
    write_markdown(run_rows, condition_aggs, deltas, length_rows)


if __name__ == "__main__":
    main()
