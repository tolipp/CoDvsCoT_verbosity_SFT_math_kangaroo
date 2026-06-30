#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results_so_far" / "verbosity_analysis_2026-06-27"
TABLES_DIR = OUT_DIR / "tables"

CONDITION_MAP = {
    "1": "official human",
    "2a": "verbose solve",
    "3a": "concise solve",
    "4a": "concise rewrite",
    "5a": "verbose rewrite",
}

SFT_FILES = {
    "official human": ROOT / "generated_files" / "kaggle_qwen3_32b_5_conditions_2021_2023" / "data" / "train_official_human_2021_2023_seed42_full_186.jsonl",
    "verbose solve": ROOT / "generated_files" / "kaggle_qwen3_32b_5_conditions_2021_2023" / "data" / "train_qwen3_32b_verbose_2021_2023.jsonl",
    "concise solve": ROOT / "generated_files" / "kaggle_qwen3_32b_5_conditions_2021_2023" / "data" / "train_qwen3_32b_concise_2021_2023.jsonl",
    "concise rewrite": ROOT / "generated_files" / "kaggle_qwen3_32b_5_conditions_2021_2023" / "data" / "train_qwen3_32b_concise_rewrite_2021_2023.jsonl",
    "verbose rewrite": ROOT / "generated_files" / "kaggle_qwen3_32b_5_conditions_2021_2023" / "data" / "train_qwen3_32b_verbose_rewrite_2021_2023.jsonl",
}


def word_tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def ttr(text: str) -> float:
    toks = word_tokens(text)
    if not toks:
        return 0.0
    return len(set(tok.lower() for tok in toks)) / len(toks)


def step_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    pieces = []
    for block in re.split(r"\n+", text):
        block = block.strip()
        if not block:
            continue
        sub = re.split(r"(?<=[.!?])\s+", block)
        for item in sub:
            item = item.strip()
            if item:
                pieces.append(item)
    return max(1, len(pieces))


def tokens_per_step(text: str) -> float:
    steps = step_count(text)
    if steps == 0:
        return 0.0
    return len(word_tokens(text)) / steps


def condition_sort_key(name: str) -> int:
    order = {
        "official human": 1,
        "concise solve": 2,
        "verbose solve": 3,
        "concise rewrite": 4,
        "verbose rewrite": 5,
    }
    return order[name]


def parse_eval_condition(path: Path) -> str:
    m = re.search(r"eval_(1|2a|3a|4a|5a)_seed\d+_", path.as_posix())
    if not m:
        raise ValueError(f"Could not parse condition from {path}")
    return CONDITION_MAP[m.group(1)]


def load_eval_records(pattern_root: Path, corpus_label: str) -> list[dict]:
    candidates = list(pattern_root.rglob("evaluation_results_changed.jsonl"))
    chosen: dict[str, Path] = {}
    for path in candidates:
        run_dir = path.parent.name
        if "/shard_" in path.as_posix():
            continue
        if "/from_downloads/" in path.as_posix():
            continue
        current = chosen.get(run_dir)
        if current is None or len(path.as_posix()) < len(current.as_posix()):
            chosen[run_dir] = path

    records = []
    for run_name, path in sorted(chosen.items()):
        condition = parse_eval_condition(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("generated_response", "") or ""
                records.append(
                    {
                        "corpus": corpus_label,
                        "condition": condition,
                        "run": run_name,
                        "id": obj.get("id"),
                        "text": text,
                        "max_token_hit": bool(obj.get("hit_max_new_tokens")),
                    }
                )
    return records


def load_sft_records() -> list[dict]:
    records = []
    for condition, path in SFT_FILES.items():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                assistant = ""
                for msg in obj.get("messages", []):
                    if msg.get("role") == "assistant":
                        assistant = msg.get("content", "") or ""
                        break
                records.append(
                    {
                        "corpus": "SFT target",
                        "condition": condition,
                        "run": path.name,
                        "id": obj.get("id"),
                        "text": assistant,
                        "max_token_hit": False,
                    }
                )
    return records


def enrich(records: list[dict]) -> list[dict]:
    out = []
    for row in records:
        text = row["text"]
        words = len(word_tokens(text))
        steps = step_count(text)
        out.append(
            {
                **row,
                "word_tokens": words,
                "TTR": ttr(text),
                "steps": steps,
                "tokens_per_step": 0.0 if steps == 0 else words / steps,
            }
        )
    return out


def summarize(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["corpus"], row["condition"], "all")].append(row)
        if not row["max_token_hit"]:
            grouped[(row["corpus"], row["condition"], "normal_termination")].append(row)

    rows = []
    for (corpus, condition, termination), items in grouped.items():
        if not items:
            continue
        n = len(items)
        rows.append(
            {
                "corpus": corpus,
                "condition": condition,
                "termination": termination,
                "n": n,
                "max_token_hit_rate": round(sum(1 for x in items if x["max_token_hit"]) / n, 4),
                "mean_word_tokens": round(sum(x["word_tokens"] for x in items) / n, 1),
                "TTR": round(sum(x["TTR"] for x in items) / n, 4),
                "steps": round(sum(x["steps"] for x in items) / n, 2),
                "tokens_per_step": round(sum(x["tokens_per_step"] for x in items) / n, 2),
            }
        )
    rows.sort(key=lambda r: (r["corpus"], condition_sort_key(r["condition"]), r["termination"]))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    records.extend(load_eval_records(ROOT / "kaggle_outputs" / "kaggle_kernel_outputs", "2025 eval"))
    records.extend(load_eval_records(ROOT / "kaggle_outputs" / "qwen3_8b_eval_2026", "2026 eval"))
    records.extend(load_sft_records())
    enriched = enrich(records)
    summary = summarize(enriched)

    write_csv(TABLES_DIR / "verbosity_metrics.csv", summary)
    write_csv(TABLES_DIR / "verbosity_record_level.csv", enriched)

    (TABLES_DIR / "verbosity_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (TABLES_DIR / "verbosity_record_level.json").write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = OUT_DIR / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Verbosity Analysis",
                "",
                "This package contains the final word-level verbosity analysis used for the thesis-facing interpretation layer.",
                "",
                "Included corpora:",
                "- `2025 eval`: refreshed 2025 student outputs, aggregated across 5 conditions x 3 seeds",
                "- `2026 eval`: 2026 student outputs, aggregated across 5 conditions x 3 seeds",
                "- `SFT target`: the five supervision target files used for the final analysis layer; the official-human source uses the 186-row variant",
                "",
                "Included metrics:",
                "- `max_token_hit_rate`",
                "- `mean_word_tokens`",
                "- `TTR`",
                "- `steps`",
                "- `tokens_per_step`",
                "",
                "Excluded metric:",
                "- `MATTR50` was deliberately dropped from the final package",
                "",
                "Generated by:",
                "- `python3 scripts/build_verbosity_analysis.py`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {TABLES_DIR / 'verbosity_metrics.csv'}")


if __name__ == "__main__":
    main()
