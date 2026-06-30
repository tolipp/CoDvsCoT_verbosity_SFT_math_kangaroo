from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results_so_far" / "qwen3_8b_refreshed_deep_analysis_2026-05-28_to_31"
TABLES_DIR = OUT_DIR / "tables"
NOTES_DIR = OUT_DIR / "notes"
SAMPLES_DIR = OUT_DIR / "samples"

try:
    TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
except Exception:
    TOKENIZER = None

RUN_SPECS = [
    {
        "condition": "1",
        "seed": 42,
        "label": "1_seed42",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed42_1" / "eval_outputs" / "eval_1_seed42_test2025" / "evaluation_results_changed.jsonl",
        "run_type": "kaggle_eval",
    },
    {
        "condition": "1",
        "seed": 1337,
        "label": "1_seed1337",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed1337_1" / "eval_outputs" / "eval_1_seed1337_test2025" / "evaluation_results_changed.jsonl",
        "run_type": "kaggle_eval",
    },
    {
        "condition": "1",
        "seed": 2024,
        "label": "1_seed2024",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed2024_1" / "eval_outputs" / "eval_1_seed2024_test2025" / "evaluation_results_changed.jsonl",
        "run_type": "kaggle_eval",
    },
    {
        "condition": "2a",
        "seed": 42,
        "label": "2a_seed42",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed42_2a" / "eval_outputs" / "eval_2a_seed42_test2025" / "evaluation_results_changed.jsonl",
        "run_type": "kaggle_eval",
    },
    {
        "condition": "2a",
        "seed": 1337,
        "label": "2a_seed1337",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed1337_2a" / "eval_outputs" / "eval_2a_seed1337_test2025" / "evaluation_results_changed.jsonl",
        "run_type": "kaggle_eval",
    },
    {
        "condition": "2a",
        "seed": 2024,
        "label": "2a_seed2024",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed2024_2a" / "eval_outputs" / "eval_2a_seed2024_test2025" / "evaluation_results_changed.jsonl",
        "run_type": "kaggle_eval",
    },
    {
        "condition": "3a",
        "seed": 42,
        "label": "3a_seed42",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed42_3a" / "eval_outputs" / "eval_3a_seed42_test2025" / "evaluation_results_changed.jsonl",
        "run_type": "kaggle_eval",
    },
    {
        "condition": "3a",
        "seed": 1337,
        "label": "3a_seed1337",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed1337_3a" / "eval_outputs" / "eval_3a_seed1337_test2025" / "evaluation_results_changed.jsonl",
        "run_type": "kaggle_eval",
    },
    {
        "condition": "3a",
        "seed": 2024,
        "label": "3a_seed2024",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed2024_3a" / "eval_outputs" / "eval_3a_seed2024_test2025" / "evaluation_results_changed.jsonl",
        "run_type": "kaggle_eval",
    },
    {
        "condition": "4a",
        "seed": 42,
        "label": "4a_seed42",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed42_4a" / "eval_outputs" / "eval_4a_seed42_test2025" / "evaluation_results_changed.jsonl",
        "run_type": "kaggle_eval",
    },
    {
        "condition": "4a",
        "seed": 1337,
        "label": "4a_seed1337",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed1337_4a" / "eval_outputs" / "eval_4a_seed1337_test2025" / "evaluation_results_changed.jsonl",
        "run_type": "kaggle_eval",
    },
    {
        "condition": "4a",
        "seed": 2024,
        "label": "4a_seed2024",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed2024_4a" / "eval_outputs" / "eval_4a_seed2024_test2025" / "evaluation_results_changed.jsonl",
        "run_type": "kaggle_eval",
    },
    {
        "condition": "5a",
        "seed": 42,
        "label": "5a_seed42",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed42_5a" / "unzipped" / "eval_outputs" / "eval_5a_seed42_test2025" / "evaluation_results_changed.jsonl",
        "run_type": "kaggle_eval",
    },
    {
        "condition": "5a",
        "seed": 1337,
        "label": "5a_seed1337",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed1337_5a" / "eval_outputs" / "eval_5a_seed1337_test2025" / "evaluation_results_changed.jsonl",
        "run_type": "kaggle_eval",
    },
    {
        "condition": "5a",
        "seed": 2024,
        "label": "5a_seed2024",
        "row_path": ROOT / "kaggle_outputs" / "kaggle_kernel_outputs" / "seed2024_5a" / "eval_outputs" / "eval_5a_seed2024_test2025" / "evaluation_results_changed.jsonl",
        "run_type": "kaggle_eval",
    },
]

BASELINE_PATHS = {
    condition: ROOT / "kaggle_outputs" / "openrouter_qwen3_8b_pretrain_baselines" / f"condition_{condition}_openrouter_qwen3_8b_test_2025.jsonl"
    for condition in ("1", "2a", "3a", "4a", "5a")
}

TEACHER_FILES = {
    "2a": ROOT / "generated_files" / "dataset_answers_qwen_qwen3_32b_clean" / "kaenguru_annotated_qwen_qwen3_32b_2a_full_2021_2025.jsonl",
    "3a": ROOT / "generated_files" / "dataset_answers_qwen_qwen3_32b_clean" / "kaenguru_annotated_qwen_qwen3_32b_3a_full_2021_2025.jsonl",
    "rewrite": ROOT / "generated_files" / "dataset_answers_qwen_qwen3_32b_clean" / "kaenguru_annotated_qwen_qwen3_32b_rewrite_only_full_2021_2025.jsonl",
}

SOURCE_PATH = ROOT / "kaenguru_2021_2025_no_diagram_finetune.json"

GUESS_WORD_RE = re.compile(r"\b(rate|geraten|vermute|vermuten|sch[aä]tze|wahrscheinlich|vielleicht|ich denke|ich tippe)\b", re.I)
ANSWER_RE = re.compile(r"Antwort\s*:\s*([A-E])", re.I)
PLACEHOLDER_RE = re.compile(r"<\s*(BUCHSTABE|ZIFFER)\s*>", re.I)
MATH_SIGNAL_RE = re.compile(r"[\d=+\-*/]|\\Rightarrow|\\text|\bSei\b|\bDann\b|\balso\b|\binsgesamt\b", re.I)
INSTRUCTION_RE = re.compile(r"L[öo]se die Aufgabe|Antwortoptionen|Antwort:\s*<BUCHSTABE>", re.I)


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def token_count(text: str) -> int:
    if TOKENIZER is not None:
        return len(TOKENIZER.encode(text, add_special_tokens=False))
    return len(re.findall(r"\w+|[^\w\s]", text or ""))


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def safe_stdev(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return stdev(values)


def build_source_map() -> dict[str, dict]:
    rows = load_json(SOURCE_PATH)
    out = {}
    for row in rows:
        tags = row.get("tags") or ["untagged"]
        out[row["id"]] = {
            "id": row["id"],
            "year": row["source"]["year"],
            "age_group": row["source"]["age_group"],
            "question_number": row["source"]["question_number"],
            "primary_category": tags[0],
            "all_categories": tags,
            "answer_key": row["answer_key"],
            "prompt_de": row["prompt_de"],
            "choices_de": row["choices_de"],
            "official_solution_de": row["official_solution_de"],
        }
    return out


def split_sentences(text: str) -> list[str]:
    text = text.replace("\r", "\n")
    chunks = re.split(r"(?:\n{2,}|(?<=[.!?])\s+)", text)
    return [c.strip() for c in chunks if c.strip()]


def repeated_sentence_flag(text: str) -> bool:
    sentences = split_sentences(text)
    counts = Counter(sentences)
    return any(len(s) >= 30 and c >= 3 for s, c in counts.items())


def repeated_ngram_flag(text: str, n: int = 8, threshold: int = 3) -> bool:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(tokens) < n:
        return False
    counts = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return any(c >= threshold for c in counts.values())


def distinct_answer_markers(text: str) -> list[str]:
    return list(dict.fromkeys(m.upper() for m in ANSWER_RE.findall(text or "")))


def choice_mapping_tag(meta: dict, response: str, predicted: str, expected: str) -> bool:
    if not predicted or predicted == "?" or predicted == expected:
        return False
    choices = meta.get("choices_de") or {}
    expected_text = normalize(choices.get(expected, ""))
    predicted_text = normalize(choices.get(predicted, ""))
    norm_resp = normalize(response)
    if expected_text and expected_text in norm_resp and (not predicted_text or predicted_text not in norm_resp):
        return True
    return False


def guess_risk_tags(text: str, token_len: int) -> list[str]:
    tags = []
    if GUESS_WORD_RE.search(text or ""):
        tags.append("guess_language")
    if token_len <= 35:
        tags.append("very_short_correct")
    if not MATH_SIGNAL_RE.search(text or ""):
        tags.append("no_math_signal")
    return tags


def failure_tags(meta: dict, response: str, predicted: str, expected: str, token_len: int, max_hit: bool) -> list[str]:
    tags = []
    markers = distinct_answer_markers(response)
    if predicted in (None, "", "?"):
        tags.append("no_extractable_answer")
    if max_hit:
        tags.append("max_token_hit")
    if PLACEHOLDER_RE.search(response or ""):
        tags.append("template_placeholder_leak")
    if INSTRUCTION_RE.search(response or ""):
        tags.append("instruction_echo")
    if repeated_sentence_flag(response or "") or repeated_ngram_flag(response or ""):
        tags.append("repetition_loop")
    if len(set(markers)) > 1:
        tags.append("contradictory_answer_markers")
    if choice_mapping_tag(meta, response or "", predicted or "", expected or ""):
        tags.append("possible_option_mapping_error")
    if token_len <= 20:
        tags.append("very_short_failure")
    if GUESS_WORD_RE.search(response or ""):
        tags.append("guess_language")
    if not MATH_SIGNAL_RE.search(response or ""):
        tags.append("no_math_signal")
    if not tags:
        tags.append("reasoning_error_without_surface_artifact")
    return tags


def primary_failure_tag(tags: list[str]) -> str:
    priority = [
        "no_extractable_answer",
        "template_placeholder_leak",
        "max_token_hit",
        "repetition_loop",
        "contradictory_answer_markers",
        "possible_option_mapping_error",
        "guess_language",
        "very_short_failure",
        "no_math_signal",
        "reasoning_error_without_surface_artifact",
    ]
    for tag in priority:
        if tag in tags:
            return tag
    return tags[0]


def load_eval_records(source_map: dict[str, dict]) -> list[dict]:
    records = []
    for spec in RUN_SPECS:
        rows = load_jsonl(spec["row_path"])
        for row in rows:
            if spec["run_type"] == "runpod_rescored":
                response = row.get("generated", "")
                predicted = row.get("answer_predicted")
                expected = row.get("answer_correct")
                correct = row.get("correct") is True
                tok_len = token_count(response)
                max_hit = False
                stopped_by_eos = None
                full_response = response
            else:
                response = row.get("generated_response", "")
                predicted = row.get("generated_answer")
                expected = row.get("expected_answer")
                correct = row.get("is_correct") is True
                tok_len = int(row.get("generated_token_count", 0))
                max_hit = row.get("hit_max_new_tokens") is True
                stopped_by_eos = row.get("stopped_by_eos")
                full_response = row.get("full_response", response)

            meta = source_map[row["id"]]
            rec = {
                "condition": spec["condition"],
                "seed": spec["seed"],
                "label": spec["label"],
                "id": row["id"],
                "year": meta["year"],
                "age_group": meta["age_group"],
                "question_number": meta["question_number"],
                "primary_category": meta["primary_category"],
                "all_categories": "|".join(meta["all_categories"]),
                "expected_answer": expected,
                "generated_answer": predicted,
                "is_correct": correct,
                "generated_token_count": tok_len,
                "hit_max_new_tokens": max_hit,
                "stopped_by_eos": stopped_by_eos,
                "generated_response": response,
                "full_response": full_response,
                "artifact_path": str(spec["row_path"].relative_to(ROOT)),
            }
            if correct:
                gtags = guess_risk_tags(response, tok_len)
                rec["guess_risk_tags"] = "|".join(gtags)
                rec["maybe_guess"] = bool({"guess_language", "very_short_correct", "no_math_signal"} & set(gtags)) and len(gtags) >= 2
                rec["failure_tags"] = ""
                rec["primary_failure_tag"] = ""
            else:
                ftags = failure_tags(meta, response, predicted, expected, tok_len, max_hit)
                rec["guess_risk_tags"] = ""
                rec["maybe_guess"] = False
                rec["failure_tags"] = "|".join(ftags)
                rec["primary_failure_tag"] = primary_failure_tag(ftags)
            records.append(rec)
    return records


def load_baseline_records(source_map: dict[str, dict]) -> list[dict]:
    records = []
    for condition, path in BASELINE_PATHS.items():
        rows = load_jsonl(path)
        dedup = {}
        for row in rows:
            dedup[row["id"]] = row
        for row in dedup.values():
            meta = source_map[row["id"]]
            response = row.get("generated", "")
            tok_len = int(row.get("usage", {}).get("completion_tokens", token_count(response)))
            predicted = row.get("answer_predicted")
            correct = row.get("correct") is True
            rec = {
                "condition": condition,
                "id": row["id"],
                "year": meta["year"],
                "age_group": meta["age_group"],
                "primary_category": meta["primary_category"],
                "all_categories": "|".join(meta["all_categories"]),
                "expected_answer": row.get("answer_correct"),
                "generated_answer": predicted,
                "is_correct": correct,
                "generated_token_count": tok_len,
                "finish_reason": row.get("finish_reason"),
                "generated_response": response,
            }
            if not correct:
                rec["failure_tags"] = "|".join(
                    failure_tags(meta, response, predicted, row.get("answer_correct"), tok_len, row.get("finish_reason") == "length")
                )
            else:
                rec["failure_tags"] = ""
            records.append(rec)
    return records


def summarize_runs(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in records:
        grouped[(rec["condition"], rec["seed"])].append(rec)
    out = []
    for (condition, seed), rows in sorted(grouped.items()):
        total = len(rows)
        correct = sum(r["is_correct"] for r in rows)
        scorable = sum(r["generated_answer"] not in (None, "", "?") for r in rows)
        unscorable = total - scorable
        token_counts = [r["generated_token_count"] for r in rows]
        max_hits = [r["hit_max_new_tokens"] for r in rows if r["hit_max_new_tokens"] is not None]
        out.append(
            {
                "condition": condition,
                "seed": seed,
                "correct": correct,
                "total": total,
                "strict_accuracy_percent": 100 * correct / total,
                "scorable": scorable,
                "unscorable": unscorable,
                "scorable_accuracy_percent": 100 * correct / scorable if scorable else None,
                "max_token_hits": sum(bool(x) for x in max_hits),
                "mean_generated_tokens": safe_mean(token_counts),
                "median_generated_tokens": median(token_counts),
                "max_generated_tokens": max(token_counts),
            }
        )
    return out


def summarize_conditions(run_summary: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in run_summary:
        grouped[row["condition"]].append(row)
    out = []
    for condition, rows in sorted(grouped.items()):
        stricts = [r["strict_accuracy_percent"] for r in rows]
        scorables = [r["scorable_accuracy_percent"] for r in rows if r["scorable_accuracy_percent"] is not None]
        max_hits = [r["max_token_hits"] for r in rows]
        tokens = [r["mean_generated_tokens"] for r in rows if r["mean_generated_tokens"] is not None]
        out.append(
            {
                "condition": condition,
                "n_seeds": len(rows),
                "mean_strict_accuracy_percent": safe_mean(stricts),
                "sd_strict_accuracy_percent": safe_stdev(stricts),
                "mean_scorable_accuracy_percent": safe_mean(scorables),
                "mean_max_token_hits": safe_mean(max_hits),
                "mean_generated_tokens": safe_mean(tokens),
            }
        )
    out.sort(key=lambda r: r["mean_strict_accuracy_percent"], reverse=True)
    return out


def student_condition_behavior_summary(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        grouped[rec["condition"]].append(rec)
    out = []
    for condition, rows in sorted(grouped.items()):
        total = len(rows)
        correct_rows = [r for r in rows if r["is_correct"]]
        incorrect_rows = [r for r in rows if not r["is_correct"]]
        repetition_rows = [r for r in incorrect_rows if "repetition_loop" in split_tags(r["failure_tags"])]
        no_extractable_rows = [r for r in incorrect_rows if "no_extractable_answer" in split_tags(r["failure_tags"])]
        reasoning_only_rows = [r for r in incorrect_rows if r["primary_failure_tag"] == "reasoning_error_without_surface_artifact"]
        max_hit_rows = [r for r in rows if r["hit_max_new_tokens"]]
        guessy_correct_rows = [r for r in correct_rows if r["maybe_guess"]]
        out.append(
            {
                "condition": condition,
                "n_item_seed_rows": total,
                "repetition_failure_count": len(repetition_rows),
                "repetition_failure_percent": 100 * len(repetition_rows) / total,
                "repetition_among_incorrect_percent": 100 * len(repetition_rows) / len(incorrect_rows) if incorrect_rows else 0.0,
                "no_extractable_count": len(no_extractable_rows),
                "no_extractable_percent": 100 * len(no_extractable_rows) / total,
                "reasoning_only_failure_count": len(reasoning_only_rows),
                "reasoning_only_failure_percent": 100 * len(reasoning_only_rows) / total,
                "max_token_hit_count": len(max_hit_rows),
                "max_token_hit_percent": 100 * len(max_hit_rows) / total,
                "guess_risk_correct_count": len(guessy_correct_rows),
                "guess_risk_correct_percent": 100 * len(guessy_correct_rows) / len(correct_rows) if correct_rows else 0.0,
                "mean_correct_generated_tokens": safe_mean([r["generated_token_count"] for r in correct_rows]),
                "mean_incorrect_generated_tokens": safe_mean([r["generated_token_count"] for r in incorrect_rows]),
            }
        )
    return out


def category_stats(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        grouped[(rec["condition"], rec["primary_category"])].append(rec)
    out = []
    for (condition, category), rows in sorted(grouped.items()):
        total = len(rows)
        correct = sum(r["is_correct"] for r in rows)
        max_hits = sum(bool(r["hit_max_new_tokens"]) for r in rows)
        incorrect_rows = [r for r in rows if not r["is_correct"]]
        repetition_rows = [r for r in incorrect_rows if "repetition_loop" in split_tags(r["failure_tags"])]
        no_extractable_rows = [r for r in incorrect_rows if "no_extractable_answer" in split_tags(r["failure_tags"])]
        out.append(
            {
                "condition": condition,
                "primary_category": category,
                "n_item_seed_rows": total,
                "correct": correct,
                "strict_accuracy_percent": 100 * correct / total,
                "max_token_hits": max_hits,
                "mean_generated_tokens": safe_mean([r["generated_token_count"] for r in rows]),
                "mean_correct_generated_tokens": safe_mean([r["generated_token_count"] for r in rows if r["is_correct"]]),
                "mean_incorrect_generated_tokens": safe_mean([r["generated_token_count"] for r in incorrect_rows]),
                "repetition_failure_count": len(repetition_rows),
                "repetition_failure_percent": 100 * len(repetition_rows) / total,
                "repetition_among_incorrect_percent": 100 * len(repetition_rows) / len(incorrect_rows) if incorrect_rows else 0.0,
                "no_extractable_count": len(no_extractable_rows),
                "no_extractable_percent": 100 * len(no_extractable_rows) / total,
            }
        )
    return out


def failure_tag_counts(records: list[dict]) -> list[dict]:
    rows = []
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        if rec["is_correct"]:
            continue
        grouped[(rec["condition"], rec["primary_failure_tag"])].append(rec)
    for (condition, tag), entries in sorted(grouped.items()):
        rows.append({"condition": condition, "primary_failure_tag": tag, "count": len(entries)})
    return rows


def failure_tag_counts_by_category(records: list[dict]) -> list[dict]:
    rows = []
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for rec in records:
        if rec["is_correct"]:
            continue
        grouped[(rec["condition"], rec["primary_category"], rec["primary_failure_tag"])].append(rec)
    for (condition, category, tag), entries in sorted(grouped.items()):
        rows.append(
            {
                "condition": condition,
                "primary_category": category,
                "primary_failure_tag": tag,
                "count": len(entries),
            }
        )
    return rows


def pairwise_seed_deltas(run_summary: list[dict]) -> list[dict]:
    by_key = {(r["condition"], r["seed"]): r for r in run_summary}
    out = []
    for seed in (42, 1337, 2024):
        base = by_key[("1", seed)]
        for condition in ("2a", "3a", "4a", "5a"):
            row = by_key[(condition, seed)]
            out.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "strict_accuracy_percent": row["strict_accuracy_percent"],
                    "condition_1_strict_accuracy_percent": base["strict_accuracy_percent"],
                    "delta_vs_condition_1_percent": row["strict_accuracy_percent"] - base["strict_accuracy_percent"],
                    "delta_max_token_hits": row["max_token_hits"] - base["max_token_hits"],
                }
            )
    return out


def pairwise_item_comparisons(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], dict[str, dict]] = defaultdict(dict)
    for rec in records:
        grouped[(rec["seed"], rec["id"])][rec["condition"]] = rec
    out = []
    for (seed, item_id), by_condition in grouped.items():
        if "1" not in by_condition:
            continue
        base = by_condition["1"]
        for condition in ("2a", "3a", "4a", "5a"):
            if condition not in by_condition:
                continue
            comp = by_condition[condition]
            out.append(
                {
                    "seed": seed,
                    "id": item_id,
                    "condition": condition,
                    "condition_correct": comp["is_correct"],
                    "condition_1_correct": base["is_correct"],
                    "pair_outcome": (
                        "win"
                        if comp["is_correct"] and not base["is_correct"]
                        else "loss"
                        if base["is_correct"] and not comp["is_correct"]
                        else "tie_correct"
                        if comp["is_correct"] and base["is_correct"]
                        else "tie_wrong"
                    ),
                    "primary_category": comp["primary_category"],
                }
            )
    return out


def two_sided_sign_pvalue(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def summarize_pairwise_item_comparisons(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    out = []
    for condition, entries in sorted(grouped.items()):
        counts = Counter(e["pair_outcome"] for e in entries)
        out.append(
            {
                "condition": condition,
                "wins_vs_condition_1": counts["win"],
                "losses_vs_condition_1": counts["loss"],
                "ties_correct": counts["tie_correct"],
                "ties_wrong": counts["tie_wrong"],
                "exploratory_sign_test_pvalue": two_sided_sign_pvalue(counts["win"], counts["loss"]),
            }
        )
    return out


def teacher_summary(source_map: dict[str, dict]) -> list[dict]:
    out = []

    rows_2a = load_jsonl(TEACHER_FILES["2a"])
    rows_3a = load_jsonl(TEACHER_FILES["3a"])
    rows_rw = load_jsonl(TEACHER_FILES["rewrite"])

    def summarize_teacher_rows(condition: str, items: list[dict], anno_key: str) -> list[dict]:
        results = []
        for scope_name, year_filter in [("all_2021_2025", None), ("heldout_2025", 2025), ("train_2021_2023", (2021, 2022, 2023))]:
            selected = []
            for row in items:
                year = row["source"]["year"]
                if year_filter is None:
                    pass
                elif isinstance(year_filter, tuple):
                    if year not in year_filter:
                        continue
                elif year != year_filter:
                    continue
                ann = row["annotations"].get(anno_key)
                if ann:
                    selected.append((row, ann))
            lengths = [ann.get("token_count") for _, ann in selected if ann.get("token_count") is not None]
            correct_lengths = [ann.get("token_count") for _, ann in selected if ann.get("token_count") is not None and ann.get("is_correct") is True]
            source_types = Counter(ann.get("source_type", "unknown") for _, ann in selected)
            results.append(
                {
                    "condition": condition,
                    "scope": scope_name,
                    "n_annotations": len(selected),
                    "n_correct": sum(1 for _, ann in selected if ann.get("is_correct") is True),
                    "n_incorrect": sum(1 for _, ann in selected if ann.get("is_correct") is not True),
                    "accuracy_percent": 100
                    * sum(1 for _, ann in selected if ann.get("is_correct") is True)
                    / len(selected)
                    if selected
                    else None,
                    "mean_token_count": safe_mean(lengths),
                    "median_token_count": median(lengths) if lengths else None,
                    "correct_only_mean_token_count": safe_mean(correct_lengths),
                    "correct_only_median_token_count": median(correct_lengths) if correct_lengths else None,
                }
            )
            for k, v in source_types.items():
                results[-1][f"source_type_{k}"] = v
            if condition in {"2a", "3a"}:
                correct_attempt_counts = [ann.get("teacher_correct_attempts", 0) for _, ann in selected]
                results[-1]["mean_teacher_correct_attempts"] = safe_mean(correct_attempt_counts)
                consistency_counts = Counter(ann.get("original_solve_consistency", "unknown") for _, ann in selected)
                for k, v in consistency_counts.items():
                    results[-1][f"consistency_{k.replace('/', '_of_')}"] = v
        return results

    out.extend(summarize_teacher_rows("2a_teacher_verbose", rows_2a, "teacher_verbose"))
    out.extend(summarize_teacher_rows("3a_teacher_concise", rows_3a, "teacher_concise"))
    out.extend(summarize_teacher_rows("4a_rewrite_concise", rows_rw, "reformulated_concise"))
    out.extend(summarize_teacher_rows("5a_rewrite_verbose", rows_rw, "reformulated_verbose"))

    # Independent attempt stats for the two teacher-solve conditions.
    for condition, items, anno_key in [
        ("2a_teacher_verbose", rows_2a, "teacher_verbose"),
        ("3a_teacher_concise", rows_3a, "teacher_concise"),
    ]:
        for scope_name, year_filter in [("all_2021_2025", None), ("heldout_2025", 2025)]:
            attempts = []
            for row in items:
                year = row["source"]["year"]
                if year_filter is not None and year != year_filter:
                    continue
                ann = row["annotations"].get(anno_key)
                if not ann:
                    continue
                for att in ann.get("independent_solve_attempts", []):
                    attempts.append(att)
            if attempts:
                out.append(
                    {
                        "condition": condition,
                        "scope": f"{scope_name}_independent_attempts",
                        "n_attempts": len(attempts),
                        "attempt_accuracy_percent": 100 * sum(att.get("is_correct") is True for att in attempts) / len(attempts),
                        "mean_attempt_token_count": safe_mean([att.get("token_count") for att in attempts if att.get("token_count") is not None]),
                        "n_attempt_length_failures": sum(att.get("finish_reason") == "length" for att in attempts),
                        "n_attempt_api_failures": sum(att.get("finish_reason") == "api_failure" for att in attempts),
                    }
                )

    return out


def teacher_trace_casebook(source_map: dict[str, dict]) -> None:
    configs = [
        ("2a_teacher_verbose", TEACHER_FILES["2a"], "teacher_verbose"),
        ("3a_teacher_concise", TEACHER_FILES["3a"], "teacher_concise"),
        ("4a_rewrite_concise", TEACHER_FILES["rewrite"], "reformulated_concise"),
        ("5a_rewrite_verbose", TEACHER_FILES["rewrite"], "reformulated_verbose"),
    ]

    for label, path, key in configs:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["source"]["year"] != 2025:
                continue
            ann = row["annotations"].get(key)
            if not ann:
                continue
            meta = source_map[row["id"]]
            content = ann.get("content", "")
            if not content and ann.get("independent_solve_attempts"):
                # Keep the richest failed attempt for diagnosis.
                attempts = sorted(
                    ann["independent_solve_attempts"],
                    key=lambda a: (a.get("token_count") or 0, a.get("finish_reason") != "length"),
                    reverse=True,
                )
                if attempts:
                    content = attempts[0].get("content", "")
            rec = {
                "id": row["id"],
                "primary_category": meta["primary_category"],
                "all_categories": meta["all_categories"],
                "is_correct": ann.get("is_correct") is True,
                "source_type": ann.get("source_type"),
                "token_count": ann.get("token_count"),
                "finish_reason": ann.get("finish_reason"),
                "extracted_answer": ann.get("extracted_answer"),
                "teacher_correct_attempts": ann.get("teacher_correct_attempts"),
                "original_solve_consistency": ann.get("original_solve_consistency"),
                "content": content,
                "independent_attempts": ann.get("independent_solve_attempts", []),
            }
            grouped[meta["primary_category"]].append(rec)

        sections = [f"# Teacher Trace Casebook: {label}", ""]
        for category, rows in sorted(grouped.items()):
            correct_rows = [r for r in rows if r["is_correct"]]
            wrong_rows = [r for r in rows if not r["is_correct"]]
            correct_rows.sort(key=lambda r: (r["token_count"] or 0, r["id"]))
            wrong_rows.sort(key=lambda r: (r["token_count"] or 0, r["id"]), reverse=True)
            chosen_correct = correct_rows[:2]
            chosen_wrong = wrong_rows[:5]
            sections.extend(
                [
                    f"## {category}",
                    "",
                    f"Correct teacher rows: {len(correct_rows)}",
                    f"Incorrect teacher rows: {len(wrong_rows)}",
                    "",
                    "### Correct teacher traces",
                    "",
                ]
            )
            for row in chosen_correct:
                sections.extend(
                    [
                        f"- `{row['id']}` / source `{row['source_type']}` / consistency `{row['original_solve_consistency']}` / tokens `{row['token_count']}`",
                        "",
                        "```text",
                        row["content"],
                        "```",
                        "",
                    ]
                )
            sections.append("### Incorrect teacher traces")
            sections.append("")
            for row in chosen_wrong:
                sections.extend(
                    [
                        f"- `{row['id']}` / source `{row['source_type']}` / extracted `{row['extracted_answer']}` / consistency `{row['original_solve_consistency']}` / tokens `{row['token_count']}` / finish `{row['finish_reason']}`",
                        "",
                        "```text",
                        row["content"] or "[no content captured]",
                        "```",
                        "",
                    ]
                )
                if row["independent_attempts"]:
                    sections.append("Independent attempts snapshot:")
                    sections.append("")
                    for att in row["independent_attempts"][:4]:
                        sections.extend(
                            [
                                f"- predicted `{att.get('predicted')}` / correct `{att.get('is_correct')}` / tokens `{att.get('token_count')}` / finish `{att.get('finish_reason')}`",
                            ]
                        )
                    sections.append("")
        (SAMPLES_DIR / f"teacher_trace_casebook_{label}.md").write_text("\n".join(sections) + "\n")


def baseline_summary(baseline_records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in baseline_records:
        grouped[rec["condition"]].append(rec)
    out = []
    for condition, rows in sorted(grouped.items()):
        total = len(rows)
        correct = sum(r["is_correct"] for r in rows)
        scorable = sum(r["generated_answer"] not in (None, "", "?") for r in rows)
        out.append(
            {
                "condition_label": condition,
                "total": total,
                "correct": correct,
                "strict_accuracy_percent": 100 * correct / total,
                "scorable_accuracy_percent": 100 * correct / scorable if scorable else None,
                "unscorable": total - scorable,
                "max_completion_hits_1024": sum(r["generated_token_count"] >= 1024 or r.get("finish_reason") == "length" for r in rows),
                "mean_generated_tokens": safe_mean([r["generated_token_count"] for r in rows]),
                "median_generated_tokens": median([r["generated_token_count"] for r in rows]),
            }
        )
    return out


def teacher_heldout_records(source_map: dict[str, dict]) -> list[dict]:
    records = []

    def add_record(condition: str, row: dict, ann: dict, mode: str) -> None:
        meta = source_map[row["id"]]
        attempts = ann.get("independent_solve_attempts") or []
        attempt_correct = [att.get("is_correct") is True for att in attempts]
        content = ann.get("content") or ""
        if not content and attempts:
            attempts_sorted = sorted(
                attempts,
                key=lambda a: ((a.get("token_count") or 0), a.get("finish_reason") == "length"),
                reverse=True,
            )
            content = attempts_sorted[0].get("content") or ""
        records.append(
            {
                "condition": condition,
                "mode": mode,
                "id": row["id"],
                "primary_category": meta["primary_category"],
                "all_categories": "|".join(meta["all_categories"]),
                "expected_answer": meta["answer_key"],
                "generated_answer": ann.get("extracted_answer"),
                "is_correct": ann.get("is_correct") is True,
                "source_type": ann.get("source_type"),
                "teacher_correct_attempts": ann.get("teacher_correct_attempts"),
                "token_count": ann.get("token_count"),
                "finish_reason": ann.get("finish_reason"),
                "content": content,
                "attempt_count": len(attempts),
                "attempt_accuracy_percent": (100 * sum(attempt_correct) / len(attempt_correct)) if attempt_correct else None,
                "all_predictions": "|".join(ann.get("all_predictions") or []),
            }
        )

    for line in TEACHER_FILES["2a"].read_text().splitlines():
        row = json.loads(line)
        if row["source"]["year"] == 2025:
            add_record("2a", row, row["annotations"]["teacher_verbose"], "teacher_direct_solve")

    for line in TEACHER_FILES["3a"].read_text().splitlines():
        row = json.loads(line)
        if row["source"]["year"] == 2025:
            add_record("3a", row, row["annotations"]["teacher_concise"], "teacher_direct_solve")

    for line in TEACHER_FILES["rewrite"].read_text().splitlines():
        row = json.loads(line)
        if row["source"]["year"] == 2025:
            add_record("4a", row, row["annotations"]["reformulated_concise"], "teacher_rewrite")
            add_record("5a", row, row["annotations"]["reformulated_verbose"], "teacher_rewrite")

    return records


def teacher_tag_summary(teacher_records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in teacher_records:
        grouped[(rec["condition"], rec["primary_category"])].append(rec)

    out = []
    for (condition, category), rows in sorted(grouped.items()):
        attempts = [r["attempt_accuracy_percent"] for r in rows if r["attempt_accuracy_percent"] is not None]
        lengths = [r["token_count"] for r in rows if r["token_count"] is not None]
        correct_lengths = [r["token_count"] for r in rows if r["token_count"] is not None and r["is_correct"]]
        out.append(
            {
                "condition": condition,
                "primary_category": category,
                "n_problems": len(rows),
                "teacher_final_accuracy_percent": 100 * sum(r["is_correct"] for r in rows) / len(rows),
                "teacher_4of4_pass_percent": 100 * sum(r["teacher_correct_attempts"] == 4 for r in rows) / len(rows),
                "teacher_failed_percent": 100 * sum(r["source_type"] == "failed" for r in rows) / len(rows),
                "teacher_hint_assisted_percent": 100 * sum(r["source_type"] == "hint_assisted" for r in rows) / len(rows),
                "teacher_independent_percent": 100 * sum(r["source_type"] == "independent_solve" for r in rows) / len(rows),
                "teacher_mean_attempt_accuracy_percent": safe_mean(attempts),
                "teacher_mean_token_count": safe_mean(lengths),
                "teacher_mean_token_count_correct_only": safe_mean(correct_lengths),
                "teacher_length_finish_percent": 100 * sum(r["finish_reason"] == "length" for r in rows) / len(rows),
            }
        )
    return out


def split_tags(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x]
    if not value:
        return []
    return [x for x in str(value).split("|") if x]


def all_tag_accuracy_summary(records: list[dict], condition_filter: str | None = None) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        if condition_filter is not None and rec["condition"] != condition_filter:
            continue
        for tag in split_tags(rec.get("all_categories")):
            grouped[(rec["condition"], tag)].append(rec)

    out = []
    for (condition, tag), rows in sorted(grouped.items()):
        total = len(rows)
        correct = sum(r["is_correct"] for r in rows)
        incorrect_rows = [r for r in rows if not r["is_correct"]]
        repetition_rows = [r for r in incorrect_rows if "repetition_loop" in split_tags(r["failure_tags"])]
        no_extractable_rows = [r for r in incorrect_rows if "no_extractable_answer" in split_tags(r["failure_tags"])]
        out.append(
            {
                "condition": condition,
                "tag": tag,
                "n_item_seed_rows": total,
                "n_unique_problems": len({r["id"] for r in rows}),
                "correct": correct,
                "strict_accuracy_percent": 100 * correct / total,
                "max_token_hits": sum(r["primary_failure_tag"] == "max_token_hit" for r in rows),
                "mean_generated_tokens": safe_mean([r["generated_token_count"] for r in rows]),
                "mean_correct_generated_tokens": safe_mean([r["generated_token_count"] for r in rows if r["is_correct"]]),
                "mean_incorrect_generated_tokens": safe_mean([r["generated_token_count"] for r in incorrect_rows]),
                "repetition_failure_count": len(repetition_rows),
                "repetition_failure_percent": 100 * len(repetition_rows) / total,
                "no_extractable_count": len(no_extractable_rows),
                "no_extractable_percent": 100 * len(no_extractable_rows) / total,
            }
        )
    return out


def baseline_all_tag_summary(baseline_records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in baseline_records:
        for tag in split_tags(rec.get("all_categories")):
            grouped[(rec["condition"], tag)].append(rec)

    out = []
    for (condition, tag), rows in sorted(grouped.items()):
        total = len(rows)
        correct = sum(r["is_correct"] for r in rows)
        out.append(
            {
                "condition": condition,
                "tag": tag,
                "n_rows": total,
                "n_unique_problems": len({r["id"] for r in rows}),
                "strict_accuracy_percent": 100 * correct / total,
                "mean_generated_tokens": safe_mean([r["generated_token_count"] for r in rows]),
                "length_finish_percent": 100 * sum(r["finish_reason"] == "length" or r["generated_token_count"] >= 1024 for r in rows) / total,
            }
        )
    return out


def teacher_all_tag_summary(teacher_records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in teacher_records:
        for tag in split_tags(rec.get("all_categories")):
            grouped[(rec["condition"], tag)].append(rec)

    out = []
    for (condition, tag), rows in sorted(grouped.items()):
        attempts = [r["attempt_accuracy_percent"] for r in rows if r["attempt_accuracy_percent"] is not None]
        lengths = [r["token_count"] for r in rows if r["token_count"] is not None]
        correct_lengths = [r["token_count"] for r in rows if r["token_count"] is not None and r["is_correct"]]
        teacher_4of4_pass = 100 * sum(r["teacher_correct_attempts"] == 4 for r in rows) / len(rows)
        mean_attempt_acc = safe_mean(attempts)
        out.append(
            {
                "condition": condition,
                "tag": tag,
                "n_problems": len(rows),
                "n_unique_problems": len({r["id"] for r in rows}),
                "teacher_final_accuracy_percent": 100 * sum(r["is_correct"] for r in rows) / len(rows),
                "teacher_4of4_pass_percent": teacher_4of4_pass,
                "teacher_nonperfect_4try_percent": 100 - teacher_4of4_pass if condition in {"2a", "3a"} else None,
                "teacher_failed_percent": 100 * sum(r["source_type"] == "failed" for r in rows) / len(rows),
                "teacher_hint_assisted_percent": 100 * sum(r["source_type"] == "hint_assisted" for r in rows) / len(rows),
                "teacher_independent_percent": 100 * sum(r["source_type"] == "independent_solve" for r in rows) / len(rows),
                "teacher_mean_attempt_accuracy_percent": mean_attempt_acc,
                "teacher_independent_attempt_failure_percent": (100 - mean_attempt_acc) if mean_attempt_acc is not None else None,
                "teacher_mean_token_count": safe_mean(lengths),
                "teacher_mean_token_count_correct_only": safe_mean(correct_lengths),
                "teacher_length_finish_percent": 100 * sum(r["finish_reason"] == "length" for r in rows) / len(rows),
            }
        )
    return out


def student_teacher_all_tag_linkage(
    student_tag_rows: list[dict],
    teacher_tag_rows: list[dict],
    baseline_tag_rows: list[dict],
) -> list[dict]:
    student_map = {(r["condition"], r["tag"]): r for r in student_tag_rows}
    teacher_map = {(r["condition"], r["tag"]): r for r in teacher_tag_rows}
    baseline_map = {(r["condition"], r["tag"]): r for r in baseline_tag_rows}
    cond1_map = {(r["condition"], r["tag"]): r for r in student_tag_rows if r["condition"] == "1"}
    rows = []
    all_tags = sorted({r["tag"] for r in student_tag_rows})
    for condition in ("2a", "3a", "4a", "5a"):
        for tag in all_tags:
            s = student_map.get((condition, tag))
            t = teacher_map.get((condition, tag))
            b = baseline_map.get((condition, tag))
            c1 = cond1_map.get(("1", tag))
            if not s or not t:
                continue
            student_acc = s["strict_accuracy_percent"]
            baseline_acc = b["strict_accuracy_percent"] if b else None
            cond1_acc = c1["strict_accuracy_percent"] if c1 else None
            rows.append(
                {
                    "condition": condition,
                    "tag": tag,
                    "student_accuracy_percent": student_acc,
                    "student_max_token_hits": s["max_token_hits"],
                    "student_mean_generated_tokens": s["mean_generated_tokens"],
                    "student_mean_correct_generated_tokens": s["mean_correct_generated_tokens"],
                    "student_mean_incorrect_generated_tokens": s["mean_incorrect_generated_tokens"],
                    "student_repetition_failure_percent": s["repetition_failure_percent"],
                    "student_no_extractable_percent": s["no_extractable_percent"],
                    "baseline_8b_accuracy_percent": baseline_acc,
                    "condition1_accuracy_percent": cond1_acc,
                    "student_minus_baseline8b_percent": (student_acc - baseline_acc) if baseline_acc is not None else None,
                    "student_minus_condition1_percent": (student_acc - cond1_acc) if cond1_acc is not None else None,
                    "teacher_final_accuracy_percent": t["teacher_final_accuracy_percent"],
                    "teacher_4of4_pass_percent": t["teacher_4of4_pass_percent"],
                    "teacher_nonperfect_4try_percent": t["teacher_nonperfect_4try_percent"],
                    "teacher_failed_percent": t["teacher_failed_percent"],
                    "teacher_hint_assisted_percent": t["teacher_hint_assisted_percent"],
                    "teacher_mean_attempt_accuracy_percent": t["teacher_mean_attempt_accuracy_percent"],
                    "teacher_independent_attempt_failure_percent": t["teacher_independent_attempt_failure_percent"],
                    "teacher_mean_token_count": t["teacher_mean_token_count"],
                    "teacher_mean_token_count_correct_only": t["teacher_mean_token_count_correct_only"],
                    "teacher_length_finish_percent": t["teacher_length_finish_percent"],
                }
            )
    return rows


def baseline_category_summary(baseline_records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in baseline_records:
        grouped[(rec["condition"], rec["primary_category"])].append(rec)
    out = []
    for (condition, category), rows in sorted(grouped.items()):
        total = len(rows)
        correct = sum(r["is_correct"] for r in rows)
        out.append(
            {
                "condition": condition,
                "primary_category": category,
                "n_rows": total,
                "strict_accuracy_percent": 100 * correct / total,
                "mean_generated_tokens": safe_mean([r["generated_token_count"] for r in rows]),
                "length_finish_percent": 100 * sum(r["finish_reason"] == "length" or r["generated_token_count"] >= 1024 for r in rows) / total,
            }
        )
    return out


def student_teacher_tag_linkage(category_rows: list[dict], teacher_tag_rows: list[dict]) -> list[dict]:
    student_map = {(r["condition"], r["primary_category"]): r for r in category_rows}
    teacher_map = {(r["condition"], r["primary_category"]): r for r in teacher_tag_rows}
    rows = []
    for condition in ("2a", "3a", "4a", "5a"):
        for category in sorted({r["primary_category"] for r in category_rows}):
            s = student_map.get((condition, category))
            t = teacher_map.get((condition, category))
            if not s or not t:
                continue
            rows.append(
                {
                    "condition": condition,
                    "primary_category": category,
                    "student_accuracy_percent": s["strict_accuracy_percent"],
                    "student_max_token_hits": s["max_token_hits"],
                    "student_mean_generated_tokens": s["mean_generated_tokens"],
                    "student_mean_correct_generated_tokens": s["mean_correct_generated_tokens"],
                    "student_mean_incorrect_generated_tokens": s["mean_incorrect_generated_tokens"],
                    "student_repetition_failure_percent": s["repetition_failure_percent"],
                    "student_repetition_among_incorrect_percent": s["repetition_among_incorrect_percent"],
                    "student_no_extractable_percent": s["no_extractable_percent"],
                    "teacher_final_accuracy_percent": t["teacher_final_accuracy_percent"],
                    "teacher_4of4_pass_percent": t["teacher_4of4_pass_percent"],
                    "teacher_failed_percent": t["teacher_failed_percent"],
                    "teacher_hint_assisted_percent": t["teacher_hint_assisted_percent"],
                    "teacher_mean_attempt_accuracy_percent": t["teacher_mean_attempt_accuracy_percent"],
                    "teacher_mean_token_count": t["teacher_mean_token_count"],
                    "teacher_mean_token_count_correct_only": t["teacher_mean_token_count_correct_only"],
                    "teacher_length_finish_percent": t["teacher_length_finish_percent"],
                }
            )
    return rows


def official_solution_condition_stats(source_rows: list[dict]) -> dict[str, dict]:
    out = {}
    for label, years in [("train_2021_2023", {2021, 2022, 2023}), ("heldout_2025", {2025})]:
        selected = [r for r in source_rows if r["source"]["year"] in years]
        lengths = [token_count(r.get("official_solution_de", "")) for r in selected]
        out[label] = {
            "n_annotations": len(selected),
            "accuracy_percent": 100.0,
            "mean_token_count": safe_mean(lengths),
            "correct_only_mean_token_count": safe_mean(lengths),
        }
    return out


def condition_annotation_outcome_linkage(
    condition_summary: list[dict],
    behavior_rows: list[dict],
    teacher_rows: list[dict],
    source_rows: list[dict],
) -> list[dict]:
    teacher_name_to_condition = {
        "2a_teacher_verbose": "2a",
        "3a_teacher_concise": "3a",
        "4a_rewrite_concise": "4a",
        "5a_rewrite_verbose": "5a",
    }
    train_map = {}
    heldout_map = {}
    for row in teacher_rows:
        cond = teacher_name_to_condition.get(row["condition"])
        if not cond:
            continue
        if row["scope"] == "train_2021_2023":
            train_map[cond] = row
        elif row["scope"] == "heldout_2025":
            heldout_map[cond] = row

    official_map = official_solution_condition_stats(source_rows)
    cond_summary_map = {r["condition"]: r for r in condition_summary}
    behavior_map = {r["condition"]: r for r in behavior_rows}

    rows = []
    for condition in ("1", "2a", "3a", "4a", "5a"):
        c = cond_summary_map[condition]
        b = behavior_map[condition]
        if condition == "1":
            train_row = official_map["train_2021_2023"]
            heldout_row = official_map["heldout_2025"]
            annotation_family = "official_human"
        else:
            train_row = train_map.get(condition, {})
            heldout_row = heldout_map.get(condition, {})
            annotation_family = "teacher_direct_solve" if condition in {"2a", "3a"} else "teacher_rewrite"
        rows.append(
            {
                "condition": condition,
                "annotation_family": annotation_family,
                "train_annotation_accuracy_percent": train_row.get("accuracy_percent"),
                "train_annotation_mean_token_count": train_row.get("mean_token_count"),
                "train_annotation_correct_only_mean_token_count": train_row.get("correct_only_mean_token_count"),
                "heldout_annotation_accuracy_percent": heldout_row.get("accuracy_percent"),
                "heldout_annotation_mean_token_count": heldout_row.get("mean_token_count"),
                "heldout_annotation_correct_only_mean_token_count": heldout_row.get("correct_only_mean_token_count"),
                "student_mean_strict_accuracy_percent": c["mean_strict_accuracy_percent"],
                "student_sd_strict_accuracy_percent": c["sd_strict_accuracy_percent"],
                "student_mean_scorable_accuracy_percent": c["mean_scorable_accuracy_percent"],
                "student_mean_generated_tokens": c["mean_generated_tokens"],
                "student_mean_max_token_hits": c["mean_max_token_hits"],
                "student_repetition_failure_percent": b["repetition_failure_percent"],
                "student_repetition_among_incorrect_percent": b["repetition_among_incorrect_percent"],
                "student_no_extractable_percent": b["no_extractable_percent"],
                "student_reasoning_only_failure_percent": b["reasoning_only_failure_percent"],
                "student_guess_risk_correct_percent": b["guess_risk_correct_percent"],
                "student_mean_correct_generated_tokens": b["mean_correct_generated_tokens"],
                "student_mean_incorrect_generated_tokens": b["mean_incorrect_generated_tokens"],
            }
        )
    return rows


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def trace_casebook(records: list[dict]) -> None:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        grouped[(rec["condition"], rec["primary_category"])].append(rec)

    by_condition: dict[str, list[str]] = defaultdict(list)
    for (condition, category), rows in sorted(grouped.items()):
        correct_rows = [r for r in rows if r["is_correct"]]
        wrong_rows = [r for r in rows if not r["is_correct"]]

        correct_rows.sort(key=lambda r: (not r["maybe_guess"], r["generated_token_count"], r["seed"], r["id"]))
        chosen_correct = []
        seen_ids = set()
        for row in correct_rows:
            if row["id"] in seen_ids:
                continue
            chosen_correct.append(row)
            seen_ids.add(row["id"])
            if len(chosen_correct) == 2:
                break

        wrong_grouped: dict[str, list[dict]] = defaultdict(list)
        for row in wrong_rows:
            wrong_grouped[row["primary_failure_tag"]].append(row)
        for rows2 in wrong_grouped.values():
            rows2.sort(key=lambda r: (-r["generated_token_count"], r["seed"], r["id"]))

        chosen_wrong = []
        used = set()
        for tag in sorted(wrong_grouped, key=lambda t: (-len(wrong_grouped[t]), t)):
            for row in wrong_grouped[tag]:
                key = (row["seed"], row["id"])
                if key in used:
                    continue
                chosen_wrong.append(row)
                used.add(key)
                break
            if len(chosen_wrong) >= 5:
                break
        if len(chosen_wrong) < 5:
            for row in sorted(wrong_rows, key=lambda r: (-r["generated_token_count"], r["seed"], r["id"])):
                key = (row["seed"], row["id"])
                if key in used:
                    continue
                chosen_wrong.append(row)
                used.add(key)
                if len(chosen_wrong) >= 5:
                    break

        block = [f"## {category}", ""]
        block.append(f"Correct sample count in pool: {len(correct_rows)}")
        block.append(f"Incorrect sample count in pool: {len(wrong_rows)}")
        block.append("")
        block.append("### Correct traces")
        block.append("")
        for row in chosen_correct:
            block.extend(
                [
                    f"- `{row['label']}` / `{row['id']}` / tokens `{row['generated_token_count']}` / guess-risk `{row['maybe_guess']}` / guess tags `{row['guess_risk_tags'] or 'none'}`",
                    "",
                    "```text",
                    row["full_response"],
                    "```",
                    "",
                ]
            )
        block.append("### Incorrect traces")
        block.append("")
        for row in chosen_wrong:
            block.extend(
                [
                    f"- `{row['label']}` / `{row['id']}` / pred `{row['generated_answer']}` vs gold `{row['expected_answer']}` / tokens `{row['generated_token_count']}` / tags `{row['failure_tags']}`",
                    "",
                    "```text",
                    row["full_response"],
                    "```",
                    "",
                ]
            )
        by_condition[condition].append("\n".join(block))

    for condition, sections in by_condition.items():
        path = SAMPLES_DIR / f"trace_casebook_{condition}.md"
        path.write_text(f"# Trace Casebook: {condition}\n\n" + "\n\n".join(sections) + "\n")


def representative_run_labels(run_summary: list[dict], condition_summary: list[dict]) -> dict[str, str]:
    mean_map = {row["condition"]: row["mean_strict_accuracy_percent"] for row in condition_summary}
    reps = {}
    for condition in sorted(mean_map):
        candidates = [r for r in run_summary if r["condition"] == condition]
        target = mean_map[condition]
        mean_token_target = mean([(c["mean_generated_tokens"] or 0) for c in candidates])
        candidates.sort(
            key=lambda r: (
                abs(r["strict_accuracy_percent"] - target),
                r["max_token_hits"],
                abs((r["mean_generated_tokens"] or 0) - mean_token_target),
                r["seed"],
            )
        )
        reps[condition] = f"{condition}_seed{candidates[0]['seed']}"
    return reps


def condition_example_digest(records: list[dict], run_summary: list[dict], condition_summary: list[dict]) -> list[dict]:
    rep_labels = representative_run_labels(run_summary, condition_summary)
    chosen_rows = []

    for condition in ("1", "2a", "3a", "4a", "5a"):
        label = rep_labels[condition]
        rows = [r for r in records if r["label"] == label]
        if not rows:
            continue

        token_median = median([r["generated_token_count"] for r in rows])
        correct_rows = [r for r in rows if r["is_correct"]]
        wrong_rows = [r for r in rows if not r["is_correct"]]

        correct_rows.sort(
            key=lambda r: (
                r["maybe_guess"],
                abs(r["generated_token_count"] - token_median),
                r["primary_category"],
                r["id"],
            )
        )
        chosen_correct = []
        seen_categories = set()
        for row in correct_rows:
            if row["primary_category"] in seen_categories and len(seen_categories) < len({x["primary_category"] for x in correct_rows}):
                continue
            chosen_correct.append(row)
            seen_categories.add(row["primary_category"])
            if len(chosen_correct) == 2:
                break
        if len(chosen_correct) < 2:
            for row in correct_rows:
                if row not in chosen_correct:
                    chosen_correct.append(row)
                if len(chosen_correct) == 2:
                    break

        wrong_by_tag: dict[str, list[dict]] = defaultdict(list)
        for row in wrong_rows:
            wrong_by_tag[row["primary_failure_tag"]].append(row)
        for rows2 in wrong_by_tag.values():
            rows2.sort(
                key=lambda r: (
                    -r["generated_token_count"],
                    r["primary_category"],
                    r["id"],
                )
            )

        chosen_wrong = []
        used_ids = set()
        for tag in sorted(wrong_by_tag, key=lambda t: (-len(wrong_by_tag[t]), t)):
            for row in wrong_by_tag[tag]:
                if row["id"] in used_ids:
                    continue
                chosen_wrong.append(row)
                used_ids.add(row["id"])
                break
            if len(chosen_wrong) >= 5:
                break
        if len(chosen_wrong) < 5:
            for row in sorted(
                wrong_rows,
                key=lambda r: (
                    -r["generated_token_count"],
                    r["primary_failure_tag"],
                    r["primary_category"],
                    r["id"],
                ),
            ):
                if row["id"] in used_ids:
                    continue
                chosen_wrong.append(row)
                used_ids.add(row["id"])
                if len(chosen_wrong) >= 5:
                    break

        lines = [
            f"# Representative 7-Example Digest: {condition}",
            "",
            f"Representative run: `{label}`",
            "",
            "Selection rule:",
            f"- chosen run = seed closest to the condition mean strict accuracy, then lower max-token-hit count, then lower seed",
            "- chosen traces = 2 correct traces with low guess-risk and category diversity, plus 5 incorrect traces with failure-tag diversity",
            "",
            "## Correct traces",
            "",
        ]
        for row in chosen_correct:
            chosen_rows.append(
                {
                    "family": "student",
                    "condition": condition,
                    "representative_label": label,
                    "kind": "correct",
                    "id": row["id"],
                    "primary_category": row["primary_category"],
                    "generated_token_count": row["generated_token_count"],
                    "maybe_guess": row["maybe_guess"],
                    "guess_risk_tags": row["guess_risk_tags"],
                    "failure_tags": "",
                }
            )
            lines.extend(
                [
                    f"- `{row['id']}` / category `{row['primary_category']}` / tokens `{row['generated_token_count']}` / maybe-guess `{row['maybe_guess']}` / guess tags `{row['guess_risk_tags'] or 'none'}`",
                    "",
                    "```text",
                    row["full_response"],
                    "```",
                    "",
                ]
            )
        lines.extend(["## Incorrect traces", ""])
        for row in chosen_wrong:
            chosen_rows.append(
                {
                    "family": "student",
                    "condition": condition,
                    "representative_label": label,
                    "kind": "incorrect",
                    "id": row["id"],
                    "primary_category": row["primary_category"],
                    "generated_token_count": row["generated_token_count"],
                    "maybe_guess": False,
                    "guess_risk_tags": "",
                    "failure_tags": row["failure_tags"],
                }
            )
            lines.extend(
                [
                    f"- `{row['id']}` / category `{row['primary_category']}` / pred `{row['generated_answer']}` vs gold `{row['expected_answer']}` / tokens `{row['generated_token_count']}` / tags `{row['failure_tags']}`",
                    "",
                    "```text",
                    row["full_response"],
                    "```",
                    "",
                ]
            )
        (SAMPLES_DIR / f"condition_digest_{condition}.md").write_text("\n".join(lines) + "\n")

    return chosen_rows


def teacher_example_digest(source_map: dict[str, dict]) -> list[dict]:
    configs = [
        ("2a", "2a_teacher_verbose", TEACHER_FILES["2a"], "teacher_verbose"),
        ("3a", "3a_teacher_concise", TEACHER_FILES["3a"], "teacher_concise"),
        ("4a", "4a_rewrite_concise", TEACHER_FILES["rewrite"], "reformulated_concise"),
        ("5a", "5a_rewrite_verbose", TEACHER_FILES["rewrite"], "reformulated_verbose"),
    ]
    chosen_rows = []

    for condition, label, path, key in configs:
        rows = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["source"]["year"] != 2025:
                continue
            ann = row["annotations"].get(key)
            if not ann:
                continue
            meta = source_map[row["id"]]
            content = ann.get("content", "")
            if not content and ann.get("independent_solve_attempts"):
                attempts = sorted(
                    ann["independent_solve_attempts"],
                    key=lambda a: ((a.get("token_count") or 0), a.get("finish_reason") != "length"),
                    reverse=True,
                )
                if attempts:
                    content = attempts[0].get("content", "")
            rows.append(
                {
                    "id": row["id"],
                    "primary_category": meta["primary_category"],
                    "is_correct": ann.get("is_correct") is True,
                    "source_type": ann.get("source_type"),
                    "token_count": ann.get("token_count"),
                    "finish_reason": ann.get("finish_reason"),
                    "extracted_answer": ann.get("extracted_answer"),
                    "teacher_correct_attempts": ann.get("teacher_correct_attempts"),
                    "original_solve_consistency": ann.get("original_solve_consistency"),
                    "content": content,
                    "independent_attempts": ann.get("independent_solve_attempts", []),
                }
            )

        correct_rows = [r for r in rows if r["is_correct"]]
        wrong_rows = [r for r in rows if not r["is_correct"]]
        correct_rows.sort(key=lambda r: ((r["token_count"] or 0), r["primary_category"], r["id"]))
        wrong_rows.sort(
            key=lambda r: (
                r["finish_reason"] != "length",
                -(r["token_count"] or 0),
                r["source_type"] or "",
                r["id"],
            )
        )

        chosen_correct = []
        seen_categories = set()
        for row in correct_rows:
            if row["primary_category"] in seen_categories and len(seen_categories) < len({x["primary_category"] for x in correct_rows}):
                continue
            chosen_correct.append(row)
            seen_categories.add(row["primary_category"])
            if len(chosen_correct) == 2:
                break
        if len(chosen_correct) < 2:
            for row in correct_rows:
                if row not in chosen_correct:
                    chosen_correct.append(row)
                if len(chosen_correct) == 2:
                    break

        chosen_wrong = []
        used_ids = set()
        wrong_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in wrong_rows:
            wrong_groups[(row["source_type"] or "unknown", row["finish_reason"] or "none")].append(row)
        for group_rows in wrong_groups.values():
            group_rows.sort(key=lambda r: (-(r["token_count"] or 0), r["primary_category"], r["id"]))
        for group in sorted(wrong_groups, key=lambda g: (-len(wrong_groups[g]), g[0], g[1])):
            row = wrong_groups[group][0]
            chosen_wrong.append(row)
            used_ids.add(row["id"])
            if len(chosen_wrong) == 5:
                break
        if len(chosen_wrong) < 5:
            for row in wrong_rows:
                if row["id"] in used_ids:
                    continue
                chosen_wrong.append(row)
                used_ids.add(row["id"])
                if len(chosen_wrong) == 5:
                    break

        lines = [
            f"# Teacher 7-Example Digest: {label}",
            "",
            "Selection rule:",
            "- 2 correct traces with category diversity",
            "- 5 incorrect traces with source-type / finish-reason diversity",
            "",
            "## Correct teacher traces",
            "",
        ]
        for row in chosen_correct:
            chosen_rows.append(
                {
                    "family": "teacher",
                    "condition": condition,
                    "representative_label": label,
                    "kind": "correct",
                    "id": row["id"],
                    "primary_category": row["primary_category"],
                    "generated_token_count": row["token_count"],
                    "maybe_guess": "",
                    "guess_risk_tags": "",
                    "failure_tags": "",
                }
            )
            lines.extend(
                [
                    f"- `{row['id']}` / category `{row['primary_category']}` / source `{row['source_type']}` / consistency `{row['original_solve_consistency']}` / tokens `{row['token_count']}`",
                    "",
                    "```text",
                    row["content"] or "[no content captured]",
                    "```",
                    "",
                ]
            )
        lines.extend(["## Incorrect teacher traces", ""])
        for row in chosen_wrong:
            chosen_rows.append(
                {
                    "family": "teacher",
                    "condition": condition,
                    "representative_label": label,
                    "kind": "incorrect",
                    "id": row["id"],
                    "primary_category": row["primary_category"],
                    "generated_token_count": row["token_count"],
                    "maybe_guess": "",
                    "guess_risk_tags": "",
                    "failure_tags": f"source_type={row['source_type']}|finish_reason={row['finish_reason']}",
                }
            )
            lines.extend(
                [
                    f"- `{row['id']}` / category `{row['primary_category']}` / source `{row['source_type']}` / extracted `{row['extracted_answer']}` / consistency `{row['original_solve_consistency']}` / tokens `{row['token_count']}` / finish `{row['finish_reason']}`",
                    "",
                    "```text",
                    row["content"] or "[no content captured]",
                    "```",
                    "",
                ]
            )
            if row["independent_attempts"]:
                lines.extend(["Independent attempts snapshot:", ""])
                for att in row["independent_attempts"][:4]:
                    lines.append(
                        f"- predicted `{att.get('predicted')}` / correct `{att.get('is_correct')}` / tokens `{att.get('token_count')}` / finish `{att.get('finish_reason')}`"
                    )
                lines.append("")

        (SAMPLES_DIR / f"teacher_digest_{condition}.md").write_text("\n".join(lines) + "\n")

    return chosen_rows


def write_report(
    run_summary: list[dict],
    condition_summary: list[dict],
    condition_linkage_rows: list[dict],
    behavior_rows: list[dict],
    baseline_rows: list[dict],
    baseline_cat_rows: list[dict],
    baseline_all_tag_rows: list[dict],
    teacher_rows: list[dict],
    teacher_tag_rows: list[dict],
    linkage_rows: list[dict],
    teacher_all_tag_rows: list[dict],
    all_tag_linkage_rows: list[dict],
    category_rows: list[dict],
    failure_rows: list[dict],
    failure_cat_rows: list[dict],
    pairwise_rows: list[dict],
    pairwise_summary: list[dict],
    records: list[dict],
) -> None:
    best_run = max(run_summary, key=lambda r: r["strict_accuracy_percent"])
    worst_run = min(run_summary, key=lambda r: r["strict_accuracy_percent"])

    solve_mean = safe_mean([r["mean_strict_accuracy_percent"] for r in condition_summary if r["condition"] in {"2a", "3a"}])
    rewrite_mean = safe_mean([r["mean_strict_accuracy_percent"] for r in condition_summary if r["condition"] in {"4a", "5a"}])

    failure_by_condition = defaultdict(list)
    for row in failure_rows:
        failure_by_condition[row["condition"]].append(row)

    guess_rows = [r for r in records if r["is_correct"] and r["maybe_guess"]]
    linkage_by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in linkage_rows:
        linkage_by_condition[row["condition"]].append(row)
    all_tag_linkage_by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in all_tag_linkage_rows:
        all_tag_linkage_by_condition[row["condition"]].append(row)
    condition_linkage_map = {r["condition"]: r for r in condition_linkage_rows}

    lines = [
        "# Qwen3-8B Deep Analysis",
        "",
        "## Scope",
        "",
        "- Conditions analyzed: `1`, `2a`, `3a`, `4a`, `5a`.",
        "- Seeds analyzed: `42`, `1337`, `2024`.",
        "- Test set: `71` held-out 2025 Känguru problems.",
        "- Teacher-side data: Qwen3-32B annotation files for direct solve (`2a`, `3a`) and rewrite (`4a`, `5a`).",
        "- Baseline data: repeated unchanged Qwen3-8B OpenRouter runs on the same held-out prompt setup.",
        "",
        "## High-Level Results",
        "",
        f"- Best run: `{best_run['condition']}_seed{best_run['seed']}` at `{best_run['strict_accuracy_percent']:.1f}%`.",
        f"- Worst run: `{worst_run['condition']}_seed{worst_run['seed']}` at `{worst_run['strict_accuracy_percent']:.1f}%`.",
        f"- Mean solve-style strict accuracy (`2a`,`3a`): `{solve_mean:.1f}%`.",
        f"- Mean rewrite-style strict accuracy (`4a`,`5a`): `{rewrite_mean:.1f}%`.",
        "",
        "## Condition Ranking",
        "",
    ]
    for idx, row in enumerate(condition_summary, start=1):
        lines.append(
            f"{idx}. `{row['condition']}`: mean strict `{row['mean_strict_accuracy_percent']:.1f}%`, sd `{row['sd_strict_accuracy_percent']:.1f}`, mean max-token hits `{row['mean_max_token_hits']:.1f}`"
        )

    lines.extend(
        [
            "",
            "## Annotation-To-Outcome Connection",
            "",
            "The table `condition_annotation_outcome_linkage.csv` directly connects annotation quality and annotation length to final 2025 student behavior. It is the cleanest summary for discussing why some conditions transfer better than others.",
            "",
        ]
    )
    for condition in ("1", "2a", "3a", "4a", "5a"):
        row = condition_linkage_map[condition]
        lines.append(
            f"- `{condition}`: train annotation acc `{row['train_annotation_accuracy_percent']:.1f}%`, train annotation mean tokens `{row['train_annotation_mean_token_count']:.1f}`, held-out annotation acc `{row['heldout_annotation_accuracy_percent']:.1f}%`, student strict `{row['student_mean_strict_accuracy_percent']:.1f}%`, student mean tokens `{row['student_mean_generated_tokens']:.1f}`, repetition `{row['student_repetition_failure_percent']:.1f}%`, no-extractable `{row['student_no_extractable_percent']:.1f}%`."
        )

    lines.extend(
        [
            "",
            "## 8B Unchanged Baseline",
            "",
            "The OpenRouter baseline runs use the same held-out prompt setup and therefore act as repeated estimates of a shared unchanged-model baseline. They still vary because provider-side nondeterminism and truncation behavior are not fully controlled.",
            "",
        ]
    )
    for row in baseline_rows:
        lines.append(
            f"- `{row['condition_label']}`: strict `{row['strict_accuracy_percent']:.1f}%`, scorable `{row['scorable_accuracy_percent']:.1f}%`, unscorable `{row['unscorable']}`, mean tokens `{row['mean_generated_tokens']:.1f}`"
        )

    lines.extend(
        [
            "",
            "## 32B Teacher Data",
            "",
            "The teacher-side data shows a clean difference between direct-solve and rewrite conditions. Direct-solve annotations (`2a`, `3a`) have measurable consistency and fallback behavior. Rewrite conditions are effectively correctness-preserving style transforms with much tighter control over the final answer.",
            "",
        ]
    )
    for row in teacher_rows:
        if row["scope"] in {"heldout_2025", "heldout_2025_independent_attempts"}:
            items = [f"`{k}`={v}" for k, v in row.items() if k not in {"condition", "scope"} and v is not None]
            lines.append(f"- `{row['condition']}` / `{row['scope']}`: " + ", ".join(items))

    lines.extend(
        [
            "",
            "## Teacher Tag Linkage By Primary Category",
            "",
            "This section connects student outcomes on each primary category to the held-out 2025 teacher behavior on the same category. These numbers align well with the primary-category casebooks, but they only use the first listed tag per problem.",
            "",
        ]
    )
    for condition in ("2a", "3a", "4a", "5a"):
        rows = linkage_by_condition[condition]
        xs_final = [r["teacher_final_accuracy_percent"] for r in rows]
        ys_student = [r["student_accuracy_percent"] for r in rows]
        corr_final = pearson(xs_final, ys_student)
        line = f"- `{condition}` teacher-final-vs-student tag correlation: `{corr_final:.2f}`." if corr_final is not None else f"- `{condition}` teacher-final-vs-student tag correlation: n/a."
        if condition in {"2a", "3a"}:
            xs_4of4 = [r["teacher_4of4_pass_percent"] for r in rows]
            corr_4of4 = pearson(xs_4of4, ys_student)
            line += f" 4/4-pass-vs-student correlation: `{corr_4of4:.2f}`." if corr_4of4 is not None else " 4/4-pass-vs-student correlation: n/a."
        lines.append(line)
        worst = sorted(rows, key=lambda r: (r["teacher_4of4_pass_percent"] if condition in {"2a", "3a"} else r["teacher_final_accuracy_percent"]))[:3]
        for row in worst:
            instability_label = "teacher_4of4" if condition in {"2a", "3a"} else "teacher_final"
            instability_value = row["teacher_4of4_pass_percent"] if condition in {"2a", "3a"} else row["teacher_final_accuracy_percent"]
            lines.append(
                f"  - `{row['primary_category']}`: student `{row['student_accuracy_percent']:.1f}%`, {instability_label} `{instability_value:.1f}%`, teacher failed `{row['teacher_failed_percent']:.1f}%`, hint `{row['teacher_hint_assisted_percent']:.1f}%`, teacher length-finish `{row['teacher_length_finish_percent']:.1f}%`."
            )

    lines.extend(
        [
            "",
            "## Teacher Tag Linkage Across All Attached Tags",
            "",
            "The next pass uses all tags attached to each problem, not just the first one. This is the stronger table for thesis discussion because many Känguru items are cross-tagged. Tag rows therefore overlap and should not be summed.",
            "",
        ]
    )
    for condition in ("2a", "3a", "4a", "5a"):
        rows = all_tag_linkage_by_condition[condition]
        if not rows:
            continue
        xs_final = [r["teacher_final_accuracy_percent"] for r in rows]
        ys_student = [r["student_accuracy_percent"] for r in rows]
        ys_gain = [r["student_minus_condition1_percent"] for r in rows if r["student_minus_condition1_percent"] is not None]
        xs_final_gain = [r["teacher_final_accuracy_percent"] for r in rows if r["student_minus_condition1_percent"] is not None]
        corr_final = pearson(xs_final, ys_student)
        corr_final_gain = pearson(xs_final_gain, ys_gain)
        line = f"- `{condition}` all-tag teacher-final-vs-student correlation: `{corr_final:.2f}`." if corr_final is not None else f"- `{condition}` all-tag teacher-final-vs-student correlation: n/a."
        if corr_final_gain is not None:
            line += f" Teacher-final-vs-gain-over-`1` correlation: `{corr_final_gain:.2f}`."
        if condition in {"2a", "3a"}:
            xs_4of4 = [r["teacher_4of4_pass_percent"] for r in rows]
            xs_4of4_gain = [r["teacher_4of4_pass_percent"] for r in rows if r["student_minus_condition1_percent"] is not None]
            corr_4of4 = pearson(xs_4of4, ys_student)
            corr_4of4_gain = pearson(xs_4of4_gain, ys_gain)
            line += f" 4/4-pass-vs-student correlation: `{corr_4of4:.2f}`." if corr_4of4 is not None else " 4/4-pass-vs-student correlation: n/a."
            if corr_4of4_gain is not None:
                line += f" 4/4-pass-vs-gain-over-`1` correlation: `{corr_4of4_gain:.2f}`."
        lines.append(line)

        if condition in {"2a", "3a"}:
            weakest_teacher = sorted(
                rows,
                key=lambda r: (
                    r["teacher_4of4_pass_percent"],
                    r["teacher_independent_attempt_failure_percent"] if r["teacher_independent_attempt_failure_percent"] is not None else -1,
                ),
            )[:3]
        else:
            weakest_teacher = sorted(rows, key=lambda r: (r["teacher_final_accuracy_percent"], -r["teacher_length_finish_percent"]))[:3]
        biggest_student_losses = sorted(rows, key=lambda r: (r["student_minus_condition1_percent"] if r["student_minus_condition1_percent"] is not None else float("inf")))[:3]

        lines.append("  - weakest teacher-signal tags:")
        for row in weakest_teacher:
            lines.append(
                f"    `{row['tag']}`: student `{row['student_accuracy_percent']:.1f}%`, gain-vs-`1` `{row['student_minus_condition1_percent']:.1f}%`, teacher final `{row['teacher_final_accuracy_percent']:.1f}%`, teacher 4/4 `{row['teacher_4of4_pass_percent']:.1f}%`, teacher independent-failure `{(row['teacher_independent_attempt_failure_percent'] or 0):.1f}%`."
            )
        lines.append("  - largest student losses vs `1`:")
        for row in biggest_student_losses:
            lines.append(
                f"    `{row['tag']}`: student `{row['student_accuracy_percent']:.1f}%`, baseline-8B `{row['baseline_8b_accuracy_percent']:.1f}%`, `1` `{row['condition1_accuracy_percent']:.1f}%`, teacher final `{row['teacher_final_accuracy_percent']:.1f}%`, teacher 4/4 `{row['teacher_4of4_pass_percent']:.1f}%`, teacher nonperfect-4try `{(row['teacher_nonperfect_4try_percent'] or 0):.1f}%`."
            )

    lines.extend(
        [
            "",
            "Interpretation of the held-out teacher data:",
            "",
            "- `2a_teacher_verbose` is unstable on 2025: many final rows are either outright failures or long incorrect traces, and the independent-attempt accuracy is much lower than the curated training split suggests. However, the all-tag correlations are not cleanly positive. The strongest interpretation is narrower: teacher instability clearly predicts the worst logic-heavy failures, but it does not explain the full tag ranking because the verbose teacher is itself unusually brittle on some tags where the student later performs well.",
            "- `3a_teacher_concise` is much more robust: it achieves high held-out teacher accuracy, but its failed hint-assisted cases still tend to run into the token ceiling. Its remaining student weaknesses are concentrated on tags where fallback is common and independent-solve stability is still limited.",
            "- Rewrite conditions separate correctness from learnability. `4a` keeps very high held-out teacher final accuracy across most tags but still collapses badly at the student level, so the bottleneck is not whether the teacher knew the answer but what kind of trace the student learned to imitate.",
            "- `5a` shows that verbose rewrites are more learnable than concise rewrites in this setup, but they still remain clearly below the direct-solve conditions on the most reasoning-sensitive tags.",
            "",
        ]
    )

    lines.extend(
        [
            "",
            "## Failure Patterns",
            "",
            "The heuristics are intentionally conservative. They tag visible artifacts in the reasoning trace rather than claiming access to the model's internal state. The most important recurring patterns are:",
            "",
            "- `max_token_hit`: generation ran into the 4096-token ceiling.",
            "- `repetition_loop`: the answer trace repeated long spans or sentences.",
            "- `template_placeholder_leak`: placeholders such as `<BUCHSTABE>` or `<ZIFFER>` survived into the answer.",
            "- `possible_option_mapping_error`: the response text may support the correct option value but map it to the wrong letter.",
            "- `reasoning_error_without_surface_artifact`: wrong answer without obvious formatting or truncation artifact.",
            "",
        ]
    )

    for condition in ("1", "2a", "3a", "4a", "5a"):
        rows = sorted(failure_by_condition[condition], key=lambda r: (-r["count"], r["primary_failure_tag"]))
        if not rows:
            continue
        top = ", ".join(f"{r['primary_failure_tag']} ({r['count']})" for r in rows[:5])
        lines.append(f"- `{condition}` top failure tags: {top}")

    lines.extend(
        [
            "",
            "## Guess-Risk Interpretation",
            "",
            f"- Correct traces flagged as possible guess-like responses: `{len(guess_rows)}` item-seed rows.",
            "- This flag is deliberately high-precision rather than high-recall. It only catches very short or linguistically uncertain traces and should be used as a qualitative warning signal, not as a definitive claim of guessing.",
            "",
            "## Category-Level Interpretation",
            "",
            "Primary-category tables remain useful for readable examples and casebook navigation. For quantitative discussion, prefer the all-tag linkage tables first and then inspect the primary-category casebooks for full traces.",
            "",
            "## Pairwise Comparison vs Condition 1",
            "",
        ]
    )
    for row in pairwise_summary:
        lines.append(
            f"- `{row['condition']}` vs `1`: wins `{row['wins_vs_condition_1']}`, losses `{row['losses_vs_condition_1']}`, ties-correct `{row['ties_correct']}`, ties-wrong `{row['ties_wrong']}`, exploratory sign-test p `{row['exploratory_sign_test_pvalue']:.4g}`."
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `tables/post_sft_trace_records.csv`: one row per condition-seed-item trace with failure tags.",
            "- `tables/post_sft_run_summary.csv`: one row per condition-seed run.",
            "- `tables/post_sft_condition_summary.csv`: mean and spread by condition.",
            "- `tables/student_condition_behavior_summary.csv`: repetition, extraction failure, and answer-length behavior by condition.",
            "- `tables/condition_annotation_outcome_linkage.csv`: direct join from annotation quality/length to final student outcomes by condition.",
            "- `tables/category_accuracy_summary.csv`: category-level accuracy and token stats.",
            "- `tables/failure_tag_counts.csv`: primary failure tag counts per condition.",
            "- `tables/failure_tag_counts_by_category.csv`: primary failure tag counts per condition-category pair.",
            "- `tables/teacher_annotation_summary.csv`: teacher-side annotation and attempt statistics.",
            "- `tables/teacher_tag_summary.csv`: held-out 2025 teacher instability by mathematical tag.",
            "- `tables/student_teacher_tag_linkage.csv`: student tag accuracy joined to teacher instability on the same tag.",
            "- `tables/baseline_qwen3_8b_summary.csv`: unchanged Qwen3-8B baseline summary.",
            "- `tables/baseline_qwen3_8b_category_summary.csv`: unchanged Qwen3-8B baseline by tag.",
            "- `tables/student_all_tag_accuracy_summary.csv`: student accuracy using all attached tags per problem.",
            "- `tables/teacher_all_tag_summary.csv`: teacher held-out 2025 stability using all attached tags per problem.",
            "- `tables/baseline_qwen3_8b_all_tag_summary.csv`: unchanged Qwen3-8B baseline using all attached tags.",
            "- `tables/student_teacher_all_tag_linkage.csv`: joined student, teacher, and baseline all-tag view with teacher four-try failure rates and student gains over `1`.",
            "- `tables/representative_example_index.csv`: compact index of the selected 7-example student and teacher digests.",
            "- `samples/trace_casebook_<condition>.md`: full trace samples with 2 correct and up to 5 incorrect traces per category and condition.",
            "- `samples/condition_digest_<condition>.md`: one representative run per condition with exactly 2 correct and 5 incorrect full traces.",
            "- `samples/teacher_trace_casebook_<teacher_condition>.md`: held-out 2025 teacher traces and failure examples from the 32B annotation JSONLs.",
            "- `samples/teacher_digest_<condition>.md`: compact 7-example teacher digest per condition.",
        ]
    )

    (NOTES_DIR / "DEEP_ANALYSIS_REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Qwen3-8B Deep Analysis",
                "",
                "This folder contains the detailed post-hoc analysis for the completed Qwen3-8B reasoning-verbosity experiment.",
                "",
                "## Main Report",
                "",
                "- `notes/DEEP_ANALYSIS_REPORT.md`",
                "",
                "## Core Tables",
                "",
                "- `tables/post_sft_trace_records.csv`",
                "- `tables/post_sft_run_summary.csv`",
                "- `tables/post_sft_condition_summary.csv`",
                "- `tables/student_condition_behavior_summary.csv`",
                "- `tables/condition_annotation_outcome_linkage.csv`",
                "- `tables/category_accuracy_summary.csv`",
                "- `tables/failure_tag_counts.csv`",
                "- `tables/failure_tag_counts_by_category.csv`",
                "- `tables/teacher_annotation_summary.csv`",
                "- `tables/teacher_tag_summary.csv`",
                "- `tables/student_teacher_tag_linkage.csv`",
                "- `tables/baseline_qwen3_8b_summary.csv`",
                "- `tables/baseline_qwen3_8b_category_summary.csv`",
                "- `tables/student_all_tag_accuracy_summary.csv`",
                "- `tables/teacher_all_tag_summary.csv`",
                "- `tables/baseline_qwen3_8b_all_tag_summary.csv`",
                "- `tables/student_teacher_all_tag_linkage.csv`",
                "- `tables/pairwise_item_vs_condition1.csv`",
                "- `tables/pairwise_item_vs_condition1_summary.csv`",
                "- `tables/representative_example_index.csv`",
                "",
                "## Trace Casebooks",
                "",
                "- `samples/trace_casebook_1.md`",
                "- `samples/trace_casebook_2a.md`",
                "- `samples/trace_casebook_3a.md`",
                "- `samples/trace_casebook_4a.md`",
                "- `samples/trace_casebook_5a.md`",
                "- `samples/condition_digest_1.md`",
                "- `samples/condition_digest_2a.md`",
                "- `samples/condition_digest_3a.md`",
                "- `samples/condition_digest_4a.md`",
                "- `samples/condition_digest_5a.md`",
                "- `samples/teacher_trace_casebook_2a_teacher_verbose.md`",
                "- `samples/teacher_trace_casebook_3a_teacher_concise.md`",
                "- `samples/teacher_trace_casebook_4a_rewrite_concise.md`",
                "- `samples/teacher_trace_casebook_5a_rewrite_verbose.md`",
                "- `samples/teacher_digest_2a.md`",
                "- `samples/teacher_digest_3a.md`",
                "- `samples/teacher_digest_4a.md`",
                "- `samples/teacher_digest_5a.md`",
            ]
        )
        + "\n"
    )


def main() -> None:
    ensure_dirs()
    source_rows = load_json(SOURCE_PATH)
    source_map = build_source_map()
    records = load_eval_records(source_map)
    baseline_records = load_baseline_records(source_map)

    run_summary = summarize_runs(records)
    condition_summary = summarize_conditions(run_summary)
    behavior_rows = student_condition_behavior_summary(records)
    category_rows = category_stats(records)
    failure_rows = failure_tag_counts(records)
    failure_cat_rows = failure_tag_counts_by_category(records)
    pairwise_rows = pairwise_item_comparisons(records)
    pairwise_summary = summarize_pairwise_item_comparisons(pairwise_rows)
    teacher_rows = teacher_summary(source_map)
    teacher_records = teacher_heldout_records(source_map)
    teacher_tag_rows = teacher_tag_summary(teacher_records)
    teacher_all_tag_rows = teacher_all_tag_summary(teacher_records)
    baseline_rows = baseline_summary(baseline_records)
    baseline_cat_rows = baseline_category_summary(baseline_records)
    baseline_all_tag_rows = baseline_all_tag_summary(baseline_records)
    linkage_rows = student_teacher_tag_linkage(category_rows, teacher_tag_rows)
    student_all_tag_rows = all_tag_accuracy_summary(records)
    all_tag_linkage_rows = student_teacher_all_tag_linkage(student_all_tag_rows, teacher_all_tag_rows, baseline_all_tag_rows)
    condition_linkage_rows = condition_annotation_outcome_linkage(condition_summary, behavior_rows, teacher_rows, source_rows)

    write_csv(TABLES_DIR / "post_sft_trace_records.csv", records)
    write_json(TABLES_DIR / "post_sft_trace_records.json", records)
    write_csv(TABLES_DIR / "post_sft_run_summary.csv", run_summary)
    write_json(TABLES_DIR / "post_sft_run_summary.json", run_summary)
    write_csv(TABLES_DIR / "post_sft_condition_summary.csv", condition_summary)
    write_json(TABLES_DIR / "post_sft_condition_summary.json", condition_summary)
    write_csv(TABLES_DIR / "student_condition_behavior_summary.csv", behavior_rows)
    write_json(TABLES_DIR / "student_condition_behavior_summary.json", behavior_rows)
    write_csv(TABLES_DIR / "condition_annotation_outcome_linkage.csv", condition_linkage_rows)
    write_json(TABLES_DIR / "condition_annotation_outcome_linkage.json", condition_linkage_rows)
    write_csv(TABLES_DIR / "category_accuracy_summary.csv", category_rows)
    write_json(TABLES_DIR / "category_accuracy_summary.json", category_rows)
    write_csv(TABLES_DIR / "failure_tag_counts.csv", failure_rows)
    write_json(TABLES_DIR / "failure_tag_counts.json", failure_rows)
    write_csv(TABLES_DIR / "failure_tag_counts_by_category.csv", failure_cat_rows)
    write_json(TABLES_DIR / "failure_tag_counts_by_category.json", failure_cat_rows)
    write_csv(TABLES_DIR / "pairwise_item_vs_condition1.csv", pairwise_rows)
    write_json(TABLES_DIR / "pairwise_item_vs_condition1.json", pairwise_rows)
    write_csv(TABLES_DIR / "pairwise_item_vs_condition1_summary.csv", pairwise_summary)
    write_json(TABLES_DIR / "pairwise_item_vs_condition1_summary.json", pairwise_summary)
    write_csv(TABLES_DIR / "teacher_annotation_summary.csv", teacher_rows)
    write_json(TABLES_DIR / "teacher_annotation_summary.json", teacher_rows)
    write_csv(TABLES_DIR / "teacher_tag_summary.csv", teacher_tag_rows)
    write_json(TABLES_DIR / "teacher_tag_summary.json", teacher_tag_rows)
    write_csv(TABLES_DIR / "student_teacher_tag_linkage.csv", linkage_rows)
    write_json(TABLES_DIR / "student_teacher_tag_linkage.json", linkage_rows)
    write_csv(TABLES_DIR / "baseline_qwen3_8b_summary.csv", baseline_rows)
    write_json(TABLES_DIR / "baseline_qwen3_8b_summary.json", baseline_rows)
    write_csv(TABLES_DIR / "baseline_qwen3_8b_category_summary.csv", baseline_cat_rows)
    write_json(TABLES_DIR / "baseline_qwen3_8b_category_summary.json", baseline_cat_rows)
    write_csv(TABLES_DIR / "student_all_tag_accuracy_summary.csv", student_all_tag_rows)
    write_json(TABLES_DIR / "student_all_tag_accuracy_summary.json", student_all_tag_rows)
    write_csv(TABLES_DIR / "teacher_all_tag_summary.csv", teacher_all_tag_rows)
    write_json(TABLES_DIR / "teacher_all_tag_summary.json", teacher_all_tag_rows)
    write_csv(TABLES_DIR / "baseline_qwen3_8b_all_tag_summary.csv", baseline_all_tag_rows)
    write_json(TABLES_DIR / "baseline_qwen3_8b_all_tag_summary.json", baseline_all_tag_rows)
    write_csv(TABLES_DIR / "student_teacher_all_tag_linkage.csv", all_tag_linkage_rows)
    write_json(TABLES_DIR / "student_teacher_all_tag_linkage.json", all_tag_linkage_rows)

    trace_casebook(records)
    teacher_trace_casebook(source_map)
    representative_rows = condition_example_digest(records, run_summary, condition_summary)
    representative_rows += teacher_example_digest(source_map)
    write_csv(TABLES_DIR / "representative_example_index.csv", representative_rows)
    write_json(TABLES_DIR / "representative_example_index.json", representative_rows)
    write_report(
        run_summary,
        condition_summary,
        condition_linkage_rows,
        behavior_rows,
        baseline_rows,
        baseline_cat_rows,
        baseline_all_tag_rows,
        teacher_rows,
        teacher_tag_rows,
        linkage_rows,
        teacher_all_tag_rows,
        all_tag_linkage_rows,
        category_rows,
        failure_rows,
        failure_cat_rows,
        pairwise_rows,
        pairwise_summary,
        records,
    )


if __name__ == "__main__":
    main()
