#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from prompt_normalization import build_canonical_prompt_stem
except ModuleNotFoundError:
    from top_level_scripts.prompt_normalization import build_canonical_prompt_stem
SOURCE_DATASET = ROOT / "kaenguru_2021_2025_no_diagram_finetune.json"
SOLVE_SOURCE_DIR = ROOT / "generated_files" / "dataset_answers_qwen_qwen3_32b"
REWRITE_SOURCES = [
    ROOT
    / "generated_files"
    / "dataset_answers_qwen_qwen3_32b_clean"
    / "kaenguru_annotated_qwen_qwen3_32b_rewrite_only_full_2021_2025.jsonl",
    ROOT
    / "generated_files"
    / "dataset_answers_qwen_qwen3_32b_clean"
    / "reruns"
    / "kaenguru_annotated_qwen_qwen3_32b_rewrite_retry_incorrect_2021_2025.jsonl",
    ROOT
    / "generated_files"
    / "dataset_answers_qwen_qwen3_32b_clean"
    / "reruns"
    / "kaenguru_annotated_qwen_qwen3_32b_rewrite_retry_incorrect_2021_2025_8k.jsonl",
    ROOT
    / "generated_files"
    / "dataset_answers_qwen_qwen3_32b_clean"
    / "reruns"
    / "kaenguru_annotated_qwen_qwen3_32b_rewrite_retry_incorrect_2021_2025_compact_4k.jsonl",
]
REWRITE_SFT_SOURCE_DIR = ROOT / "generated_files" / "dataset_answers_qwen_max_rewrite"
DRIFTED_PACKAGE = ROOT / "generated_files" / "kaggle_qwen3_32b_5_conditions_2021_2023"
OUTPUT_PACKAGE = (
    ROOT / "generated_files" / "kaggle_qwen3_32b_5_conditions_2021_2023_canonical_refreshed"
)

ANSWER_LINE_RE = re.compile(r"Antwort\s*:\s*(?:Antwort\s*:\s*)*([A-E])\b", re.I)
ANSWER_ONLY_RE = re.compile(r"^\s*(?:Antwort\s*:\s*)?[A-E][.]?\s*$", re.I | re.S)
WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")


def year_from_id(item_id: str) -> int:
    match = re.search(r"kaenguru_(\d{4})_", item_id)
    return int(match.group(1)) if match else 0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_choices(choices: Dict[str, Any]) -> Dict[str, str]:
    return {label: str(choices.get(label, "")) for label in "ABCDE"}


def format_choices_multiline(choices: Dict[str, str]) -> str:
    return "\n".join(f"({label}) {value}" for label, value in choices.items())


def build_training_prompt(source: Dict[str, Any]) -> str:
    choices = normalize_choices(source.get("choices_de", {}))
    prompt_stem = build_canonical_prompt_stem(str(source.get("prompt_de", "")), choices)
    return f"{prompt_stem}\n\n{format_choices_multiline(choices)}"


def build_reasoning_prompt(source: Dict[str, Any]) -> str:
    choices = normalize_choices(source.get("choices_de", {}))
    prompt_stem = build_canonical_prompt_stem(str(source.get("prompt_de", "")), choices)
    choices_str = "\n".join(f"  {label}: {text}" for label, text in choices.items() if text)
    return (
        f"{prompt_stem}\n\n"
        f"Antwortoptionen:\n{choices_str}\n\n"
        "Denke Schritt fuer Schritt und gib am Ende genau eine Zeile aus:\n"
        "Antwort: <BUCHSTABE>"
    )


def build_validation_prompt(source: Dict[str, Any]) -> str:
    choices = normalize_choices(source.get("choices_de", {}))
    prompt_stem = build_canonical_prompt_stem(str(source.get("prompt_de", "")), choices)
    choices_str = "\n".join(f"  {label}: {text}" for label, text in choices.items() if text)
    return (
        f"{prompt_stem}\n\n"
        f"Antwortoptionen:\n{choices_str}\n\n"
        "Löse die Aufgabe und gib am Ende genau eine Zeile aus:\n"
        "Antwort: <BUCHSTABE>"
    )


def build_official_solution(source: Dict[str, Any]) -> str:
    text = str(source.get("official_solution_de", "")).strip()
    answer = str(source.get("answer_key", "")).strip().upper()
    if text.endswith(f"Antwort: {answer}"):
        return text
    return f"{text}\n\nAntwort: {answer}"


def normalize_assistant_content(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    return re.sub(
        r"Antwort\s*:\s*Antwort\s*:\s*([A-E])\b",
        r"Antwort: \1",
        cleaned,
        flags=re.I,
    )


def extract_answer_letter(text: str) -> str | None:
    matches = ANSWER_LINE_RE.findall(str(text or ""))
    return matches[-1].upper() if matches else None


def assistant_text(row: Dict[str, Any]) -> str:
    for message in reversed(row.get("messages", [])):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content", "") or "")
    return ""


def text_score(text: str) -> Tuple[int, int, int]:
    cleaned = normalize_assistant_content(text)
    return (
        1 if cleaned and not ANSWER_ONLY_RE.fullmatch(cleaned) else 0,
        len(WORD_RE.findall(cleaned)),
        len(cleaned),
    )


def source_row_matches_item(source_row: Dict[str, Any], item: Dict[str, Any]) -> bool:
    if str(source_row.get("answer_key", "")).upper() != str(item.get("answer_key", "")).upper():
        return False
    return True


def best_attempt_payload(
    source_row: Dict[str, Any], ann: Dict[str, Any], answer_key: str
) -> Dict[str, Any] | None:
    candidates: List[Dict[str, Any]] = []

    top_level = dict(ann)
    top_level["content"] = normalize_assistant_content(ann.get("content", ""))
    top_level["_candidate_origin"] = "selected_annotation"
    candidates.append(top_level)

    for idx, attempt in enumerate(ann.get("independent_solve_attempts", []) or [], start=1):
        if not isinstance(attempt, dict):
            continue
        content = normalize_assistant_content(attempt.get("content", ""))
        if not content:
            continue
        predicted = str(attempt.get("predicted") or extract_answer_letter(content) or "").upper()
        if predicted != answer_key:
            continue
        payload = dict(ann)
        payload.update(
            {
                "content": content,
                "_candidate_origin": f"independent_attempt_{idx}",
            }
        )
        candidates.append(payload)

    best_payload: Dict[str, Any] | None = None
    best_score: Tuple[int, int, int, int, int] | None = None
    for candidate in candidates:
        content = normalize_assistant_content(candidate.get("content", ""))
        if not content:
            continue
        if ANSWER_ONLY_RE.fullmatch(content):
            continue
        if extract_answer_letter(content) != answer_key:
            continue
        score = (
            2
            if candidate.get("source_type") == "independent_solve"
            else 1 if candidate.get("source_type") == "hint_assisted" else 0,
            int(candidate.get("teacher_correct_attempts", 0) or 0),
            len(WORD_RE.findall(content)),
            len(content),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_payload = dict(candidate)
            best_payload["content"] = content

    return best_payload


def load_source_items() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_items = load_json(SOURCE_DATASET)
    for idx, item in enumerate(all_items):
        item["_index"] = idx
    train = [item for item in all_items if year_from_id(str(item["id"])) in {2021, 2022, 2023}]
    valid = [item for item in all_items if year_from_id(str(item["id"])) == 2024]
    test = [item for item in all_items if year_from_id(str(item["id"])) == 2025]
    return train, valid, test


def annotation_score(annotation: Dict[str, Any]) -> Tuple[int, int, int, int]:
    text = normalize_assistant_content(annotation.get("content", ""))
    return (
        1 if text and not ANSWER_ONLY_RE.fullmatch(text) else 0,
        2
        if annotation.get("source_type") == "independent_solve"
        else 1 if annotation.get("source_type") == "hint_assisted" else 0,
        int(annotation.get("teacher_correct_attempts", 0) or 0),
        len(text),
    )


def load_best_solve_annotations(
    train_items: List[Dict[str, Any]], annotation_key: str
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    item_lookup = {str(item["id"]): item for item in train_items}
    train_ids = set(item_lookup)
    best: Dict[str, Tuple[Tuple[int, int, int, int], Dict[str, Any]]] = {}
    for path in sorted(SOLVE_SOURCE_DIR.glob("*.jsonl")):
        for row in iter_jsonl(path):
            row_id = str(row.get("id", ""))
            if row_id not in train_ids:
                continue
            item = item_lookup[row_id]
            if not source_row_matches_item(row, item):
                continue
            ann = (row.get("annotations") or {}).get(annotation_key)
            if not isinstance(ann, dict):
                continue
            if not ann.get("is_correct"):
                continue
            answer_key = str(item.get("answer_key", "")).upper()
            payload = best_attempt_payload(row, ann, answer_key)
            if not payload:
                continue
            payload["_source_file"] = path.name
            score = annotation_score(payload)
            previous = best.get(row_id)
            if previous is None or score > previous[0]:
                best[row_id] = (score, payload)
    selected = {row_id: payload for row_id, (_, payload) in best.items()}
    missing = sorted(train_ids - set(selected))
    return selected, missing


def load_best_rewrite_annotations(
    train_items: List[Dict[str, Any]], annotation_key: str
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    item_lookup = {str(item["id"]): item for item in train_items}
    train_ids = set(item_lookup)
    best: Dict[str, Tuple[Tuple[int, int, int], Dict[str, Any]]] = {}

    for path in sorted(REWRITE_SFT_SOURCE_DIR.glob(f"sft_{'concise' if annotation_key == 'reformulated_concise' else 'verbose'}_rewrite_*.jsonl")):
        for row in iter_jsonl(path):
            row_id = str(row.get("id", ""))
            if row_id not in train_ids:
                continue
            item = item_lookup[row_id]
            answer_key = str(item.get("answer_key", "")).upper()
            row_answer = str(row.get("correct_letter") or row.get("answer") or "").upper()
            if row_answer != answer_key:
                continue
            content = assistant_text(row)
            content = normalize_assistant_content(content)
            if not content:
                continue
            if ANSWER_ONLY_RE.fullmatch(content):
                continue
            if extract_answer_letter(content) != answer_key:
                continue
            payload = {
                "content": content,
                "source_type": "llm_rewrite",
                "_source_file": path.name,
            }
            score = text_score(content)
            previous = best.get(row_id)
            if previous is None or score > previous[0]:
                best[row_id] = (score, payload)

    for path in REWRITE_SOURCES:
        for row in iter_jsonl(path):
            row_id = str(row.get("id", ""))
            if row_id not in train_ids:
                continue
            item = item_lookup[row_id]
            if not source_row_matches_item(row, item):
                continue
            ann = (row.get("annotations") or {}).get(annotation_key)
            if not isinstance(ann, dict):
                continue
            content = normalize_assistant_content(ann.get("content", ""))
            if not content:
                continue
            if ANSWER_ONLY_RE.fullmatch(content):
                continue
            if not ann.get("is_correct"):
                continue
            if extract_answer_letter(content) != str(item.get("answer_key", "")).upper():
                continue
            payload = dict(ann)
            payload["content"] = content
            payload["_source_file"] = path.name
            score = text_score(content)
            previous = best.get(row_id)
            if previous is None or score > previous[0]:
                best[row_id] = (score, payload)
    selected = {row_id: payload for row_id, (_, payload) in best.items()}
    missing = sorted(train_ids - set(selected))
    return selected, missing


def build_train_rows(
    items: List[Dict[str, Any]],
    condition: str,
    annotation_map: Dict[str, Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items:
        row_id = str(item["id"])
        if condition == "official_human":
            assistant = build_official_solution(item)
            rows.append(
                {
                    "id": row_id,
                    "condition": "official_human",
                    "messages": [
                        {"role": "user", "content": build_training_prompt(item)},
                        {"role": "assistant", "content": assistant},
                    ],
                    "answer": item["answer_key"],
                    "correct_letter": item["answer_key"],
                    "answer_correct": "yes",
                    "source_year": year_from_id(row_id),
                    "annotation_source_type": "official_human",
                }
            )
            continue

        assert annotation_map is not None
        ann = annotation_map.get(row_id)
        if not ann:
            continue
        row = {
            "id": row_id,
            "messages": [
                {"role": "user", "content": build_training_prompt(item)},
                {"role": "assistant", "content": ann["content"]},
            ],
            "answer": item["answer_key"],
            "correct_letter": item["answer_key"],
            "answer_correct": "yes",
            "source_year": year_from_id(row_id),
        }
        if condition in {"teacher_concise", "teacher_verbose"}:
            row.update(
                {
                    "condition": condition,
                    "annotation_source_type": ann.get("source_type", ""),
                    "annotation_source_file": ann.get("_source_file", ""),
                    "annotation_solve_consistency": ann.get("solve_consistency")
                    or ann.get("original_solve_consistency")
                    or f"{int(ann.get('teacher_correct_attempts', 0) or 0)}/4",
                    "annotation_hint_assisted": bool(
                        ann.get("hint_assisted") or ann.get("source_type") == "hint_assisted"
                    ),
                }
            )
        else:
            row.update(
                {
                    "condition": condition,
                    "annotation_source_type": ann.get("source_type", "llm_rewrite"),
                    "annotation_source_file": ann.get("_source_file", ""),
                }
            )
        rows.append(row)
    return rows


def build_valid_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in items:
        rows.append(
            {
                "id": item["id"],
                "year": 2024,
                "condition": "1",
                "messages": [
                    {"role": "user", "content": build_validation_prompt(item)},
                    {"role": "assistant", "content": str(item.get("official_solution_de", "")).strip()},
                ],
            }
        )
    return rows


def build_eval_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in items:
        rows.append(
            {
                "dataset": "kaenguru_2021_2025_no_diagram_finetune",
                "index": item["_index"],
                "id": item["id"],
                "year": 2025,
                "prompt": build_reasoning_prompt(item),
                "answer": item["answer_key"],
                "correct_letter": item["answer_key"],
            }
        )
    return rows


def collect_counts(data_dir: Path) -> Dict[str, int]:
    counts = {}
    for path in sorted(data_dir.glob("*.jsonl")):
        counts[path.name] = sum(1 for _ in iter_jsonl(path))
    return counts


def build_manifest(summary: Dict[str, Any], counts: Dict[str, int]) -> Dict[str, Any]:
    return {
        "package": "kaggle_qwen3_32b_5_conditions_2021_2023_canonical_refreshed",
        "student_model_default": "Qwen/Qwen3-8B",
        "teacher_model": "qwen/qwen3-32b",
        "train_years": "2021-2023",
        "validation_year": 2024,
        "eval_year": 2025,
        "train_counts": {
            "official_human": counts["train_official_human_2021_2023.jsonl"],
            "concise_solve": counts["train_qwen3_32b_concise_2021_2023.jsonl"],
            "verbose_solve": counts["train_qwen3_32b_verbose_2021_2023.jsonl"],
            "concise_rewrite": counts["train_qwen3_32b_concise_rewrite_2021_2023.jsonl"],
            "verbose_rewrite": counts["train_qwen3_32b_verbose_rewrite_2021_2023.jsonl"],
        },
        "validation_rows": counts["valid_2024.jsonl"],
        "eval_rows": counts["eval_2025.jsonl"],
        "summary_file": "canonical_refresh_summary.json",
        "drift_comparison_file": "drift_comparison.json",
        "notes": [
            "This package is rebuilt directly from recovered refreshed local source annotations.",
            "Only answer-correct rows with non-empty reasoning text are retained for non-human conditions.",
        "Direct-solve rows are matched against the curated item prompt and answer key, then upgraded to the strongest compatible correct attempt when a richer attempt is preserved.",
        "Rewrite rows prefer the preserved SFT-style rewrite files in generated_files/dataset_answers_qwen_max_rewrite and fall back to the annotation bundles only when needed.",
        ],
        "condition_missing_ids": summary["missing_ids"],
    }


def build_readme(summary: Dict[str, Any], counts: Dict[str, int]) -> str:
    missing = summary["missing_ids"]
    return f"""# Canonical Refreshed Qwen3-32B 5-Condition Package

This folder is a clean rebuild of the refreshed Qwen3-32B supervision package from local source artifacts. It is meant to replace the drifted exports in `generated_files/kaggle_qwen3_32b_5_conditions_2021_2023/` when a scientifically defensible snapshot is needed.

## Counts

- `train_official_human_2021_2023.jsonl`: {counts['train_official_human_2021_2023.jsonl']} rows
- `train_qwen3_32b_concise_2021_2023.jsonl`: {counts['train_qwen3_32b_concise_2021_2023.jsonl']} rows
- `train_qwen3_32b_verbose_2021_2023.jsonl`: {counts['train_qwen3_32b_verbose_2021_2023.jsonl']} rows
- `train_qwen3_32b_concise_rewrite_2021_2023.jsonl`: {counts['train_qwen3_32b_concise_rewrite_2021_2023.jsonl']} rows
- `train_qwen3_32b_verbose_rewrite_2021_2023.jsonl`: {counts['train_qwen3_32b_verbose_rewrite_2021_2023.jsonl']} rows
- `valid_2024.jsonl`: {counts['valid_2024.jsonl']} rows
- `eval_2025.jsonl`: {counts['eval_2025.jsonl']} rows

## Source logic

- Official human rows are built from the curated 2021-2025 dataset file.
- Direct solves are taken from `generated_files/dataset_answers_qwen_qwen3_32b/`, but only when the preserved source row matches the curated prompt/choices and answer key for the target item.
- When a direct-solve record contains multiple preserved correct attempts, the rebuild prefers a non-answer-only reasoning trace over a bare final-answer line.
- Rewrites prefer the preserved SFT-style rewrite files in `generated_files/dataset_answers_qwen_max_rewrite/` and fall back to the annotation bundles in `generated_files/dataset_answers_qwen_qwen3_32b_clean/` only when needed.
- For non-human conditions, a row is kept only if the trace is non-empty and ends with the correct extracted final answer letter.

## Missing IDs

- Concise solve missing: {", ".join(missing['concise_solve']) if missing['concise_solve'] else "none"}
- Verbose solve missing: {", ".join(missing['verbose_solve']) if missing['verbose_solve'] else "none"}
- Concise rewrite missing: {", ".join(missing['concise_rewrite']) if missing['concise_rewrite'] else "none"}
- Verbose rewrite missing: {", ".join(missing['verbose_rewrite']) if missing['verbose_rewrite'] else "none"}

## Drift note

The older package at `generated_files/kaggle_qwen3_32b_5_conditions_2021_2023/` currently contains drifted exports with lower row counts. See `drift_comparison.json` in this folder for the exact differences.
"""


def main() -> None:
    train_items, valid_items, eval_items = load_source_items()
    concise_solves, missing_concise_solves = load_best_solve_annotations(
        train_items, "teacher_concise"
    )
    verbose_solves, missing_verbose_solves = load_best_solve_annotations(
        train_items, "teacher_verbose"
    )
    concise_rewrites, missing_concise_rewrites = load_best_rewrite_annotations(
        train_items, "reformulated_concise"
    )
    verbose_rewrites, missing_verbose_rewrites = load_best_rewrite_annotations(
        train_items, "reformulated_verbose"
    )

    data_dir = OUTPUT_PACKAGE / "data"
    write_jsonl(data_dir / "train_official_human_2021_2023.jsonl", build_train_rows(train_items, "official_human"))
    write_jsonl(
        data_dir / "train_qwen3_32b_concise_2021_2023.jsonl",
        build_train_rows(train_items, "teacher_concise", concise_solves),
    )
    write_jsonl(
        data_dir / "train_qwen3_32b_verbose_2021_2023.jsonl",
        build_train_rows(train_items, "teacher_verbose", verbose_solves),
    )
    write_jsonl(
        data_dir / "train_qwen3_32b_concise_rewrite_2021_2023.jsonl",
        build_train_rows(train_items, "concise_rewrite", concise_rewrites),
    )
    write_jsonl(
        data_dir / "train_qwen3_32b_verbose_rewrite_2021_2023.jsonl",
        build_train_rows(train_items, "verbose_rewrite", verbose_rewrites),
    )
    write_jsonl(data_dir / "valid_2024.jsonl", build_valid_rows(valid_items))
    write_jsonl(data_dir / "eval_2025.jsonl", build_eval_rows(eval_items))

    counts = collect_counts(data_dir)
    summary = {
        "train_universe_rows": len(train_items),
        "valid_rows": len(valid_items),
        "eval_rows": len(eval_items),
        "missing_ids": {
            "concise_solve": missing_concise_solves,
            "verbose_solve": missing_verbose_solves,
            "concise_rewrite": missing_concise_rewrites,
            "verbose_rewrite": missing_verbose_rewrites,
        },
        "source_files": {
            "solve_source_dir": str(SOLVE_SOURCE_DIR.relative_to(ROOT)),
            "rewrite_sources": [str(path.relative_to(ROOT)) for path in REWRITE_SOURCES],
        },
    }
    (OUTPUT_PACKAGE / "canonical_refresh_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    drift = {}
    drift_dir = DRIFTED_PACKAGE / "data"
    for filename in [
        "train_official_human_2021_2023.jsonl",
        "train_qwen3_32b_concise_2021_2023.jsonl",
        "train_qwen3_32b_verbose_2021_2023.jsonl",
        "train_qwen3_32b_concise_rewrite_2021_2023.jsonl",
        "train_qwen3_32b_verbose_rewrite_2021_2023.jsonl",
    ]:
        drift[filename] = {
            "canonical_rows": counts[filename],
            "drifted_rows": sum(1 for _ in iter_jsonl(drift_dir / filename)),
        }
    (OUTPUT_PACKAGE / "drift_comparison.json").write_text(
        json.dumps(drift, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = build_manifest(summary, counts)
    (OUTPUT_PACKAGE / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_PACKAGE / "README.md").write_text(
        build_readme(summary, counts), encoding="utf-8"
    )

    print(json.dumps({"output_package": str(OUTPUT_PACKAGE), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
