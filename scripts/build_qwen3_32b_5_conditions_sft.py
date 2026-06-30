#!/usr/bin/env python3

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = REPO_ROOT / "kaenguru_2021_2025_no_diagram_finetune.json"
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from prompt_normalization import build_canonical_prompt_stem

EXCLUDED_2021_2023_IDS = {
    "kaenguru_2022_klasse11-13_q9",
    "kaenguru_2022_klasse11-13_q11",
    "kaenguru_2022_klasse11-13_q28",
}

OUTPUT_DIR = (
    REPO_ROOT / "generated_files" / "kaggle_qwen3_32b_5_conditions_2021_2023" / "data"
)
REPORT_DIR = REPO_ROOT / "generated_files" / "reports"

ANSWER_LINE_RE = re.compile(r"Antwort\s*:\s*(?:Antwort\s*:\s*)*([A-E])\b", re.I)


def year_from_id(item_id: str) -> int:
    match = re.search(r"kaenguru_(\d{4})_", item_id)
    return int(match.group(1)) if match else 0


def normalize_choices(choices: Dict[str, Any]) -> Dict[str, str]:
    return {k: str(choices.get(k, "")) for k in "ABCDE"}


def format_choices_multiline(choices: Dict[str, Any]) -> str:
    return "\n".join(f"({k}) {v}" for k, v in normalize_choices(choices).items())


def build_training_prompt(source: Dict[str, Any]) -> str:
    choices = normalize_choices(source.get("choices_de", {}))
    prompt_stem = build_canonical_prompt_stem(str(source.get("prompt_de", "")), choices)
    return f"{prompt_stem}\n\n{format_choices_multiline(choices)}"


def build_official_solution(source: Dict[str, Any]) -> str:
    text = str(source.get("official_solution_de", "")).strip()
    answer = str(source.get("answer_key", "")).strip().upper()
    if text.endswith(f"Antwort: {answer}"):
        return text
    return f"{text}\n\nAntwort: {answer}"


def extract_answer_letter(text: str) -> str | None:
    matches = ANSWER_LINE_RE.findall(str(text or ""))
    return matches[-1].upper() if matches else None


def normalize_assistant_content(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return cleaned
    return re.sub(
        r"Antwort\s*:\s*Antwort\s*:\s*([A-E])\b",
        r"Antwort: \1",
        cleaned,
        flags=re.I,
    )


def ensure_letter_answer_line(text: str, answer_key: str) -> str:
    cleaned = normalize_assistant_content(text)
    answer_key = str(answer_key or "").strip().upper()
    if not cleaned or not answer_key:
        return cleaned
    if extract_answer_letter(cleaned) == answer_key:
        return cleaned
    return f"{cleaned}\n\nAntwort: {answer_key}"


def load_source_items() -> List[Dict[str, Any]]:
    items = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    return [
        item
        for item in items
        if year_from_id(str(item["id"])) in {2021, 2022, 2023}
        and str(item["id"]) not in EXCLUDED_2021_2023_IDS
    ]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def solve_annotation_rank(annotation: Dict[str, Any]) -> Tuple[int, int, int, int]:
    source_type = str(annotation.get("source_type", ""))
    return (
        1 if annotation.get("is_correct") else 0,
        2 if source_type == "independent_solve" else 1 if source_type == "hint_assisted" else 0,
        int(annotation.get("teacher_correct_attempts", 0) or 0),
        len(str(annotation.get("content", "") or "")),
    )


def rewrite_annotation_rank(annotation: Dict[str, Any]) -> Tuple[int, int, int, int]:
    finish_reason = str(annotation.get("finish_reason", "") or "")
    content = str(annotation.get("content", "") or "")
    return (
        1 if annotation.get("is_correct") else 0,
        1 if finish_reason == "stop" else 0,
        0 if annotation.get("source_type") == "failed" else 1,
        len(content),
    )


def best_solve_annotations(
    source_ids: set[str], condition: str
) -> Dict[str, Tuple[Dict[str, Any], str]]:
    best: Dict[str, Tuple[Tuple[int, int, int, int], Dict[str, Any], str]] = {}
    directories = [
        REPO_ROOT / "generated_files" / "dataset_answers_qwen_qwen3_32b",
        REPO_ROOT / "generated_files" / "dataset_answers_qwen_qwen3_32b_clean",
    ]
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            for row in iter_jsonl(path):
                row_id = str(row.get("id", ""))
                if row_id not in source_ids:
                    continue
                annotation = row.get("annotations", {}).get(condition)
                if not isinstance(annotation, dict):
                    continue
                score = solve_annotation_rank(annotation)
                previous = best.get(row_id)
                if previous is None or score > previous[0]:
                    best[row_id] = (score, annotation, path.name)
    return {row_id: (annotation, name) for row_id, (_, annotation, name) in best.items()}


def best_rewrite_annotations(
    source_ids: set[str], condition: str
) -> Dict[str, Tuple[Dict[str, Any], str]]:
    best: Dict[str, Tuple[Tuple[int, int, int, int], Dict[str, Any], str]] = {}
    files = [
        REPO_ROOT
        / "generated_files"
        / "dataset_answers_qwen_qwen3_32b_clean"
        / "kaenguru_annotated_qwen_qwen3_32b_rewrite_only_full_2021_2025.jsonl",
        REPO_ROOT
        / "generated_files"
        / "dataset_answers_qwen_qwen3_32b_clean"
        / "reruns"
        / "kaenguru_annotated_qwen_qwen3_32b_rewrite_retry_incorrect_2021_2025_compact_4k.jsonl",
        REPO_ROOT
        / "generated_files"
        / "dataset_answers_qwen_qwen3_32b_clean"
        / "reruns"
        / "kaenguru_annotated_qwen_qwen3_32b_rewrite_retry_incorrect_2021_2025_8k.jsonl",
    ]
    for path in files:
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            row_id = str(row.get("id", ""))
            if row_id not in source_ids:
                continue
            annotation = row.get("annotations", {}).get(condition)
            if not isinstance(annotation, dict):
                continue
            score = rewrite_annotation_rank(annotation)
            previous = best.get(row_id)
            if previous is None or score > previous[0]:
                best[row_id] = (score, annotation, path.name)
    return {row_id: (annotation, name) for row_id, (_, annotation, name) in best.items()}


def build_official_rows(source_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for src in source_items:
        rows.append(
            {
                "id": src["id"],
                "condition": "official_human",
                "messages": [
                    {"role": "user", "content": build_training_prompt(src)},
                    {"role": "assistant", "content": build_official_solution(src)},
                ],
                "answer": src["answer_key"],
                "correct_letter": src["answer_key"],
                "answer_correct": "yes",
                "source_year": year_from_id(str(src["id"])),
                "annotation_source_type": "official_human",
            }
        )
    return rows


def build_solve_rows(
    source_items: List[Dict[str, Any]],
    annotations_by_id: Dict[str, Tuple[Dict[str, Any], str]],
    *,
    condition: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for src in source_items:
        row_id = str(src["id"])
        annotation_info = annotations_by_id.get(row_id)
        if not annotation_info or not annotation_info[0].get("is_correct"):
            missing.append(row_id)
            continue
        annotation, source_file = annotation_info
        attempts = int(annotation.get("teacher_correct_attempts", 0) or 0)
        consistency = annotation.get("original_solve_consistency") or f"{attempts}/4"
        source_type = str(annotation.get("source_type", ""))
        rows.append(
            {
                "id": row_id,
                "condition": condition,
                "messages": [
                    {"role": "user", "content": build_training_prompt(src)},
                    {
                        "role": "assistant",
                        "content": ensure_letter_answer_line(
                            str(annotation.get("content", "")).strip(),
                            str(src["answer_key"]),
                        ),
                    },
                ],
                "answer": src["answer_key"],
                "correct_letter": src["answer_key"],
                "answer_correct": "yes",
                "source_year": year_from_id(row_id),
                "annotation_source_type": source_type,
                "annotation_solve_consistency": consistency,
                "annotation_hint_assisted": bool(
                    annotation.get("hint_assisted") or source_type == "hint_assisted"
                ),
                "annotation_source_file": source_file,
                "annotation_seed": annotation.get("seed"),
            }
        )
    return rows, missing


def build_rewrite_rows(
    source_items: List[Dict[str, Any]],
    annotations_by_id: Dict[str, Tuple[Dict[str, Any], str]],
    *,
    condition: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for src in source_items:
        row_id = str(src["id"])
        annotation_info = annotations_by_id.get(row_id)
        if not annotation_info or not annotation_info[0].get("is_correct"):
            missing.append(row_id)
            continue
        annotation, source_file = annotation_info
        content = ensure_letter_answer_line(
            str(annotation.get("content", "")).strip(), str(src["answer_key"])
        )
        if not content or extract_answer_letter(content) != str(src["answer_key"]).upper():
            missing.append(row_id)
            continue
        rows.append(
            {
                "id": row_id,
                "condition": condition,
                "messages": [
                    {"role": "user", "content": build_training_prompt(src)},
                    {"role": "assistant", "content": content},
                ],
                "answer": src["answer_key"],
                "correct_letter": src["answer_key"],
                "answer_correct": "yes",
                "source_year": year_from_id(row_id),
                "annotation_source_type": str(annotation.get("source_type", "")),
                "annotation_finish_reason": annotation.get("finish_reason"),
                "annotation_source_file": source_file,
                "annotation_seed": annotation.get("seed"),
                "model": "qwen/qwen3-32b",
            }
        )
    return rows, missing


def write_readme(counts: Dict[str, int], missing: Dict[str, List[str]]) -> None:
    pkg_dir = OUTPUT_DIR.parent
    text = f"""# Qwen3-32B Five-Condition SFT Package (2021-2023)

This package contains the five qwen3-32b training conditions for `2021-2023`:

- `1`: official human control
- `2a`: from-scratch verbose solve (`teacher_verbose`)
- `3a`: from-scratch concise solve (`teacher_concise`)
- `4a`: concise rewrite (`reformulated_concise`)
- `5a`: verbose rewrite (`reformulated_verbose`)

Selection rules:

- Source problems come from `kaenguru_2021_2025_no_diagram_finetune.json`.
- Only years `2021-2023` are included.
- Excluded IDs:
  - `kaenguru_2022_klasse11-13_q9`
  - `kaenguru_2022_klasse11-13_q11`
  - `kaenguru_2022_klasse11-13_q28`
- Solve conditions use only qwen3-32b same-condition annotations from:
  - `generated_files/dataset_answers_qwen_qwen3_32b/`
  - `generated_files/dataset_answers_qwen_qwen3_32b_clean/`
- Rewrite conditions use only qwen3-32b same-condition annotations from:
  - baseline rewrite-only full run
  - compact `4k` retry
  - partial `8k` retry where it rescued additional valid targets

Row counts:

- `train_official_human_2021_2023.jsonl`: {counts['official_human']} rows
- `train_qwen3_32b_verbose_2021_2023.jsonl`: {counts['teacher_verbose']} rows
- `train_qwen3_32b_concise_2021_2023.jsonl`: {counts['teacher_concise']} rows
- `train_qwen3_32b_concise_rewrite_2021_2023.jsonl`: {counts['reformulated_concise']} rows
- `train_qwen3_32b_verbose_rewrite_2021_2023.jsonl`: {counts['reformulated_verbose']} rows

Missing-but-excluded from learned conditions:

- `2a`: {len(missing['teacher_verbose'])}
- `3a`: {len(missing['teacher_concise'])}
- `4a`: {len(missing['reformulated_concise'])}
- `5a`: {len(missing['reformulated_verbose'])}
"""
    (pkg_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    source_items = load_source_items()
    source_ids = {str(item["id"]) for item in source_items}

    best_verbose = best_solve_annotations(source_ids, "teacher_verbose")
    best_concise = best_solve_annotations(source_ids, "teacher_concise")
    best_rewrite_concise = best_rewrite_annotations(source_ids, "reformulated_concise")
    best_rewrite_verbose = best_rewrite_annotations(source_ids, "reformulated_verbose")

    official_rows = build_official_rows(source_items)
    verbose_rows, verbose_missing = build_solve_rows(
        source_items, best_verbose, condition="teacher_verbose"
    )
    concise_rows, concise_missing = build_solve_rows(
        source_items, best_concise, condition="teacher_concise"
    )
    rewrite_concise_rows, rewrite_concise_missing = build_rewrite_rows(
        source_items, best_rewrite_concise, condition="concise_rewrite"
    )
    rewrite_verbose_rows, rewrite_verbose_missing = build_rewrite_rows(
        source_items, best_rewrite_verbose, condition="verbose_rewrite"
    )

    counts = {
        "official_human": write_jsonl(
            OUTPUT_DIR / "train_official_human_2021_2023.jsonl", official_rows
        ),
        "teacher_verbose": write_jsonl(
            OUTPUT_DIR / "train_qwen3_32b_verbose_2021_2023.jsonl", verbose_rows
        ),
        "teacher_concise": write_jsonl(
            OUTPUT_DIR / "train_qwen3_32b_concise_2021_2023.jsonl", concise_rows
        ),
        "reformulated_concise": write_jsonl(
            OUTPUT_DIR / "train_qwen3_32b_concise_rewrite_2021_2023.jsonl",
            rewrite_concise_rows,
        ),
        "reformulated_verbose": write_jsonl(
            OUTPUT_DIR / "train_qwen3_32b_verbose_rewrite_2021_2023.jsonl",
            rewrite_verbose_rows,
        ),
    }

    missing = {
        "teacher_verbose": verbose_missing,
        "teacher_concise": concise_missing,
        "reformulated_concise": rewrite_concise_missing,
        "reformulated_verbose": rewrite_verbose_missing,
    }

    report = {
        "source_file": str(SOURCE_PATH.relative_to(REPO_ROOT)),
        "source_rows_2021_2023": len(source_items),
        "excluded_ids": sorted(EXCLUDED_2021_2023_IDS),
        "counts": counts,
        "missing_ids": missing,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "qwen3_32b_5_conditions_2021_2023_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(counts, missing)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
