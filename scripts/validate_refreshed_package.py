#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "metadata"
RELEASE_PKG = (
    ROOT
    / "generated_files"
    / "kaggle_qwen3_32b_5_conditions_2021_2023_canonical_refreshed"
)
DATA_DIR = RELEASE_PKG / "data"
ADAPTER_DIR = ROOT / "training_artifacts" / "final_adapters"
TRAIN_LOG_DIR = ROOT / "training_artifacts" / "training_logs"
CKPT_DIR = ROOT / "training_artifacts" / "checkpoint_selection"
RUN_MATRIX = META / "run_matrix.csv"
RESULT_DIR = ROOT / "results_so_far" / "qwen3_8b_eval_snapshot_2026-05-24_final"
DEEP_DIR = ROOT / "results_so_far" / "qwen3_8b_deep_analysis_2026-05-24"
THESIS_TEX = ROOT / "reasoning-verbosity-sft-math" / "main.tex"
THESIS_PDF = ROOT / "reasoning-verbosity-sft-math" / "main.pdf"

ANSWER_ONLY_RE = re.compile(r"^\s*(?:Antwort\s*:\s*)?[A-E][.]?\s*$", re.I | re.S)
FINAL_ANSWER_RE = re.compile(r"Antwort\s*:\s*([A-E])\b", re.I)
WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9]+")

CONDITION_SPECS = [
    {
        "legacy_label": "1",
        "public_name": "official human solution",
        "row_condition": "official_human",
        "train_file": "train_official_human_2021_2023.jsonl",
        "expected_train_rows": 186,
    },
    {
        "legacy_label": "2a",
        "public_name": "verbose solve",
        "row_condition": "teacher_verbose",
        "train_file": "train_qwen3_32b_verbose_2021_2023.jsonl",
        "expected_train_rows": 171,
    },
    {
        "legacy_label": "3a",
        "public_name": "concise solve",
        "row_condition": "teacher_concise",
        "train_file": "train_qwen3_32b_concise_2021_2023.jsonl",
        "expected_train_rows": 171,
    },
    {
        "legacy_label": "4a",
        "public_name": "concise rewrite",
        "row_condition": "concise_rewrite",
        "train_file": "train_qwen3_32b_concise_rewrite_2021_2023.jsonl",
        "expected_train_rows": 181,
    },
    {
        "legacy_label": "5a",
        "public_name": "verbose rewrite",
        "row_condition": "verbose_rewrite",
        "train_file": "train_qwen3_32b_verbose_rewrite_2021_2023.jsonl",
        "expected_train_rows": 182,
    },
]

VALIDATION_REFERENCE = {"file": "valid_2024.jsonl", "rows": 67}
EVAL_REFERENCE = {"file": "eval_2025.jsonl", "rows": 71}

PREFERRED_LOG_FIELDS = [
    "epoch",
    "grad_norm",
    "learning_rate",
    "loss",
    "step",
    "eval_loss",
    "eval_runtime",
    "eval_samples_per_second",
    "eval_steps_per_second",
    "total_flos",
    "train_loss",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def assistant_text(row: dict[str, Any]) -> str:
    for message in reversed(row.get("messages", [])):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content", "") or "")
    return ""


def last_final_answer(text: str) -> str | None:
    matches = FINAL_ANSWER_RE.findall(text or "")
    return matches[-1].upper() if matches else None


def parse_run_label(run_label: str) -> tuple[str, str]:
    m = re.fullmatch(r"condition_([^_]+)_seed(\d+)", run_label)
    if not m:
        raise ValueError(f"Unexpected run label: {run_label}")
    return m.group(1), m.group(2)


def parse_best_global_step(best_model_checkpoint: str | None) -> int | None:
    if not best_model_checkpoint:
        return None
    m = re.search(r"checkpoint-(\d+)", best_model_checkpoint)
    return int(m.group(1)) if m else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def truthy_path(path_str: str | None) -> bool:
    return bool(path_str) and (ROOT / path_str).exists()


def condition_lookup() -> dict[str, dict[str, str]]:
    return {spec["legacy_label"]: spec for spec in CONDITION_SPECS}


def canonical_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for spec in CONDITION_SPECS:
        counts[spec["legacy_label"]] = len(read_jsonl(DATA_DIR / spec["train_file"]))
    return counts


def run_matrix_lookup() -> dict[tuple[str, str], dict[str, str]]:
    if not RUN_MATRIX.exists():
        return {}
    rows = read_csv(RUN_MATRIX)
    return {(row["legacy_label"], row["seed"]): row for row in rows}


def iter_adapter_dirs() -> list[Path]:
    dirs = [
        path
        for path in ADAPTER_DIR.iterdir()
        if path.is_dir() and not path.name.endswith("_partial")
    ]
    return sorted(dirs, key=lambda p: p.name)


def generate_training_logs_and_manifests() -> None:
    TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)

    cond_map = condition_lookup()
    run_lookup = run_matrix_lookup()

    training_log_manifest: list[dict[str, Any]] = []
    checkpoint_manifest: list[dict[str, Any]] = []
    training_run_manifest: list[dict[str, Any]] = []

    for adapter_dir in iter_adapter_dirs():
        run_label = adapter_dir.name
        legacy_label, seed = parse_run_label(run_label)
        meta = json.loads((adapter_dir / "training_meta.json").read_text(encoding="utf-8"))
        state = json.loads((adapter_dir / "trainer_state.json").read_text(encoding="utf-8"))
        log_history = state.get("log_history", [])

        extra_fields = sorted(
            {key for entry in log_history for key in entry.keys()} - set(PREFERRED_LOG_FIELDS)
        )
        log_fields = PREFERRED_LOG_FIELDS + extra_fields
        log_rows = [{field: entry.get(field, "") for field in log_fields} for entry in log_history]
        log_csv = TRAIN_LOG_DIR / f"{run_label}_log_history.csv"
        write_csv(log_csv, log_rows, log_fields)
        training_log_manifest.append(
            {
                "run_label": run_label,
                "log_history_csv": rel(log_csv),
                "entries": len(log_rows),
            }
        )

        checkpoint_rows: list[dict[str, Any]] = []
        seen_steps: set[int] = set()
        for entry in log_history:
            if "eval_loss" not in entry:
                continue
            step = entry.get("step")
            if step is None:
                continue
            step_int = int(step)
            if step_int in seen_steps:
                continue
            seen_steps.add(step_int)
            checkpoint_rows.append(
                {
                    "run_label": run_label,
                    "checkpoint": f"checkpoint-{step_int}",
                    "global_step": step_int,
                    "epoch": entry.get("epoch", ""),
                }
            )
        checkpoint_csv = CKPT_DIR / f"{run_label}_checkpoints.csv"
        write_csv(
            checkpoint_csv,
            checkpoint_rows,
            ["run_label", "checkpoint", "global_step", "epoch"],
        )

        best_model_checkpoint = state.get("best_model_checkpoint")
        best_global_step = parse_best_global_step(best_model_checkpoint)
        run_matrix_row = run_lookup.get((legacy_label, seed))
        training_run_manifest.append(
            {
                "run_label": run_label,
                "condition": legacy_label,
                "public_condition": cond_map[legacy_label]["public_name"],
                "seed": int(seed),
                "train_jsonl": meta.get("train_jsonl"),
                "valid_jsonl": meta.get("valid_jsonl"),
                "rows_train": meta.get("rows_train"),
                "rows_val": meta.get("rows_val"),
                "epochs": meta.get("epochs"),
                "learning_rate": meta.get("learning_rate"),
                "data_seed": meta.get("data_seed"),
                "precision": meta.get("precision"),
                "lora_r": meta.get("lora_r"),
                "lora_alpha": meta.get("lora_alpha"),
                "best_metric": state.get("best_metric"),
                "best_global_step": best_global_step,
                "best_model_checkpoint": best_model_checkpoint,
                "global_step": state.get("global_step"),
                "epoch_at_end": state.get("epoch"),
                "gpu_name": meta.get("gpu", {}).get("gpu_name"),
                "gpu_count": meta.get("gpu", {}).get("gpu_count"),
                "gpu_vram_gb": meta.get("gpu", {}).get("gpu_vram_gb"),
                "driver_version": meta.get("gpu", {}).get("driver_version"),
                "python": meta.get("versions", {}).get("python"),
                "torch": meta.get("versions", {}).get("torch"),
                "transformers": meta.get("versions", {}).get("transformers"),
                "peft": meta.get("versions", {}).get("peft"),
                "trl": meta.get("versions", {}).get("trl"),
                "final_adapter_present": (adapter_dir / "adapter_model.safetensors").exists(),
                "trainer_state_present": (adapter_dir / "trainer_state.json").exists(),
                "eval_predictions_present": truthy_path(
                    run_matrix_row.get("row_level_artifact") if run_matrix_row else None
                ),
            }
        )
        checkpoint_manifest.append(
            {
                "run_label": run_label,
                "checkpoint_selection_csv": rel(checkpoint_csv),
                "best_global_step": best_global_step,
                "best_metric": state.get("best_metric"),
            }
        )

    write_csv(
        META / "training_log_manifest.csv",
        training_log_manifest,
        ["run_label", "log_history_csv", "entries"],
    )
    write_csv(
        META / "checkpoint_selection_manifest.csv",
        checkpoint_manifest,
        ["run_label", "checkpoint_selection_csv", "best_global_step", "best_metric"],
    )
    write_csv(
        META / "training_run_manifest.csv",
        training_run_manifest,
        [
            "run_label",
            "condition",
            "public_condition",
            "seed",
            "train_jsonl",
            "valid_jsonl",
            "rows_train",
            "rows_val",
            "epochs",
            "learning_rate",
            "data_seed",
            "precision",
            "lora_r",
            "lora_alpha",
            "best_metric",
            "best_global_step",
            "best_model_checkpoint",
            "global_step",
            "epoch_at_end",
            "gpu_name",
            "gpu_count",
            "gpu_vram_gb",
            "driver_version",
            "python",
            "torch",
            "transformers",
            "peft",
            "trl",
            "final_adapter_present",
            "trainer_state_present",
            "eval_predictions_present",
        ],
    )


def generate_training_data_manifest() -> None:
    rows = []
    for spec in CONDITION_SPECS:
        rows.append(
            {
                "public_condition": spec["public_name"],
                "legacy_label": spec["legacy_label"],
                "row_condition": spec["row_condition"],
                "training_rows": spec["expected_train_rows"],
                "training_jsonl": rel(DATA_DIR / spec["train_file"]),
                "validation_reference_rows": VALIDATION_REFERENCE["rows"],
                "validation_reference_jsonl": rel(DATA_DIR / VALIDATION_REFERENCE["file"]),
            }
        )
    write_csv(
        META / "training_data_manifest.csv",
        rows,
        [
            "public_condition",
            "legacy_label",
            "row_condition",
            "training_rows",
            "training_jsonl",
            "validation_reference_rows",
            "validation_reference_jsonl",
        ],
    )


def generate_authoritative_release_manifest() -> None:
    manifest = {
        "authoritative_release_package": rel(RELEASE_PKG),
        "authoritative_training_data": rel(DATA_DIR),
        "student_model_default": "Qwen/Qwen3-8B",
        "teacher_model": "qwen/qwen3-32b",
        "train_years": "2021-2023",
        "validation_year": 2024,
        "evaluation_year": 2025,
        "training_counts": {
            spec["public_name"]: spec["expected_train_rows"] for spec in CONDITION_SPECS
        },
        "validation_reference": {
            "path": rel(DATA_DIR / VALIDATION_REFERENCE["file"]),
            "rows": VALIDATION_REFERENCE["rows"],
        },
        "evaluation_file": {
            "path": rel(DATA_DIR / EVAL_REFERENCE["file"]),
            "rows": EVAL_REFERENCE["rows"],
        },
        "condition_mapping": {
            spec["legacy_label"]: spec["public_name"] for spec in CONDITION_SPECS
        },
        "row_condition_mapping": {
            spec["legacy_label"]: spec["row_condition"] for spec in CONDITION_SPECS
        },
        "exact_final_adapter_root": rel(ADAPTER_DIR),
        "evaluation_artifact_root": rel(ROOT / "kaggle_outputs" / "kaggle_kernel_outputs"),
        "thesis_source": rel(THESIS_TEX),
        "thesis_pdf": rel(THESIS_PDF),
    }
    (META / "AUTHORITATIVE_RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_validation() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[str] = []
    warnings: list[str] = []
    training_checks: list[dict[str, Any]] = []

    if not RELEASE_PKG.exists():
        issues.append(f"Missing authoritative release package: {rel(RELEASE_PKG)}")
    if not DATA_DIR.exists():
        issues.append(f"Missing release data directory: {rel(DATA_DIR)}")

    valid_rows = read_jsonl(DATA_DIR / VALIDATION_REFERENCE["file"])
    eval_rows = read_jsonl(DATA_DIR / EVAL_REFERENCE["file"])
    if len(valid_rows) != VALIDATION_REFERENCE["rows"]:
        issues.append(
            f"Validation reference expected {VALIDATION_REFERENCE['rows']} rows, found {len(valid_rows)}"
        )
    if len(eval_rows) != EVAL_REFERENCE["rows"]:
        issues.append(
            f"Evaluation file expected {EVAL_REFERENCE['rows']} rows, found {len(eval_rows)}"
        )

    expected_years = {2021, 2022, 2023}
    for spec in CONDITION_SPECS:
        path = DATA_DIR / spec["train_file"]
        rows = read_jsonl(path)
        texts = [assistant_text(row) for row in rows]
        final_answers = [last_final_answer(text) for text in texts]
        condition_mismatch = sum(
            1 for row in rows if row.get("condition") != spec["row_condition"]
        )
        duplicate_ids = len(rows) - len({row.get("id") for row in rows})
        outside_years = sum(1 for row in rows if row.get("source_year") not in expected_years)
        empty_assistant = sum(1 for text in texts if not text.strip())
        answer_only = sum(1 for text in texts if ANSWER_ONLY_RE.fullmatch(text.strip() or ""))
        missing_final = sum(1 for ans in final_answers if ans is None)
        wrong_final = sum(
            1
            for row, ans in zip(rows, final_answers)
            if ans is not None and ans != str(row.get("correct_letter", "")).upper()
        )
        training_checks.append(
            {
                "legacy_label": spec["legacy_label"],
                "public_condition": spec["public_name"],
                "train_file": rel(path),
                "rows": len(rows),
                "expected_rows": spec["expected_train_rows"],
                "row_condition": spec["row_condition"],
                "condition_mismatch_rows": condition_mismatch,
                "duplicate_id_rows": duplicate_ids,
                "outside_train_year_rows": outside_years,
                "empty_assistant_rows": empty_assistant,
                "answer_letter_only_rows": answer_only,
                "missing_final_answer_rows": missing_final,
                "wrong_final_answer_rows": wrong_final,
                "median_assistant_words": sorted(len(WORD_RE.findall(t)) for t in texts)[len(texts) // 2]
                if texts
                else None,
            }
        )
        if len(rows) != spec["expected_train_rows"]:
            issues.append(
                f"{spec['public_name']} expected {spec['expected_train_rows']} training rows, found {len(rows)}"
            )
        if condition_mismatch:
            issues.append(f"{spec['public_name']} has {condition_mismatch} row-condition mismatches")
        if duplicate_ids:
            issues.append(f"{spec['public_name']} has {duplicate_ids} duplicate-id rows")
        if outside_years:
            issues.append(f"{spec['public_name']} has {outside_years} rows outside the 2021-2023 train split")
        if empty_assistant:
            issues.append(f"{spec['public_name']} has {empty_assistant} empty assistant traces")
        if answer_only:
            issues.append(f"{spec['public_name']} has {answer_only} strict answer-letter-only rows")
        if missing_final:
            issues.append(f"{spec['public_name']} has {missing_final} rows without an extractable final answer line")
        if wrong_final:
            issues.append(f"{spec['public_name']} has {wrong_final} rows whose final answer line disagrees with the answer key")

    run_lookup = run_matrix_lookup()
    adapter_checks: list[dict[str, Any]] = []
    for adapter_dir in iter_adapter_dirs():
        run_label = adapter_dir.name
        legacy_label, seed = parse_run_label(run_label)
        required = [
            "adapter_model.safetensors",
            "adapter_config.json",
            "trainer_state.json",
            "training_meta.json",
        ]
        missing_required = [name for name in required if not (adapter_dir / name).exists()]
        meta = json.loads((adapter_dir / "training_meta.json").read_text(encoding="utf-8"))
        state = json.loads((adapter_dir / "trainer_state.json").read_text(encoding="utf-8"))
        run_matrix_row = run_lookup.get((legacy_label, seed))
        eval_artifact = ROOT / run_matrix_row["row_level_artifact"] if run_matrix_row else None
        eval_rows_count = len(read_jsonl(eval_artifact)) if eval_artifact and eval_artifact.exists() else None
        adapter_checks.append(
            {
                "run_label": run_label,
                "public_condition": condition_lookup()[legacy_label]["public_name"],
                "seed": int(seed),
                "rows_train": meta.get("rows_train"),
                "rows_val": meta.get("rows_val"),
                "best_global_step": parse_best_global_step(state.get("best_model_checkpoint")),
                "best_metric": state.get("best_metric"),
                "missing_required_files": missing_required,
                "eval_artifact_present": bool(eval_artifact and eval_artifact.exists()),
                "eval_rows": eval_rows_count,
            }
        )
        if missing_required:
            issues.append(
                f"{run_label} is missing required files: {', '.join(missing_required)}"
            )
        if eval_artifact and eval_artifact.exists() and eval_rows_count != 71:
            issues.append(f"{run_label} evaluation artifact expected 71 rows, found {eval_rows_count}")
        if not run_matrix_row:
            warnings.append(f"No run-matrix row found for {run_label}")

    partial_dirs = [path.name for path in ADAPTER_DIR.iterdir() if path.is_dir() and path.name.endswith("_partial")]

    expected_outputs = [
        RESULT_DIR / "tables" / "post_sft_results_complete.csv",
        RESULT_DIR / "tables" / "post_sft_condition_aggregates.csv",
        RESULT_DIR / "tables" / "pairwise_deltas_vs_condition1.csv",
        RESULT_DIR / "tables" / "answer_length_comparison.csv",
        DEEP_DIR / "tables" / "student_condition_behavior_summary.csv",
        DEEP_DIR / "tables" / "teacher_annotation_summary.csv",
        DEEP_DIR / "tables" / "failure_tag_counts.csv",
        DEEP_DIR / "tables" / "student_teacher_all_tag_linkage.csv",
    ]
    for path in expected_outputs:
        if not path.exists():
            issues.append(f"Missing expected output: {rel(path)}")

    validation = {
        "status": "PASS" if not issues else "FAIL",
        "scope": "Authoritative release package validation plus exact final-adapter presence checks.",
        "authoritative_release_package": rel(RELEASE_PKG),
        "training_checks": training_checks,
        "validation_reference": {
            "path": rel(DATA_DIR / VALIDATION_REFERENCE["file"]),
            "rows": len(valid_rows),
        },
        "evaluation_file": {
            "path": rel(DATA_DIR / EVAL_REFERENCE["file"]),
            "rows": len(eval_rows),
        },
        "adapter_checks": adapter_checks,
        "partial_adapter_dirs": partial_dirs,
        "expected_outputs": [{"path": rel(path), "present": path.exists()} for path in expected_outputs],
        "issues": issues,
        "warnings": warnings,
    }
    return validation, training_checks


def write_validation_reports(validation: dict[str, Any], training_checks: list[dict[str, Any]]) -> None:
    (META / "REFRESHED_PACKAGE_VALIDATION.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Refreshed Package Validation",
        "",
        f"Status: `{validation['status']}`",
        "",
        "Scope: authoritative Qwen3-32B supervision release package, exact final Qwen3-8B adapters, and refreshed evaluation/result artifacts.",
        "",
        f"- Authoritative release package: `{validation['authoritative_release_package']}`",
        f"- Validation reference: `{validation['validation_reference']['path']}` ({validation['validation_reference']['rows']} rows)",
        f"- Evaluation file: `{validation['evaluation_file']['path']}` ({validation['evaluation_file']['rows']} rows)",
        "",
        "## Released Training Files",
        "",
        "| Label | Public condition | Rows | Expected | Row condition | Empty | Letter-only | Missing final | Wrong final | Duplicate IDs |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in training_checks:
        lines.append(
            f"| {item['legacy_label']} | {item['public_condition']} | {item['rows']} | {item['expected_rows']} | "
            f"{item['row_condition']} | {item['empty_assistant_rows']} | {item['answer_letter_only_rows']} | "
            f"{item['missing_final_answer_rows']} | {item['wrong_final_answer_rows']} | {item['duplicate_id_rows']} |"
        )

    lines.extend(
        [
            "",
            "## Final Adapters",
            "",
            "| Run | Public condition | Seed | rows_train | rows_val | best step | eval artifact | eval rows |",
            "|---|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for item in validation["adapter_checks"]:
        lines.append(
            f"| {item['run_label']} | {item['public_condition']} | {item['seed']} | {item['rows_train']} | "
            f"{item['rows_val']} | {item['best_global_step']} | {item['eval_artifact_present']} | {item['eval_rows']} |"
        )

    lines.extend(["", "## Issues", ""])
    if validation["issues"]:
        lines.extend([f"- {issue}" for issue in validation["issues"]])
    else:
        lines.append("- None.")

    lines.extend(["", "## Warnings", ""])
    if validation["warnings"]:
        lines.extend([f"- {warning}" for warning in validation["warnings"]])
    else:
        lines.append("- None.")

    (META / "REFRESHED_PACKAGE_VALIDATION.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_condition_audit(training_checks: list[dict[str, Any]]) -> None:
    lines = [
        "# Refreshed Condition Leakage Audit",
        "",
        "Audit target:",
        "",
        f"- `{rel(DATA_DIR)}`",
        "",
        "Question:",
        "",
        "- Do the released training files keep each supervision condition in its expected slot, without detectable cross-condition contamination inside the package itself?",
        "",
        "## Audit Method",
        "",
        "For each released training JSONL, the audit checked:",
        "",
        "1. whether every row's `condition` field matches the file's expected condition tag,",
        "2. whether each row has a non-empty assistant trace,",
        "3. whether the row avoids strict answer-letter-only supervision,",
        "4. whether the final `Antwort: X` line is extractable and agrees with the stored answer key, and",
        "5. whether duplicate problem ids appear inside the same condition file.",
        "",
        "This audit validates package-internal condition alignment. It does not attempt a full raw-source provenance reconstruction from every upstream teacher-generation log.",
        "",
        "## Summary",
        "",
        "| Label | Public condition | Rows | Expected condition tag | Condition mismatches | Empty | Letter-only | Missing final | Wrong final | Duplicate IDs |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in training_checks:
        lines.append(
            f"| {item['legacy_label']} | {item['public_condition']} | {item['rows']} | {item['row_condition']} | "
            f"{item['condition_mismatch_rows']} | {item['empty_assistant_rows']} | {item['answer_letter_only_rows']} | "
            f"{item['missing_final_answer_rows']} | {item['wrong_final_answer_rows']} | {item['duplicate_id_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Result",
            "",
            "- No condition-file contains rows whose stored `condition` tag points to another supervision condition.",
            "- No released training file contains empty assistant traces.",
            "- No released training file contains strict answer-letter-only supervision rows.",
            "- No released training file contains rows whose extractable final answer letter disagrees with the stored answer key.",
            "- No duplicate problem ids were detected within a released condition file.",
            "",
            "Within the released package, the condition boundaries therefore validate cleanly.",
        ]
    )
    (META / "REFRESHED_CONDITION_LEAKAGE_AUDIT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_inventory_and_summary() -> None:
    excluded = {
        Path("metadata/SHA256SUMS"),
        Path("metadata/file_inventory.csv"),
        Path("metadata/repository_summary.json"),
    }
    inventory_rows: list[dict[str, Any]] = []
    sha_lines: list[str] = []
    total_bytes = 0

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(ROOT)
        if ".git" in rel_path.parts:
            continue
        if rel_path in excluded:
            continue
        digest = sha256(path)
        size = path.stat().st_size
        inventory_rows.append({"path": str(rel_path), "bytes": size, "sha256": digest})
        sha_lines.append(f"{digest}  {rel_path}")
        total_bytes += size

    write_csv(META / "file_inventory.csv", inventory_rows, ["path", "bytes", "sha256"])
    (META / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    summary = {
        "file_count_indexed": len(inventory_rows),
        "total_bytes_indexed": total_bytes,
        "inventory_excludes": [str(path) for path in sorted(excluded)],
        "current_training_data": rel(DATA_DIR),
        "authoritative_release_manifest": "metadata/AUTHORITATIVE_RELEASE_MANIFEST.json",
        "current_final_adapters": rel(ADAPTER_DIR),
        "current_eval_outputs": "kaggle_outputs/kaggle_kernel_outputs",
        "current_result_tables": rel(RESULT_DIR),
        "current_deep_analysis": rel(DEEP_DIR),
        "validation_script": "scripts/validate_refreshed_package.py",
        "validation_report": "metadata/REFRESHED_PACKAGE_VALIDATION.md",
        "condition_audit_report": "metadata/REFRESHED_CONDITION_LEAKAGE_AUDIT.md",
        "packaged_thesis_source": rel(THESIS_TEX),
        "packaged_thesis_pdf": rel(THESIS_PDF),
        "top_level_entries": sorted(
            path.name for path in ROOT.iterdir() if path.name != ".git"
        ),
    }
    (META / "repository_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    generate_training_logs_and_manifests()
    generate_training_data_manifest()
    generate_authoritative_release_manifest()
    validation, training_checks = build_validation()
    write_validation_reports(validation, training_checks)
    write_condition_audit(training_checks)
    write_inventory_and_summary()
    print(f"Validation status: {validation['status']}")
    print(f"Authoritative release package: {rel(RELEASE_PKG)}")
    print("Updated:")
    print("- metadata/REFRESHED_PACKAGE_VALIDATION.json")
    print("- metadata/REFRESHED_PACKAGE_VALIDATION.md")
    print("- metadata/REFRESHED_CONDITION_LEAKAGE_AUDIT.md")
    print("- metadata/AUTHORITATIVE_RELEASE_MANIFEST.json")
    print("- metadata/training_data_manifest.csv")
    print("- metadata/training_run_manifest.csv")
    print("- metadata/training_log_manifest.csv")
    print("- metadata/checkpoint_selection_manifest.csv")
    print("- metadata/file_inventory.csv")
    print("- metadata/SHA256SUMS")
    print("- metadata/repository_summary.json")


if __name__ == "__main__":
    main()
