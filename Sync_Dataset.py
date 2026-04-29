from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def env_path(name: str, default: Optional[Path] = None) -> Optional[Path]:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    return default


SOURCE_VIDEO_FOLDER_ENV = "ESPEX_SOURCE_VIDEO_DIR"
FORMS_XLSX_ENV = "ESPEX_FORMS_XLSX"
LOCAL_VIDEO_DATA_ENV = "ESPEX_VIDEO_DATA_DIR"
LOCAL_DATASET_CSV_ENV = "ESPEX_DATASET_CSV"
MANIFEST_ENV = "ESPEX_IMPORT_MANIFEST"

ONEDRIVE_VIDEO_FOLDER = env_path(SOURCE_VIDEO_FOLDER_ENV)
FORMS_XLSX_PATH = env_path(FORMS_XLSX_ENV)
LOCAL_VIDEO_DATA_FOLDER = env_path(LOCAL_VIDEO_DATA_ENV, BASE_DIR / "Video Data")
LOCAL_DATASET_CSV = env_path(LOCAL_DATASET_CSV_ENV, BASE_DIR / "dataset.csv")

# Stores info about imported rows/files
MANIFEST_PATH = env_path(MANIFEST_ENV, LOCAL_VIDEO_DATA_FOLDER / "_forms_imported.json")

# Allowed video file extensions
VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".wmv", ".webm"
}


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def find_column(df: pd.DataFrame, target_name: str) -> Optional[str]:
    target_norm = re.sub(r"\s+", " ", target_name.strip().lower())
    for col in df.columns:
        col_norm = re.sub(r"\s+", " ", str(col).strip().lower())
        if col_norm == target_norm:
            return col
    return None


def clean_filename(filename: str) -> str:
    text = str(filename).strip()
    if not text:
        return ""

    parsed = urlparse(text)
    if parsed.scheme and parsed.path:
        text = parsed.path

    text = unquote(text)
    return Path(text).name.strip()


def canonical_video_name(filename: str) -> str:
    cleaned = clean_filename(filename)
    if not cleaned:
        return ""

    stem = Path(cleaned).stem.strip()
    suffix = Path(cleaned).suffix.lower()
    canonical_stem = re.sub(r"(?:\s+\d+|\s*\(\d+\))$", "", stem).strip()
    return f"{canonical_stem}{suffix}".lower()


def _flatten_items(value):
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _flatten_items(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten_items(item)
    else:
        yield value


def parse_upload_cell(cell_value: object) -> list[str]:
    text = normalize_text(cell_value)
    if not text:
        return []

    candidates: list[str] = []

    def maybe_add(name: str) -> None:
        name = clean_filename(name)
        if Path(name).suffix.lower() in VIDEO_EXTENSIONS and name not in candidates:
            candidates.append(name)

    # 1) Direct regex extraction of video filenames
    file_pattern = re.compile(
        r'([^\\/:*?"<>|\r\n]+?\.(?:mp4|mov|m4v|avi|mkv|wmv|webm))',
        re.IGNORECASE
    )
    for match in file_pattern.findall(text):
        maybe_add(match)

    # 2) Try JSON / Python literal parsing
    parsed = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            break
        except Exception:
            continue

    if parsed is not None:
        if isinstance(parsed, dict):
            for value in parsed.values():
                for item in _flatten_items(value):
                    maybe_add(str(item))
        else:
            for item in _flatten_items(parsed):
                maybe_add(str(item))

    # 3) Fallback split on common separators
    for part in re.split(r"[;\n,]+", text):
        maybe_add(part)

    return candidates


def build_video_index(video_folder: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}

    for path in video_folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            exact_key = path.name.lower()
            canonical_key = canonical_video_name(path.name)
            index.setdefault(exact_key, path)
            if canonical_key:
                index.setdefault(canonical_key, path)

    return index


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"imported": {}}


def save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def copy_video_if_needed(source_path: Path, dest_path: Path) -> bool:
    if dest_path.exists():
        try:
            if source_path.stat().st_size == dest_path.stat().st_size:
                return False
        except OSError:
            pass

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)
    return True


def import_forms_data(
    onedrive_video_folder: Path,
    forms_xlsx_path: Path,
    local_video_data_folder: Path,
    local_dataset_csv: Path,
    manifest_path: Path,
) -> None:
    if not onedrive_video_folder.exists():
        raise FileNotFoundError(f"Source video folder not found: {onedrive_video_folder}")

    if not forms_xlsx_path.exists():
        raise FileNotFoundError(f"Forms workbook not found: {forms_xlsx_path}")

    local_video_data_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(forms_xlsx_path)
    df.columns = [str(col).strip() for col in df.columns]

    required_targets = {
        "Id": None,
        "Completion time": None,
        "Espresso Extraction Level": None,
        "Was There Channeling, Spraying or Fast Blonding?": None,
        "Upload Espresso Extraction Video Here": None,
    }

    for target in required_targets:
        match = find_column(df, target)
        if match is None:
            raise ValueError(f"Required column not found in workbook: {target}")
        required_targets[target] = match

    id_col = required_targets["Id"]
    completion_col = required_targets["Completion time"]
    level_col = required_targets["Espresso Extraction Level"]
    defect_col = required_targets["Was There Channeling, Spraying or Fast Blonding?"]
    upload_col = required_targets["Upload Espresso Extraction Video Here"]

    video_index = build_video_index(onedrive_video_folder)
    manifest = load_manifest(manifest_path)

    output_rows: list[dict[str, str]] = []
    copied_count = 0
    skipped_missing_video = 0

    for _, row in df.iterrows():
        raw_id = normalize_text(row[id_col])
        if not raw_id:
            continue

        try:
            row_id = str(int(float(raw_id)))
        except ValueError:
            row_id = raw_id

        completion_time = normalize_text(row[completion_col])
        extraction_level = normalize_text(row[level_col])
        defect_flag = normalize_text(row[defect_col])
        upload_value = row[upload_col]

        candidates = parse_upload_cell(upload_value)

        matched_source = None
        for candidate in candidates:
            candidate_key = clean_filename(candidate).lower()
            matched_source = video_index.get(candidate_key)
            if matched_source is None:
                matched_source = video_index.get(canonical_video_name(candidate))
            if matched_source is not None:
                break

        if matched_source is None:
            skipped_missing_video += 1
            print(f"[WARN] No imported video found for Id={row_id}. Upload cell value: {upload_value}")
            continue

        dest_name = f"{row_id}{matched_source.suffix.lower()}"
        dest_path = local_video_data_folder / dest_name

        copied = copy_video_if_needed(matched_source, dest_path)
        if copied:
            copied_count += 1
            print(f"[COPIED] {matched_source.name} -> {dest_name}")
        else:
            print(f"[OK] Already present: {dest_name}")

        manifest["imported"][row_id] = {
            "source_file": matched_source.name,
            "local_file": dest_path.name,
            "completion_time": completion_time,
        }

        output_rows.append({
            "Id": row_id,
            "Video Name": dest_name,
            "Completion time": completion_time,
            "Espresso Extraction Level": extraction_level,
            "Was There Channeling, Spraying or Fast Blonding?": defect_flag,
        })

    def sort_key(item: dict[str, str]):
        value = item["Id"]
        return (0, int(value)) if str(value).isdigit() else (1, str(value))

    output_rows.sort(key=sort_key)

    fieldnames = [
        "Id",
        "Video Name",
        "Completion time",
        "Espresso Extraction Level",
        "Was There Channeling, Spraying or Fast Blonding?",
    ]

    local_dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(local_dataset_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    save_manifest(manifest_path, manifest)

    print("\nImport Complete")
    print(f"Rows written to dataset CSV: {len(output_rows)}")
    print(f"Videos copied/updated:       {copied_count}")
    print(f"Rows skipped (no video):     {skipped_missing_video}")
    print(f"Dataset CSV:                 {local_dataset_csv}")
    print(f"Manifest:                    {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import Microsoft Forms espresso video labels into the local project dataset."
    )
    parser.add_argument(
        "--source-video-dir",
        type=Path,
        default=ONEDRIVE_VIDEO_FOLDER,
        help=f"Folder containing source videos. Can also be set with {SOURCE_VIDEO_FOLDER_ENV}.",
    )
    parser.add_argument(
        "--forms-xlsx",
        type=Path,
        default=FORMS_XLSX_PATH,
        help=f"Path to the Forms workbook. Can also be set with {FORMS_XLSX_ENV}.",
    )
    parser.add_argument(
        "--local-video-data-dir",
        type=Path,
        default=LOCAL_VIDEO_DATA_FOLDER,
        help=f"Destination video folder. Defaults to ./Video Data or {LOCAL_VIDEO_DATA_ENV}.",
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=LOCAL_DATASET_CSV,
        help=f"Output dataset CSV. Defaults to ./dataset.csv or {LOCAL_DATASET_CSV_ENV}.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=MANIFEST_PATH,
        help=f"Import manifest path. Defaults to ./Video Data/_forms_imported.json or {MANIFEST_ENV}.",
    )
    return parser.parse_args()


def require_path(value: Optional[Path], label: str, arg_name: str, env_name: str) -> Path:
    if value is None:
        raise SystemExit(f"{label} is required. Pass {arg_name} or set {env_name}.")
    return value.expanduser().resolve()


if __name__ == "__main__":
    args = parse_args()
    import_forms_data(
        onedrive_video_folder=require_path(
            args.source_video_dir,
            "Source video folder",
            "--source-video-dir",
            SOURCE_VIDEO_FOLDER_ENV,
        ),
        forms_xlsx_path=require_path(
            args.forms_xlsx,
            "Forms workbook",
            "--forms-xlsx",
            FORMS_XLSX_ENV,
        ),
        local_video_data_folder=args.local_video_data_dir.expanduser().resolve(),
        local_dataset_csv=args.dataset_csv.expanduser().resolve(),
        manifest_path=args.manifest_path.expanduser().resolve(),
    )
