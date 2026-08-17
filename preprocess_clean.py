#!/usr/bin/env python3
"""
preprocess_clean.py

Streams through 'downloaded_repos', repo-by-repo, extracts clean unique Java
source files into 'dataset.jsonl', and IMMEDIATELY deletes each raw repo
folder after processing to keep disk footprint low (target: under 1-2GB
at any point in time).

Guardrails implemented:
  1. Dynamic disk purging      -> shutil.rmtree() after each repo is processed
  2. Cryptographic dedup       -> SHA-256 over cleaned file content
  3. Strict noise filtering    -> skip test files/dirs, target/build/out/.git
  4. VRAM protection           -> discard files <15 or >800 lines
  5. Text cleansing            -> strip leading license/copyright comment blocks
  6. Standardized JSONL output -> {"text": "// FILE: <name>.java\n<code>"}
"""

import os
import re
import sys
import json
import shutil
import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

REPOS_DIR = Path("downloaded_repos")
OUTPUT_FILE = Path("dataset.jsonl")

MIN_LINES = 15
MAX_LINES = 800

SKIP_DIR_NAMES = {
    ".git", "target", "build", "out", "test", "tests",
    "node_modules", ".idea", ".settings", "bin", ".gradle", ".mvn",
    "generated", "gen", "__pycache__",
}

TEST_FILENAME_PATTERN = re.compile(r'(Test|Tests|IT|TestCase)\.java$')

LICENSE_KEYWORDS = re.compile(
    r'(copyright|license|apache license|mit license|'
    r'permission is hereby granted|all rights reserved|licensed under)',
    re.IGNORECASE,
)

LEADING_BLOCK_COMMENT = re.compile(r'\A\s*/\*.*?\*/\s*', re.DOTALL)

# ---------------------------------------------------------------------------
# COUNTERS (global, across all repos)
# ---------------------------------------------------------------------------

seen_hashes = set()
stats = {
    "repos_processed": 0,
    "files_saved": 0,
    "duplicates_dropped": 0,
    "boilerplate_filtered": 0,   # test files, wrong size, unreadable, etc.
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def is_test_file(filepath: Path, repo_root: Path) -> bool:
    """Check if a file lives in a test directory or matches a test filename pattern."""
    try:
        rel_parts = [p.lower() for p in filepath.relative_to(repo_root).parts]
    except ValueError:
        rel_parts = [p.lower() for p in filepath.parts]

    if any(p in {"test", "tests", "androidtest"} for p in rel_parts):
        return True
    if TEST_FILENAME_PATTERN.search(filepath.name):
        return True
    if filepath.name.startswith("Test"):
        return True
    return False


def strip_license_header(text: str) -> str:
    """Remove a leading /* ... */ or leading run of // comment lines if it
    looks like a copyright/license block. Leaves normal doc-comments alone."""

    # Case 1: leading block comment /* ... */
    match = LEADING_BLOCK_COMMENT.match(text)
    if match:
        block = match.group(0)
        if LICENSE_KEYWORDS.search(block):
            return text[match.end():].lstrip("\n")

    # Case 2: leading run of single-line // comments
    lines = text.split("\n")
    idx = 0
    while idx < len(lines) and lines[idx].strip().startswith("//"):
        idx += 1

    if idx > 0:
        leading_block = "\n".join(lines[:idx])
        if LICENSE_KEYWORDS.search(leading_block):
            return "\n".join(lines[idx:]).lstrip("\n")

    return text


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def count_lines(text: str) -> int:
    stripped = text.strip("\n")
    if not stripped:
        return 0
    return len(stripped.split("\n"))


# ---------------------------------------------------------------------------
# CORE PROCESSING
# ---------------------------------------------------------------------------

def process_repo(repo_path: Path, out_handle) -> dict:
    """Walk one repo folder, write clean unique Java files to out_handle,
    and return per-repo stats. Does NOT delete the folder -- caller does that."""

    local_saved = 0
    local_dupes = 0
    local_filtered = 0

    for root, dirs, files in os.walk(repo_path):
        # Prune noisy/irrelevant directories in-place so os.walk skips them
        dirs[:] = [
            d for d in dirs
            if d.lower() not in SKIP_DIR_NAMES and not d.startswith(".")
        ]

        for filename in files:
            if not filename.endswith(".java"):
                continue

            filepath = Path(root) / filename

            if is_test_file(filepath, repo_path):
                print(f"    [SKIP-TEST]     {filepath.relative_to(repo_path)}")
                local_filtered += 1
                continue

            try:
                raw_text = filepath.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeDecodeError) as e:
                print(f"    [SKIP-UNREADABLE] {filepath.name} ({e})")
                local_filtered += 1
                continue

            cleaned_text = strip_license_header(raw_text)
            line_count = count_lines(cleaned_text)

            if line_count < MIN_LINES or line_count > MAX_LINES:
                print(f"    [SKIP-SIZE]     {filename} ({line_count} lines)")
                local_filtered += 1
                continue

            file_hash = sha256_of_text(cleaned_text)
            if file_hash in seen_hashes:
                print(f"    [SKIP-DUPLICATE] {filename}")
                local_dupes += 1
                continue

            seen_hashes.add(file_hash)

            record = {"text": f"// FILE: {filename}\n{cleaned_text.strip()}"}
            out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_handle.flush()

            print(f"    [SAVED]         {filename} ({line_count} lines)")
            local_saved += 1

    return {
        "saved": local_saved,
        "dupes": local_dupes,
        "filtered": local_filtered,
    }


def main():
    if not REPOS_DIR.exists():
        print(f"ERROR: '{REPOS_DIR}' does not exist. Nothing to process.")
        sys.exit(1)

    repo_dirs = sorted([p for p in REPOS_DIR.iterdir() if p.is_dir()])
    total_repos = len(repo_dirs)

    if total_repos == 0:
        print(f"No repo folders found in '{REPOS_DIR}'. Exiting.")
        sys.exit(0)

    print("=" * 70)
    print(f"STARTING PREPROCESSING: {total_repos} repositories found")
    print(f"Output dataset: {OUTPUT_FILE.resolve()}")
    print("=" * 70)

    # Open once, append mode, so we never hold the full dataset in memory
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_handle:
        for i, repo_path in enumerate(repo_dirs, start=1):
            print(f"\n[{i}/{total_repos}] PROCESSING REPO: {repo_path.name}")
            print("-" * 70)

            repo_stats = process_repo(repo_path, out_handle)

            stats["files_saved"] += repo_stats["saved"]
            stats["duplicates_dropped"] += repo_stats["dupes"]
            stats["boilerplate_filtered"] += repo_stats["filtered"]
            stats["repos_processed"] += 1

            print(
                f"  Repo summary -> saved: {repo_stats['saved']}, "
                f"duplicates: {repo_stats['dupes']}, "
                f"filtered: {repo_stats['filtered']}"
            )

            # --- GUARDRAIL 1: Dynamic Disk Purging ---
            try:
                shutil.rmtree(repo_path)
                print(f"  [DELETED] Raw repo folder wiped: {repo_path}")
            except OSError as e:
                print(f"  [WARNING] Could not delete {repo_path}: {e}")

    # -----------------------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------------------
    remaining = list(REPOS_DIR.iterdir()) if REPOS_DIR.exists() else []
    downloaded_repos_clear = len(remaining) == 0

    print("\n" + "=" * 70)
    print("FINAL SUMMARY REPORT")
    print("=" * 70)
    print(f"Repositories processed:      {stats['repos_processed']}")
    print(f"Unique files saved:          {stats['files_saved']}")
    print(f"Duplicates dropped:          {stats['duplicates_dropped']}")
    print(f"Boilerplate/filtered files:  {stats['boilerplate_filtered']}")
    print(f"Dataset written to:          {OUTPUT_FILE.resolve()}")

    if downloaded_repos_clear:
        print(f"'{REPOS_DIR}' folder confirmed COMPLETELY WIPED CLEAR.")
    else:
        print(
            f"WARNING: '{REPOS_DIR}' still contains {len(remaining)} item(s): "
            f"{[p.name for p in remaining]}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
