#!/usr/bin/env python3
"""
clean_pass2.py

Second cleaning pass over an already-built dataset.jsonl. Fixes what the
first pass missed:
  1. Low-content "stub" files that passed the line-count filter but have
     almost no real code (mostly blank lines / lone braces).
  2. License headers that weren't at the very top of the file (e.g. preceded
     by a "GENERATED FILE" comment line).
  3. Near-duplicate files that differ only in whitespace/formatting -- exact
     SHA-256 dedup misses these since the raw bytes differ.

Usage:
    python3 clean_pass2.py dataset.jsonl
"""

import sys
import re
import json
import hashlib

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

MIN_MEANINGFUL_LINES = 8   # lines that contain real code, not just braces/blank
OUTPUT_SUFFIX = "_clean.jsonl"

# A "meaningful" line has more than just braces, semicolons, or whitespace
TRIVIAL_LINE_RE = re.compile(r'^[\s\{\}\(\);]*$')

LICENSE_BLOCK_RE = re.compile(
    r'/\*.*?(?:license|copyright|apache license|permission is hereby granted|mit license).*?\*/',
    re.IGNORECASE | re.DOTALL
)
LICENSE_LINE_RE = re.compile(
    r'(?:[ \t]*//.*(?:license|copyright|apache|permission).*\n)+',
    re.IGNORECASE
)


def strip_license(code):
    """Looks for a license block anywhere near the top (first 4000 chars),
    not just at position 0 -- handles files with a 'GENERATED FILE' comment
    or similar sitting before the actual license block."""
    head = code[:4000]

    block_match = LICENSE_BLOCK_RE.search(head)
    if block_match and block_match.start() < 1500:
        return code[:block_match.start()] + code[block_match.end():]

    line_match = LICENSE_LINE_RE.search(head)
    if line_match and line_match.start() < 200:
        return code[:line_match.start()] + code[line_match.end():]

    return code


def count_meaningful_lines(code):
    lines = code.split("\n")
    return sum(1 for line in lines if not TRIVIAL_LINE_RE.match(line))


def normalized_fingerprint(code):
    """Collapses all whitespace to catch near-duplicates that differ only
    in formatting/indentation, not actual code content."""
    collapsed = re.sub(r'\s+', ' ', code).strip()
    return hashlib.sha256(collapsed.encode("utf-8", errors="ignore")).hexdigest()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 clean_pass2.py <input.jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = input_path.replace(".jsonl", OUTPUT_SUFFIX)

    print(f"Reading from: {input_path}")
    print(f"Writing to:   {output_path}\n")

    stats = {
        "total_records": 0,
        "kept": 0,
        "dropped_low_content": 0,
        "dropped_near_duplicate": 0,
        "license_stripped_extra": 0,
        "bad_json_skipped": 0,
    }

    seen_fingerprints = set()

    # Count total lines first for a proper progress bar (cheap, single pass)
    print("Counting records for progress bar...")
    with open(input_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    print(f"Found {total_lines:,} records.\n")

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        iterator = infile
        if HAVE_TQDM:
            iterator = tqdm(infile, total=total_lines, desc="Cleaning", unit="rec")

        for line in iterator:
            line = line.strip()
            if not line:
                continue

            stats["total_records"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json_skipped"] += 1
                continue

            text = record.get("text", "")

            # Split off the "// FILE: name.java\n" header we added in pass 1
            if text.startswith("// FILE:"):
                header_end = text.find("\n")
                file_header = text[:header_end] if header_end != -1 else text
                code = text[header_end + 1:] if header_end != -1 else ""
            else:
                file_header = ""
                code = text

            # Try stripping any license block pass 1 missed
            before_len = len(code)
            code = strip_license(code)
            if len(code) < before_len:
                stats["license_stripped_extra"] += 1

            # Filter 1: low real-content stub files
            if count_meaningful_lines(code) < MIN_MEANINGFUL_LINES:
                stats["dropped_low_content"] += 1
                continue

            # Filter 2: near-duplicate detection via whitespace-normalized hash
            fingerprint = normalized_fingerprint(code)
            if fingerprint in seen_fingerprints:
                stats["dropped_near_duplicate"] += 1
                continue
            seen_fingerprints.add(fingerprint)

            final_text = f"{file_header}\n{code}" if file_header else code
            outfile.write(json.dumps({"text": final_text}, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    print("\n" + "=" * 60)
    print("PASS 2 CLEANING SUMMARY")
    print("=" * 60)
    print(f"Total input records:              {stats['total_records']:,}")
    print(f"Kept:                              {stats['kept']:,}")
    print(f"Dropped (low-content/stub):        {stats['dropped_low_content']:,}")
    print(f"Dropped (near-duplicate whitespace):{stats['dropped_near_duplicate']:,}")
    print(f"Extra license headers stripped:    {stats['license_stripped_extra']:,}")
    print(f"Malformed JSON lines skipped:      {stats['bad_json_skipped']:,}")
    print("-" * 60)
    print(f"Output file: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
