#!/usr/bin/env python3
"""
check_languages.py

Scans a dataset.jsonl file and reports the distribution of file extensions
found in the "// FILE: ..." headers, plus a few content-based heuristic
checks, to confirm what's actually in the dataset (should be ~100% .java
given the pipeline only ever walked files ending in .java -- this verifies
that assumption rather than just trusting it).

Usage:
    python3 check_languages.py dataset.jsonl
"""

import sys
import json
import re
from collections import Counter

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

FILE_HEADER_RE = re.compile(r'^// FILE:\s*([^\n]+)')

# A few cheap heuristic signatures for other languages, in case something
# non-Java slipped in despite the .java extension (e.g. mislabeled files,
# XML/Gradle snippets embedded inside a .java-named file, etc.)
LANGUAGE_HINTS = {
    "Kotlin-like": re.compile(r'\bfun\s+\w+\s*\(|\bval\s+\w+\s*[:=]|\bcompanion object\b'),
    "XML/HTML": re.compile(r'^\s*<\?xml|^\s*<html|^\s*<project|^\s*<manifest'),
    "Python-like": re.compile(r'^\s*def\s+\w+\s*\(.*\):|^\s*import\s+\w+\s*$'),
    "Groovy-build-script": re.compile(r'\bapply plugin:|\bdependencies\s*\{'),
    "Shell-script": re.compile(r'^#!/bin/(ba)?sh'),
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 check_languages.py <dataset.jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]

    print(f"Scanning {input_path} for file extensions and language signals...\n")

    with open(input_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    ext_counter = Counter()
    no_header_count = 0
    language_hint_counter = Counter()
    bad_json_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        iterator = tqdm(f, total=total_lines, desc="Scanning") if HAVE_TQDM else f

        for line in iterator:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_json_count += 1
                continue

            text = record.get("text", "")

            header_match = FILE_HEADER_RE.match(text)
            if header_match:
                filename = header_match.group(1).strip()
                if "." in filename:
                    ext = filename.rsplit(".", 1)[-1].lower()
                else:
                    ext = "(no extension)"
                ext_counter[ext] += 1
            else:
                no_header_count += 1

            # Cheap heuristic scan on just the first 500 chars for speed
            head = text[:500]
            for lang_name, pattern in LANGUAGE_HINTS.items():
                if pattern.search(head):
                    language_hint_counter[lang_name] += 1

    print("\n" + "=" * 60)
    print("FILE EXTENSION DISTRIBUTION (from '// FILE:' headers)")
    print("=" * 60)
    for ext, count in ext_counter.most_common(20):
        pct = 100 * count / total_lines
        print(f"  .{ext:<15} {count:>10,}  ({pct:.2f}%)")

    if no_header_count:
        pct = 100 * no_header_count / total_lines
        print(f"  (no // FILE: header) {no_header_count:>10,}  ({pct:.2f}%)")

    print("\n" + "=" * 60)
    print("CONTENT-BASED LANGUAGE HINT SCAN (non-Java signatures found)")
    print("=" * 60)
    if language_hint_counter:
        for lang_name, count in language_hint_counter.most_common():
            print(f"  {lang_name:<22} {count:>10,} record(s) matched")
        print("\n  Note: these are heuristic pattern matches, not certainties --")
        print("  a Java file can legitimately contain a string that happens to")
        print("  match one of these patterns. Worth spot-checking a few, not")
        print("  proof of actual contamination on its own.")
    else:
        print("  None detected -- no non-Java content signatures found.")

    if bad_json_count:
        print(f"\nMalformed JSON lines skipped: {bad_json_count:,}")

    print("=" * 60)


if __name__ == "__main__":
    main()
