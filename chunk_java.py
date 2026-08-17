#!/usr/bin/env python3
"""
chunk_java.py

Reads Java source from a dataset jsonl file and splits each file into
class-level and method-level chunks, tagging each chunk with repo/path/
class/method metadata for later Graph-RAG-style retrieval.
"""

import sys
import json
import re

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

OUTPUT_FILE = "java_chunks.jsonl"
LIMIT = 5000

CLASS_RE = re.compile(r'\b(?:public\s+|final\s+|abstract\s+)*(class|interface|enum)\s+(\w+)[^{]*\{')
METHOD_RE = re.compile(
    r'(?:public|private|protected|static|final|abstract|synchronized|\s)*'
    r'[\w<>\[\],\s?]+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s.]+)?\s*\{'
)
FILE_HEADER_RE = re.compile(r'^// FILE:\s*([^\n]+)')
PACKAGE_RE = re.compile(r'^\s*package\s+([a-zA-Z0-9_.]+)\s*;', re.MULTILINE)


def extract_path_and_repo(text):
    header_match = FILE_HEADER_RE.match(text)
    path = header_match.group(1).strip() if header_match else "unknown"
    pkg_match = PACKAGE_RE.search(text)
    repo = pkg_match.group(1) if pkg_match else "unknown_package"
    return path, repo


def extract_balanced(text, brace_start_idx):
    depth = 0
    for i in range(brace_start_idx, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[brace_start_idx:i + 1], i + 1
    return None, len(text)


def chunk_class_body(class_body, repo, path, class_name):
    chunks = []
    matches = list(METHOD_RE.finditer(class_body))

    if not matches:
        chunks.append({
            "repo": repo, "path": path, "class_name": class_name,
            "method_name": None, "chunk_type": "class", "text": class_body,
        })
        return chunks

    for m in matches:
        method_name = m.group(1)
        brace_idx = class_body.find('{', m.start())
        if brace_idx == -1:
            continue
        method_body, _ = extract_balanced(class_body, brace_idx)
        if method_body is None:
            continue
        full_method_text = class_body[m.start():brace_idx] + method_body
        chunks.append({
            "repo": repo, "path": path, "class_name": class_name,
            "method_name": method_name, "chunk_type": "method",
            "text": full_method_text.strip(),
        })
    return chunks


def chunk_file(text, repo, path):
    chunks = []
    matches = list(CLASS_RE.finditer(text))

    if not matches:
        chunks.append({
            "repo": repo, "path": path, "class_name": None,
            "method_name": None, "chunk_type": "file", "text": text,
        })
        return chunks

    for m in matches:
        class_name = m.group(2)
        brace_idx = text.rfind('{', m.start(), m.end())
        class_body, _ = extract_balanced(text, brace_idx)
        if class_body is None:
            continue
        chunks.extend(chunk_class_body(class_body, repo, path, class_name))
    return chunks


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 chunk_java.py <input.jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]
    print(f"Chunking Java files from {input_path} (first {LIMIT:,} records)...")

    total_files = 0
    total_chunks = 0
    class_chunks = 0
    method_chunks = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        iterator = infile
        if HAVE_TQDM:
            iterator = tqdm(infile, total=LIMIT, desc="Chunking")

        for i, line in enumerate(iterator):
            if i >= LIMIT:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            repo_from_record = record.get("repo")
            path_from_record = record.get("path")
            text = record.get("text", "")

            if not repo_from_record or not path_from_record:
                extracted_path, extracted_repo = extract_path_and_repo(text)
                repo = repo_from_record or extracted_repo
                path = path_from_record or extracted_path
            else:
                repo, path = repo_from_record, path_from_record

            total_files += 1
            file_chunks = chunk_file(text, repo, path)

            for chunk in file_chunks:
                outfile.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1
                if chunk["chunk_type"] == "method":
                    method_chunks += 1
                elif chunk["chunk_type"] == "class":
                    class_chunks += 1

    print("\n" + "=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)
    print(f"Files processed:      {total_files:,}")
    print(f"Total chunks written: {total_chunks:,}")
    print(f"  Method-level:       {method_chunks:,}")
    print(f"  Class-level (no methods found): {class_chunks:,}")
    print("-" * 60)
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
