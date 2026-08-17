#!/usr/bin/env python3
"""
pack_sorted.py

Tokenizes and packs dataset_sorted.jsonl into fixed-length blocks for
training. dataset_sorted.jsonl was reverse-engineered by sorting on Java
`package` declarations since the original repo/path metadata was lost --
this script re-extracts the package name on the fly and inserts a marker
whenever it changes, giving the model a (approximate) signal for project
boundaries. See the caveats discussed in chat: package name is a proxy
for project identity, not a guarantee -- some grouping will be wrong.
"""

import json
import re
from transformers import AutoTokenizer
from datasets import Dataset
from tqdm import tqdm

INPUT_FILE = "dataset_sorted.jsonl"
OUTPUT_DIR = "dataset_packed_qwen"
MODEL_ID = "Qwen/Qwen2.5-Coder-3B"
SEQ_LEN = 4096

PACKAGE_RE = re.compile(r'^\s*package\s+([a-zA-Z0-9_.]+)\s*;', re.MULTILINE)

print(f"Loading official tokenizer: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
eos_token_id = tokenizer.eos_token_id


def extract_package(text):
    match = PACKAGE_RE.search(text)
    return match.group(1) if match else "zz_unknown_package"


def packed_generator():
    buffer = []
    current_package = None

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    print(f"\nTokenizing and packing {total_lines:,} files into {SEQ_LEN}-token blocks...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Processing files"):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            text = record.get("text", "")
            package = extract_package(text)

            if package != current_package:
                marker = f"// ==== PACKAGE GROUP: {package} ====\n"
                marker_tokens = tokenizer(marker, add_special_tokens=False)["input_ids"]
                buffer.extend(marker_tokens)
                current_package = package

            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            tokens.append(eos_token_id)
            buffer.extend(tokens)

            while len(buffer) >= SEQ_LEN:
                chunk = buffer[:SEQ_LEN]
                buffer = buffer[SEQ_LEN:]
                yield {
                    "input_ids": chunk,
                    "attention_mask": [1] * SEQ_LEN
                }


print("\nStarting generation (this may take a little while)...")
packed_dataset = Dataset.from_generator(packed_generator)

print("\n" + "=" * 55)
print("             PHASE 2 COMPLETE (package-sorted)")
print("=" * 55)
print(f"Total Packed Sequences: {len(packed_dataset):,}")
print(f"Tokens per Sequence:    {SEQ_LEN:,}")
print(f"Total Training Tokens:  {len(packed_dataset) * SEQ_LEN:,}")
print("=" * 55)
print(f"Saving Arrow dataset to: ./{OUTPUT_DIR}/")
packed_dataset.save_to_disk(OUTPUT_DIR)
print("Done! Dataset is ready for the GPU. 🚀")
