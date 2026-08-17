#!/usr/bin/env python3
"""
shrink_dataset.py

Shrinks dataset_packed_qwen down to a target on-disk size by keeping only
the FIRST N sequences (a prefix, not a random sample) -- this preserves
your existing 60-step training progress, since those sequences remain
exactly where they were.

Usage:
    python3 shrink_dataset.py
"""

import os
from datasets import load_from_disk

INPUT_DIR = "dataset_packed_qwen"
OUTPUT_DIR = "dataset_packed_qwen_small"
TARGET_SIZE_GB = 0.9  # aim slightly under 1GB to leave margin


def get_folder_size_bytes(folder):
    total = 0
    for dirpath, _, filenames in os.walk(folder):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            total += os.path.getsize(fpath)
    return total


def main():
    print(f"Loading {INPUT_DIR}...")
    dataset = load_from_disk(INPUT_DIR)
    total_sequences = len(dataset)
    print(f"Total sequences: {total_sequences:,}")

    print("Measuring current on-disk size...")
    current_bytes = get_folder_size_bytes(INPUT_DIR)
    current_gb = current_bytes / (1024 ** 3)
    print(f"Current size: {current_gb:.2f} GB\n")

    target_bytes = TARGET_SIZE_GB * (1024 ** 3)
    ratio = target_bytes / current_bytes
    keep_n = max(int(total_sequences * ratio), 8)  # keep at least one batch's worth

    print(f"Target size: {TARGET_SIZE_GB} GB")
    print(f"Keeping first {keep_n:,} of {total_sequences:,} sequences (a prefix, preserving your existing 60-step progress)...\n")

    shrunk = dataset.select(range(keep_n))

    print(f"Saving to {OUTPUT_DIR}...")
    shrunk.save_to_disk(OUTPUT_DIR)

    actual_bytes = get_folder_size_bytes(OUTPUT_DIR)
    actual_gb = actual_bytes / (1024 ** 3)

    # Recompute what this means for a full epoch at your known training pace
    seconds_per_step = 35  # your measured pace from earlier smoke test
    batch_size_effective = 8  # per_device_train_batch_size(2) * gradient_accumulation_steps(4)
    steps_per_epoch = keep_n // batch_size_effective
    total_seconds = steps_per_epoch * seconds_per_step
    total_hours = total_seconds / 3600

    print("\n" + "=" * 60)
    print("SHRINK COMPLETE")
    print("=" * 60)
    print(f"Sequences kept:        {keep_n:,} (of {total_sequences:,})")
    print(f"Actual resulting size: {actual_gb:.2f} GB")
    print(f"Output folder:         {OUTPUT_DIR}")
    print("-" * 60)
    print(f"Estimated full-epoch steps at this size: {steps_per_epoch:,}")
    print(f"Estimated full-epoch time at ~{seconds_per_step}s/step: {total_hours:.1f} hours (~{total_hours/24:.1f} days)")
    print("=" * 60)
    print("\nNext: point train.py at this new folder instead:")
    print(f'  dataset = load_from_disk("{OUTPUT_DIR}")')


if __name__ == "__main__":
    main()
