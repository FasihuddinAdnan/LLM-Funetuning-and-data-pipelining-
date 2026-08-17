import json
import sys
from tqdm import tqdm

if len(sys.argv) != 2:
    print("Usage: python eda_jsonl.py dataset.jsonl")
    sys.exit(1)

jsonl_file = sys.argv[1]

total_samples = 0
total_chars = 0
total_lines = 0
unique_files = set()

# Lists to track top 3 shortest and longest files
# Elements format: (char_length, line_count, file_identifier, snippet)
shortest_files = []
longest_files = []

def update_extrema(target_list, entry, is_shortest=True, k=3):
    """Keep track of top-k shortest or longest items without loading all into memory."""
    target_list.append(entry)
    # Sort by character length (entry[0])
    target_list.sort(key=lambda x: x[0], reverse=not is_shortest)
    if len(target_list) > k:
        target_list.pop()

print(f"Streaming through '{jsonl_file}'...\n")

with open(jsonl_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing"):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except Exception:
            continue

        total_samples += 1

        # Extract text content
        text = obj.get("text", "")
        if not text and "messages" in obj:
            text = "\n".join(m.get("content", "") for m in obj["messages"])
        elif not text and "prompt" in obj and "completion" in obj:
            text = obj["prompt"] + obj["completion"]

        # Try to extract file identifier/path, or fallback to sample index
        file_id = (
            obj.get("path") or 
            obj.get("file") or 
            obj.get("filename") or 
            obj.get("repo") or 
            f"Sample_{total_samples}"
        )
        unique_files.add(file_id)

        char_cnt = len(text)
        line_cnt = text.count("\n") + (1 if char_cnt > 0 else 0)

        total_chars += char_cnt
        total_lines += line_cnt

        # Extract a clean 250-character preview snippet
        snippet_raw = text[:250].replace("\n", " ").strip()
        snippet = snippet_raw + ("..." if char_cnt > 250 else "")

        entry = (char_cnt, line_cnt, file_id, snippet)

        update_extrema(shortest_files, entry, is_shortest=True, k=3)
        update_extrema(longest_files, entry, is_shortest=False, k=3)

# ==================== SUMMARY REPORT ====================
print("\n" + "=" * 60)
print("                  JAVA DATASET EDA REPORT                 ")
print("=" * 60)
print(f"Total Records Processed:      {total_samples:,}")
print(f"Total Unique Files/Paths:    {len(unique_files):,}")

if total_samples > 0:
    avg_chars = int(total_chars / total_samples)
    avg_lines = int(total_lines / total_samples)
    print(f"Average Characters per File: {avg_chars:,}")
    print(f"Average Lines per File:      {avg_lines:,}")

print("\n" + "-" * 60)
print("🔍 3 SHORTEST FILES (Check for empty files / license boilerplate)")
print("-" * 60)
for i, (c_len, l_cnt, name, snip) in enumerate(shortest_files, 1):
    print(f"[{i}] {name}")
    print(f"    Metrics: {c_len:,} chars | {l_cnt:,} lines")
    print(f"    Snippet: \"{snip}\"\n")

print("-" * 60)
print("🐘 3 LONGEST FILES (Check for generated code / minified data)")
print("-" * 60)
for i, (c_len, l_cnt, name, snip) in enumerate(longest_files, 1):
    print(f"[{i}] {name}")
    print(f"    Metrics: {c_len:,} chars | {l_cnt:,} lines")
    print(f"    Snippet: \"{snip}\"\n")
