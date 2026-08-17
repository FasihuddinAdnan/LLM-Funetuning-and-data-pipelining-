import sys
import json
from tqdm import tqdm

input_path = "dataset_clean.jsonl"
output_path = "dataset_final.jsonl"

dropped_short = 0
dropped_long = 0
dropped_generated = 0
kept = 0

print(f"Applying strict length & generated constraints to {input_path}...\n")

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    
    for line in tqdm(infile, desc="Final Pruning"):
        line = line.strip()
        if not line: continue
            
        record = json.loads(line)
        text = record.get("text", "")
        text_len = len(text)
        
        # Rule 1: Drop Micro-files (< 250 chars)
        if text_len < 250:
            dropped_short += 1
            continue
            
        # Rule 2: Drop the Giant Monsters (> 35,000 chars)
        if text_len > 35000:
            dropped_long += 1
            continue
            
        # Rule 3: Drop explicitly auto-generated files
        header = text[:500].lower()
        if "generated file" in header or "do not edit" in header or "auto-generated" in header:
            dropped_generated += 1
            continue
            
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
        kept += 1

print("\n" + "="*50)
print("             PASS 3: THE BOUNCER")
print("="*50)
print(f"Final Kept Files:   {kept:,}")
print(f"Dropped (Too Short): {dropped_short:,}")
print(f"Dropped (Too Long):  {dropped_long:,}")
print(f"Dropped (Generated): {dropped_generated:,}")
print("="*50)
print(f"Saved to: {output_path}")
