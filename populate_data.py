"""
Data statistics utility for the QA fine-tuning project.

This script validates and reports on the training, validation, and test datasets.
The data files are now maintained manually to ensure diversity and quality.

Usage:
    python populate_data.py
"""
import json
import os


def load_jsonl(filepath):
    """Load a JSONL file and return list of records."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_unique_instructions(records):
    """Get unique instructions from records."""
    return set(r["instruction"] for r in records)


def main():
    files = {
        "train": "data/train.jsonl",
        "validation": "data/val.jsonl",
        "test": "data/test.jsonl",
    }
    
    print("=" * 60)
    print("QA Fine-Tuning Dataset Statistics")
    print("=" * 60)
    
    all_instructions = set()
    
    for name, filepath in files.items():
        records = load_jsonl(filepath)
        unique = get_unique_instructions(records)
        
        print(f"\n{name.upper()} ({filepath}):")
        print(f"  Total examples: {len(records)}")
        print(f"  Unique instructions: {len(unique)}")
        
        if len(records) != len(unique):
            print(f"  WARNING: {len(records) - len(unique)} duplicate instructions found!")
        
        # Check for overlap with other sets
        overlap = all_instructions & unique
        if overlap:
            print(f"  WARNING: {len(overlap)} instructions overlap with other sets!")
            for instr in list(overlap)[:2]:
                print(f"    - {instr[:60]}...")
        
        all_instructions.update(unique)
    
    print("\n" + "=" * 60)
    print(f"Total unique instructions across all sets: {len(all_instructions)}")
    print("=" * 60)
    
    # Sample output formats
    print("\nSample training example format:")
    train_records = load_jsonl(files["train"])
    if train_records:
        sample = train_records[0]
        print(f"  Instruction: {sample['instruction'][:60]}...")
        output_preview = sample['output'].replace('\n', ' ')[:80]
        print(f"  Output: {output_preview}...")


if __name__ == "__main__":
    main()
