#!/usr/bin/env python3
"""Download CodeSearchNet and POJ-104 datasets."""

import os
import json
import sys


def download_codesearchnet(cache_dir="data_cache"):
    """Download CodeSearchNet dataset (all 6 languages)."""
    from datasets import load_dataset

    languages = ["python", "java", "javascript", "ruby", "go", "php"]
    csn_cache = cache_dir
    total = 0
    for lang in languages:
        print(f"  Loading CodeSearchNet/{lang}...")
        for attempt in range(5):
            try:
                ds = load_dataset("code_search_net", lang,
                                  cache_dir=csn_cache, split="train")
                break
            except Exception as e:
                if attempt == 4:
                    raise
                print(f"    Retry {attempt+1}/5: {e}")
                import time; time.sleep(30)
        n = len(ds)
        total += n
        print(f"    {lang}: {n:,} examples")
    print(f"  Total: {total:,} training examples")


def download_poj104(cache_dir="data_cache"):
    """Download POJ-104 clone detection dataset."""
    poj_dir = os.path.join(cache_dir, "poj104")
    os.makedirs(poj_dir, exist_ok=True)

    required = ["train.jsonl", "valid.jsonl", "test.jsonl"]
    if all(os.path.exists(os.path.join(poj_dir, f)) for f in required):
        print("  POJ-104 already downloaded.")
        return

    print("  Downloading POJ-104 from CodeXGLUE...")
    try:
        from datasets import load_dataset
        ds = load_dataset("google/code_x_glue_cc_clone_detection_poj104",
                          cache_dir=cache_dir, trust_remote_code=True)

        split_map = {"train": "train.jsonl", "validation": "valid.jsonl",
                     "test": "test.jsonl"}
        for split, fname in split_map.items():
            if split not in ds:
                continue
            path = os.path.join(poj_dir, fname)
            with open(path, "w") as f:
                for i, item in enumerate(ds[split]):
                    entry = {
                        "code": item["code"],
                        "label": int(item["label"]),
                        "index": int(item["id"]),
                    }
                    f.write(json.dumps(entry) + "\n")
            print(f"    {fname}: {len(ds[split]):,} examples")
    except Exception as e:
        print(f"  Auto-download failed: {e}")
        print(f"  Please manually place train.jsonl, valid.jsonl, test.jsonl in: {poj_dir}/")
        sys.exit(1)


if __name__ == "__main__":
    cache = sys.argv[1] if len(sys.argv) > 1 else "data_cache"
    print("Downloading CodeSearchNet...")
    download_codesearchnet(cache)
    print("\nDownloading POJ-104...")
    download_poj104(cache)
    print("\nDone.")
