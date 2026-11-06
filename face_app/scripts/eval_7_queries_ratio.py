"""
eval_7_queries_ratio.py
Evaluates all 7 target queries against the clean test_sample index (104 images).
Analyzes score distributions, target hits, and extrapolates false positive / result rates
for a 5,000 person database.
"""
import os
import sys
import time

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

FACE_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(FACE_APP_DIR)
sys.path.insert(0, FACE_APP_DIR)

from vector_db import FaceIndex

TEST_SAMPLE_DIR = os.path.join(REPO_ROOT, "test_sample")

QUERIES = [
    "WIN_20261029_12_17_36_Pro.jpg",
    "1000017857.jpeg",
    "Screenshot 2026-10-29 132909.png",
    "Screenshot 2026-10-29 133434.png",
    "Screenshot 2026-10-29 133444.png",
    "WIN_20250922_18_48_58_Pro.jpg",
    "WIN_20261029_12_14_31_Pro.jpg",
]

TARGET_SET = set(QUERIES)

idx = FaceIndex.load(TEST_SAMPLE_DIR, "ArcFace", "retinaface", 16)
if idx is None:
    print("Building index...")
    idx = FaceIndex.build(TEST_SAMPLE_DIR, "ArcFace", "retinaface", 16, workers=4)

total_db = len(idx.relpaths)
print(f"Index loaded: {total_db} images.")
print("=" * 80)

SCALE_FACTOR = 5000.0 / total_db

def evaluate(strategy="fused"):
    print(f"\n>>> EVALUATION MODE: STRATEGY = {strategy.upper()} <<<")
    summary = []
    
    for q_name in QUERIES:
        q_path = os.path.join(TEST_SAMPLE_DIR, q_name)
        # Search all items in DB — returns (matches, quality_warnings, effective_mode)
        matches, quality_warnings, effective_mode = idx.search(q_path, top_n=total_db, search_mode=strategy)
        if quality_warnings:
            print(f"  [Quality] {'; '.join(quality_warnings)}")
        if effective_mode != strategy:
            print(f"  [Auto-strategy] Requested '{strategy}' → auto-switched to '{effective_mode}'")
        
        # Exclude self
        other_matches = [m for m in matches if m["rel"] != q_name]
        
        # Targets found vs non-targets
        targets_found = [m for m in other_matches if m["rel"] in TARGET_SET]
        non_targets = [m for m in other_matches if m["rel"] not in TARGET_SET]
        
        # Counts at thresholds
        c_80 = sum(1 for m in other_matches if m["similarity"] >= 0.80)
        c_70 = sum(1 for m in other_matches if m["similarity"] >= 0.70)
        c_60 = sum(1 for m in other_matches if m["similarity"] >= 0.60)
        c_50 = sum(1 for m in other_matches if m["similarity"] >= 0.50)
        c_40 = sum(1 for m in other_matches if m["similarity"] >= 0.40)
        c_30 = sum(1 for m in other_matches if m["similarity"] >= 0.30)
        
        # Non-target counts (false positives)
        fp_80 = sum(1 for m in non_targets if m["similarity"] >= 0.80)
        fp_70 = sum(1 for m in non_targets if m["similarity"] >= 0.70)
        fp_60 = sum(1 for m in non_targets if m["similarity"] >= 0.60)
        fp_50 = sum(1 for m in non_targets if m["similarity"] >= 0.50)
        fp_40 = sum(1 for m in non_targets if m["similarity"] >= 0.40)
        fp_30 = sum(1 for m in non_targets if m["similarity"] >= 0.30)
        
        target_hits_str = ", ".join([f"{m['rel'][:18]} ({m['similarity']*100:.1f}%)" for m in targets_found[:4]])
        
        summary.append({
            "query": q_name,
            "target_hits": len(targets_found),
            "target_detail": target_hits_str,
            "c_70": c_70, "c_60": c_60, "c_50": c_50, "c_40": c_40, "c_30": c_30,
            "fp_70": fp_70, "fp_60": fp_60, "fp_50": fp_50, "fp_40": fp_40, "fp_30": fp_30,
            "top_score": other_matches[0]["similarity"] if other_matches else 0.0,
            "top_match": other_matches[0]["rel"] if other_matches else "",
        })
        
        print(f"\nQuery: {q_name}")
        print(f"  Target matches found: {len(targets_found)}/6")
        for t in targets_found:
            rank = next(i for i, m in enumerate(other_matches, 1) if m["rel"] == t["rel"])
            print(f"    -> Rank #{rank:02d}: {t['rel']} | Score: {t['similarity']*100:.1f}%")
        print(f"  Counts in 104 DB:   >=70%: {c_70} | >=60%: {c_60} | >=50%: {c_50} | >=40%: {c_40} | >=30%: {c_30}")
        print(f"  Projected in 5000:  >=70%: {int(c_70*SCALE_FACTOR)} | >=60%: {int(c_60*SCALE_FACTOR)} | >=50%: {int(c_50*SCALE_FACTOR)} | >=40%: {int(c_40*SCALE_FACTOR)} | >=30%: {int(c_30*SCALE_FACTOR)}")
        print(f"  Est. Non-targets 5k:>=70%: {int(fp_70*SCALE_FACTOR)} | >=60%: {int(fp_60*SCALE_FACTOR)} | >=50%: {int(fp_50*SCALE_FACTOR)} | >=40%: {int(fp_40*SCALE_FACTOR)}")

    return summary

print("\n--- Running evaluation on FUSED strategy ---")
s_fused = evaluate("fused")

print("\n--- Running evaluation on FULL FACE strategy ---")
s_full = evaluate("full")

