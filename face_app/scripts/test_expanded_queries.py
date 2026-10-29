"""
test_expanded_queries.py
Rebuilds/updates index for test_sample with ArcFace + retinaface,
then queries each of the 7 test images against the index (top 15)
and outputs a clear matrix showing which of the other 6 targets
appear in each image's top 15 results.
"""
import sys
import os
import time

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_APP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(FACE_APP_DIR)
sys.path.insert(0, FACE_APP_DIR)

import gpu
gpu.apply_device()

from vector_db import FaceIndex

TEST_SAMPLE_DIR = os.path.join(REPO_ROOT, "test_sample")

TARGETS = [
    ("1000017857.jpeg", "Diff 10/10"),
    ("Screenshot 2026-10-29 132909.png", "Diff 6/10"),
    ("Screenshot 2026-10-29 133434.png", "Diff 7/10"),
    ("Screenshot 2026-10-29 133444.png", "Diff 6.5/10"),
    ("WIN_20250922_18_48_58_Pro.jpg", "Diff 2/10"),
    ("WIN_20261029_12_14_31_Pro.jpg", "Diff 4/10"),
    ("WIN_20261029_12_17_36_Pro.jpg", "Diff 5/10"),
]

TARGET_FILES = {t[0] for t in TARGETS}

print("=" * 75)
print("  REBUILDING / UPDATING INDEX (ArcFace + retinaface) ON test_sample")
print("=" * 75)

t0 = time.time()
def progress(i, total, fname, ok):
    pct = int(i / total * 100) if total else 0
    if i % 5 == 0 or i == total:
        print(f"\r  Progress: {pct}% ({i}/{total}) - {os.path.basename(fname)}", end="", flush=True)

# Note: incremental/new files will be embedded, existing ones reused if cached
index = FaceIndex.build(
    TEST_SAMPLE_DIR,
    model="ArcFace",
    detector="retinaface",
    num_clusters=16,
    workers=4,
    progress=progress,
    force=False
)
print(f"\n  Index ready! Total indexed faces: {len(index.relpaths)} in {time.time()-t0:.1f}s")
print("=" * 75)

matrix_results = {}

for filename, diff in TARGETS:
    filepath = os.path.join(TEST_SAMPLE_DIR, filename)
    print(f"\n>>> QUERY: {filename} ({diff})")
    if not os.path.exists(filepath):
        print(f"  [!] File not found: {filepath}")
        continue
    
    try:
        t_search = time.time()
        results = index.search(filepath, top_n=15)
        dt = time.time() - t_search
    except Exception as e:
        print(f"  [X] Search failed: {e}")
        continue
    
    found_targets = []
    print(f"  Top 15 Results ({dt:.2f}s):")
    for r in results:
        base = os.path.basename(r["rel"])
        sim_pct = r["similarity"] * 100
        is_target = base in TARGET_FILES
        marker = " [TARGET MATCH!]" if (is_target and base != filename) else (" [SELF]" if base == filename else "")
        if is_target and base != filename:
            found_targets.append((base, sim_pct, r["rank"] if "rank" in r else None))
        print(f"    #{results.index(r)+1:02d} | {sim_pct:5.1f}% | {base}{marker}")
    
    matrix_results[filename] = [f[0] for f in found_targets]

print("\n" + "=" * 75)
print("  TARGET CROSS-MATCH SUMMARY MATRIX")
print("=" * 75)
print(f"{'Query Image':<35} | Other Targets Found in Top 15")
print("-" * 75)
for filename, diff in TARGETS:
    found = matrix_results.get(filename, [])
    found_str = ", ".join(found) if found else "None"
    print(f"{filename[:34]:<35} | {found_str} ({len(found)}/6)")
print("=" * 75)

