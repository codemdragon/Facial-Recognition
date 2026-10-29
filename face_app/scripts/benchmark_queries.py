"""
benchmark_queries.py
--------------------
Runs 3 query images against the Found 1 + Found 2 index
across multiple model/detector combinations and reports
top-15 results for each.

Usage:
    cd face_app
    python scripts/benchmark_queries.py

The script tries up to 3 configs (fastest first) and shows you
exactly what it finds for each query so you can visually confirm.

GREEN FLAG = all 3 queries surface the correct person in top-15.
"""
import sys
import os
import time
import textwrap

# Force UTF-8 output on Windows to handle Unicode characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── Resolve paths so we can import face_app modules ──────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
FACE_APP_DIR = os.path.dirname(SCRIPT_DIR)          # face_app/
REPO_ROOT    = os.path.dirname(FACE_APP_DIR)         # facial Recog/

sys.path.insert(0, FACE_APP_DIR)

# Apply device BEFORE importing TF / deepface
import gpu
gpu.apply_device()

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

DB_ROOTS = [
    os.path.join(REPO_ROOT, "test_sample"),
]

QUERY_IMAGES = [
    os.path.join(REPO_ROOT, "test_sample", "WIN_20250922_18_48_58_Pro.jpg"),
    os.path.join(REPO_ROOT, "test_sample", "WIN_20261029_12_14_31_Pro.jpg"),
    os.path.join(REPO_ROOT, "test_sample", "WIN_20261029_12_17_36_Pro.jpg"),
]

TOP_N = 15

# Configs to try, ordered best-for-occlusion → fastest
CONFIGS = [
    # (label,             model,          detector,      clusters, workers)
    ("ArcFace+retinaface",     "ArcFace",      "retinaface",  16,       4),
    ("GhostFaceNet+retinaface","GhostFaceNet", "retinaface",  16,       4),
    ("ArcFace+mtcnn",          "ArcFace",      "mtcnn",       16,       4),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def sep(char="─", n=70):
    print(char * n)

def header(txt):
    sep("═")
    print(f"  {txt}")
    sep("═")

def fmt_score(sim):
    pct = sim * 100
    bar_len = int(pct / 5)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    color = "🟢" if pct >= 60 else ("🟡" if pct >= 35 else "🔴")
    return f"{color} {pct:5.1f}%  [{bar}]"

def build_or_load(db_roots, model, detector, clusters, workers, force=False):
    from vector_db import FaceIndex, normalize_roots
    db_root_str = "; ".join(db_roots)
    norm = normalize_roots(db_root_str)
    if not norm:
        raise FileNotFoundError(f"DB roots not found: {db_roots}")

    print(f"\n  Checking cache for {model}+{detector} k={clusters} …", end="", flush=True)
    existing = None if force else FaceIndex.load(db_root_str, model, detector, clusters)
    if existing is not None:
        print(f" cached ✓  ({len(existing.relpaths)} faces)", flush=True)
        return existing

    print(f" not cached — building …", flush=True)
    t0 = time.time()
    last = [0]

    def progress(i, total, fname, ok):
        pct = int(i / total * 100) if total else 0
        if pct != last[0]:
            last[0] = pct
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r    [{bar}] {pct:3d}%  {i}/{total}", end="", flush=True)

    idx = FaceIndex.build(
        db_root_str,
        model=model,
        detector=detector,
        num_clusters=clusters,
        workers=workers,
        progress=progress,
        force=force,
    )
    print(f"\r    ✓ Built in {time.time()-t0:.1f}s — {len(idx.relpaths)} faces embedded", flush=True)
    return idx


def run_queries(index, label):
    header(f"Config: {label}")
    all_pass = []

    for qi, qpath in enumerate(QUERY_IMAGES, 1):
        qname = os.path.basename(qpath)
        sep("─")
        print(f"\n  Query {qi}/{len(QUERY_IMAGES)}: {qname}")

        if not os.path.isfile(qpath):
            print(f"  ⚠  FILE NOT FOUND: {qpath}")
            all_pass.append(False)
            continue

        t0 = time.time()
        try:
            results = index.search(qpath, top_n=TOP_N)
        except Exception as exc:
            print(f"  ✗ Search error: {exc}")
            all_pass.append(False)
            continue

        elapsed = time.time() - t0
        print(f"  Search completed in {elapsed:.2f}s\n")

        if not results:
            print("  ✗ No results returned.")
            all_pass.append(False)
            continue

        print(f"  {'Rank':<5} {'Score':>7}  {'Bar (20pt)':>26}  {'Filename':<30}  Section")
        sep("─")
        for rank, r in enumerate(results, 1):
            fname = os.path.basename(r["rel"])
            score_str = fmt_score(r["similarity"])
            print(f"  #{rank:<4} {score_str}  {fname:<30}  {r['section']}")

        # Ask user to visually confirm (we don't look at images)
        top_score = results[0]["similarity"] * 100 if results else 0
        print(f"\n  Top score: {top_score:.1f}%  |  {len(results)} results shown")
        print("  ↑ Please verify: does YOUR face appear in the above list?")

        # Auto-heuristic: if top result score > 45% likely a hit (adjust threshold later)
        passed = top_score >= 45.0
        all_pass.append(passed)

    sep()
    passed_count = sum(all_pass)
    print(f"\n  Config [{label}] auto-heuristic: {passed_count}/{len(QUERY_IMAGES)} queries with top-score ≥ 45%")
    if passed_count == len(QUERY_IMAGES):
        print("  🟢 AUTO-HEURISTIC PASS (not a substitute for visual check)")
    else:
        print("  🔴 Some queries scored low — may need visual check or different config")
    sep("═")
    return all_pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(textwrap.dedent("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  CPLC Face Search — Query Benchmark                             ║
    ║  3 queries × Found 1 + Found 2 database                        ║
    ║  GREEN FLAG = all 3 queries show correct person in top 15      ║
    ╚══════════════════════════════════════════════════════════════════╝
    """))

    # Validate query files exist
    missing = [q for q in QUERY_IMAGES if not os.path.isfile(q)]
    if missing:
        print("ERROR: Missing query images:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    # Validate DB roots
    missing_roots = [r for r in DB_ROOTS if not os.path.isdir(r)]
    if missing_roots:
        print("ERROR: Missing database roots:")
        for m in missing_roots:
            print(f"  {m}")
        sys.exit(1)

    print(f"  Database: {len(DB_ROOTS)} folders")
    for r in DB_ROOTS:
        count = sum(1 for _ in os.scandir(r) if _.is_file())
        print(f"    {r}  ({count} files)")
    print(f"  Query images: {len(QUERY_IMAGES)}")
    print(f"  Top-N: {TOP_N}")
    print(f"  Configs to try: {len(CONFIGS)}")

    results_summary = {}

    for (label, model, detector, clusters, workers) in CONFIGS:
        sep("═")
        print(f"\n🔧  Building/loading index: {label}")
        try:
            index = build_or_load(DB_ROOTS, model, detector, clusters, workers)
        except Exception as exc:
            print(f"  ✗ Index build failed: {exc}")
            continue

        all_pass = run_queries(index, label)
        results_summary[label] = all_pass

    # ── Final summary ─────────────────────────────────────────────────────────
    sep("═")
    print("\n  ╔══════════════════════════════════╗")
    print("  ║       FINAL SUMMARY              ║")
    print("  ╚══════════════════════════════════╝\n")
    for cfg_label, passes in results_summary.items():
        status = "🟢 ALL PASS" if all(passes) else f"🔴 {sum(passes)}/{len(passes)} pass"
        print(f"  {status:<20}  {cfg_label}")
    print()
    print("  Visual verification required: review results above and")
    print("  confirm YOUR face appears in the top-15 for all 3 queries.")
    print()
    print("  Config recommendation:")
    best = next((l for l, p in results_summary.items() if all(p)), None)
    if best:
        print(f"    ✓ Use: {best}")
    else:
        print("    No config achieved auto-heuristic pass — may still have correct")
        print("    visual matches. Check results above for low-score matches (<45%).")
    sep("═")


if __name__ == "__main__":
    main()
