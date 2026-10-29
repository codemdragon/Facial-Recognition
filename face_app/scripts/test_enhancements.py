"""
test_enhancements.py
Compares standard ArcFace matching against:
1. Illumination Normalization (CLAHE on L-channel of LAB color space)
2. Periocular / Upper-Face Crop (Top 65% of aligned face, removing mouth/hand occlusions)
3. Fused Embedding (Full Face + Upper Face)

Runs against the 7 target images and measures the similarity improvements.
"""
import os
import sys
import time
import cv2
import numpy as np

os.environ['TF_USE_LEGACY_KERAS'] = '1'
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

from deepface import DeepFace
from deepface.modules.detection import extract_faces

TEST_SAMPLE_DIR = os.path.join(REPO_ROOT, "test_sample")

TARGET_KEYS = [
    ("WIN_20250922_18_48_58_Pro.jpg", "Dim + glasses"),
    ("WIN_20261029_12_14_31_Pro.jpg", "Dark + glasses + hand on mouth/nose"),
    ("Screenshot 2026-10-29 132909.png", "Bright photoshoot ID + glasses"),
    ("Screenshot 2026-10-29 133434.png", "Standard + hat + glasses"),
    ("Screenshot 2026-10-29 133444.png", "Standard + hat + glasses + open mouth"),
    ("1000017857.jpeg", "Hat + glasses + earmuffs + bottom/mouth"),
    ("WIN_20261029_12_17_36_Pro.jpg", "Earmuffs + hand on mouth (NO glasses)"),
]

def apply_clahe(img_bgr):
    """Normalize illumination using CLAHE on the Lightness channel."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

print("Extracting faces and testing embeddings across strategies...")

# Cache for extracted face data
extracted = {}
for fname, desc in TARGET_KEYS:
    fpath = os.path.join(TEST_SAMPLE_DIR, fname)
    if not os.path.exists(fpath):
        continue
    faces = extract_faces(img_path=fpath, detector_backend='retinaface', align=True, enforce_detection=False)
    if not faces:
        print(f"Warning: No face in {fname}")
        continue
    # faces[0]['face'] is normalized float [0, 1] RGB
    face_raw = (np.asarray(faces[0]['face']) * 255.0).astype(np.uint8)
    # Convert to BGR for cv2
    face_bgr = cv2.cvtColor(face_raw, cv2.COLOR_RGB2BGR)
    extracted[fname] = face_bgr

def get_arcface_emb(img_rgb):
    # represent accepts BGR/RGB images directly
    res = DeepFace.represent(
        img_path=img_rgb,
        model_name="ArcFace",
        detector_backend="skip",
        enforce_detection=False,
        align=False,
        normalization="base"
    )
    v = np.asarray(res[0]["embedding"], dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

# Precompute vectors for:
# A) Baseline (raw aligned face)
# B) CLAHE (illumination normalized)
# C) Upper-face (top 65% of face, eliminating mouth occlusions/hands)
# D) CLAHE + Upper-face

emb_baseline = {}
emb_clahe = {}
emb_upper = {}
emb_clahe_upper = {}

for fname, bgr in extracted.items():
    rgb_base = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    emb_baseline[fname] = get_arcface_emb(rgb_base)
    
    bgr_clahe = apply_clahe(bgr)
    rgb_clahe = cv2.cvtColor(bgr_clahe, cv2.COLOR_BGR2RGB)
    emb_clahe[fname] = get_arcface_emb(rgb_clahe)
    
    # Upper face: top 65% of rows
    h, w = bgr.shape[:2]
    upper_h = int(h * 0.65)
    upper_bgr = bgr[:upper_h, :]
    # Resize back to canonical size (e.g. 112x112 for ArcFace)
    upper_resized = cv2.resize(upper_bgr, (112, 112))
    rgb_upper = cv2.cvtColor(upper_resized, cv2.COLOR_BGR2RGB)
    emb_upper[fname] = get_arcface_emb(rgb_upper)
    
    upper_clahe = apply_clahe(upper_bgr)
    upper_clahe_resized = cv2.resize(upper_clahe, (112, 112))
    rgb_upper_clahe = cv2.cvtColor(upper_clahe_resized, cv2.COLOR_BGR2RGB)
    emb_clahe_upper[fname] = get_arcface_emb(rgb_upper_clahe)

print("\n" + "=" * 80)
print("  SIMILARITY COMPARISON: BASELINE vs CLAHE vs UPPER-FACE vs CLAHE+UPPER")
print("=" * 80)

# Compare key difficult pairs:
# Pair 1: Photoshoot (Screenshot 132909) vs Dim Light (WIN_18_48_58)
# Pair 2: Hand over mouth (WIN_12_14_31) vs Photoshoot (Screenshot 132909)
# Pair 3: Hand over mouth (WIN_12_14_31) vs Dim Light (WIN_18_48_58)
# Pair 4: Open mouth (Screenshot 133444) vs Photoshoot (Screenshot 132909)
# Pair 5: 10/10 Earmuffs (1000017857) vs Photoshoot (Screenshot 132909)

TEST_PAIRS = [
    ("Screenshot 2026-10-29 132909.png", "WIN_20250922_18_48_58_Pro.jpg", "Photoshoot vs Dim"),
    ("Screenshot 2026-10-29 132909.png", "WIN_20261029_12_14_31_Pro.jpg", "Photoshoot vs Covered Mouth"),
    ("WIN_20250922_18_48_58_Pro.jpg", "WIN_20261029_12_14_31_Pro.jpg", "Dim vs Covered Mouth"),
    ("Screenshot 2026-10-29 132909.png", "Screenshot 2026-10-29 133444.png", "Photoshoot vs Open Mouth+Hat"),
    ("1000017857.jpeg", "WIN_20261029_12_14_31_Pro.jpg", "10/10 Earmuffs vs Covered Mouth"),
    ("1000017857.jpeg", "Screenshot 2026-10-29 132909.png", "10/10 Earmuffs vs Photoshoot"),
    ("WIN_20261029_12_17_36_Pro.jpg", "Screenshot 2026-10-29 132909.png", "No Glasses/Covered vs Photoshoot"),
]

print(f"{'Pair Description':<32} | {'Baseline':>9} | {'CLAHE':>8} | {'Upper-Face':>10} | {'CLAHE+Upper':>11}")
print("-" * 80)

for f1, f2, desc in TEST_PAIRS:
    if f1 in emb_baseline and f2 in emb_baseline:
        s_base = float(np.dot(emb_baseline[f1], emb_baseline[f2])) * 100
        s_clahe = float(np.dot(emb_clahe[f1], emb_clahe[f2])) * 100
        s_upper = float(np.dot(emb_upper[f1], emb_upper[f2])) * 100
        s_cu = float(np.dot(emb_clahe_upper[f1], emb_clahe_upper[f2])) * 100
        print(f"{desc:<32} | {s_base:8.1f}% | {s_clahe:7.1f}% | {s_upper:9.1f}% | {s_cu:10.1f}%")

print("=" * 80)

