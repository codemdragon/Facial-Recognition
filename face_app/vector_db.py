import hashlib
import json
import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import faiss
import numpy as np

from deepface import DeepFace
from deepface.modules.detection import extract_faces as _extract_faces

try:
    from gpu import faiss_gpu_known_supported, resolve_device
except ImportError:  # pragma: no cover
    def faiss_gpu_known_supported():
        return False

    def resolve_device():
        return "auto"

logger = logging.getLogger("vector_db")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

MODELS = ["Facenet", "GhostFaceNet", "VGG-Face", "Facenet512", "ArcFace", "SFace", "OpenFace", "DeepFace"]
DETECTORS = ["opencv", "mtcnn", "retinaface", "dlib"]

CACHE_VERSION = "v5_clean"
DETECT_BATCH = 24            # files sent to each extraction worker task
EMBED_CHUNK = 384            # aligned faces per batched GPU forward
RETRAIN_NEW_MIN = 1024       # re-cluster from scratch when this many new vectors
RETRAIN_NEW_RATIO = 0.10     # ... or when new vectors exceed this share of the DB
CLUSTER_TRAIN_MAX = 250_000  # k-means trains on at most this many vectors

SEARCH_MODES = ["auto", "full", "upper_periocular", "fused"]


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def normalize_roots(db_root):
    if isinstance(db_root, str):
        roots = [p.strip().strip('"') for p in re.split(r'[;\n|]', db_root) if p.strip()]
    elif isinstance(db_root, (list, tuple, set)):
        roots = [str(p).strip().strip('"') for p in db_root if str(p).strip()]
    else:
        roots = [str(db_root).strip().strip('"')]
    valid = []
    for r in roots:
        ar = os.path.abspath(r)
        if os.path.isdir(ar) and ar not in valid:
            valid.append(ar)
    return valid


def list_images(roots):
    norm = normalize_roots(roots) if not isinstance(roots, str) or any(c in roots for c in ";\n|") else [os.path.abspath(roots)]
    found = []
    for r in norm:
        if not os.path.isdir(r):
            continue
        for dirpath, dirnames, filenames in os.walk(r):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("__pycache__", "venv", ".venv", "node_modules")]
            for name in sorted(filenames):
                if is_image(name):
                    found.append(os.path.join(dirpath, name))
    return found


def scan_manifest(roots):
    """Scan one or more root folders for images.
    Returns {relpath: abspath}.
    If multiple roots, relpath is prefixed by folder name (e.g. 'Found 1/1040.png').
    """
    norm = normalize_roots(roots)
    manifest = {}
    is_multi = len(norm) > 1

    used_prefixes = {}
    for r in norm:
        base_name = os.path.basename(r.rstrip("\\/"))
        count = used_prefixes.get(base_name, 0)
        prefix = f"{base_name}_{count}" if count > 0 else base_name
        used_prefixes[base_name] = count + 1

        for dirpath, dirnames, filenames in os.walk(r):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("__pycache__", "venv", ".venv", "node_modules")]
            for name in sorted(filenames):
                if is_image(name):
                    full = os.path.join(dirpath, name)
                    rel = os.path.relpath(full, r).replace("\\", "/")
                    key = f"{prefix}/{rel}" if is_multi else rel
                    manifest[key] = full
    return manifest


def resolve(roots, rel):
    """Resolve a relative image path against one or more database roots."""
    if not roots or not rel:
        return None
    norm = normalize_roots(roots)
    if not norm:
        # Fallback if raw string paths
        raw = [p.strip().strip('"') for p in re.split(r'[;\n|]', str(roots)) if p.strip()]
        norm = [os.path.abspath(r) for r in raw]

    norm_rel = rel.replace("/", os.sep).replace("\\", os.sep).lstrip(os.sep)

    for r in norm:
        base = os.path.abspath(r)
        r_name = os.path.basename(base.rstrip("\\/"))

        # Check if rel starts with r_name + os.sep
        if norm_rel.startswith(r_name + os.sep):
            sub = norm_rel[len(r_name) + len(os.sep):]
            target = os.path.abspath(os.path.join(base, sub))
            try:
                if os.path.commonpath([base, target]) == base and os.path.isfile(target):
                    return target
            except ValueError:
                pass

        # Check directly under base
        target = os.path.abspath(os.path.join(base, norm_rel))
        try:
            if os.path.commonpath([base, target]) == base and os.path.isfile(target):
                return target
        except ValueError:
            pass

    return None


def file_stamp(path: str) -> str:
    st = os.stat(path)
    return f"{st.st_mtime_ns}:{st.st_size}"


def _get_upper_face(face_arr):
    """Crop upper 65% of an aligned face image (eyebrows/eyes/forehead/nose-bridge)
    and resize back to canonical input dimensions (e.g. 112x112)."""
    if face_arr is None or face_arr.ndim != 3:
        return None
    h, w = face_arr.shape[:2]
    upper_h = max(10, int(h * 0.65))
    cropped = face_arr[:upper_h, :]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def _apply_clahe(bgr_img):
    """Apply CLAHE contrast enhancement on the L channel (LAB colour space).

    bgr_img may be float32 [0,1] (from DeepFace) or uint8 [0,255].
    CLAHE requires uint8, so we normalise in/out automatically.
    """
    is_float = bgr_img.dtype != np.uint8
    if is_float:
        # Clip to [0,1] guard against tiny float overshoots, then scale to uint8
        u8 = (np.clip(bgr_img, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        u8 = bgr_img

    lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result_u8 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if is_float:
        return result_u8.astype(np.float32) / 255.0
    return result_u8


def _avg_normalized(vecs):
    """Average a list of L2-normalized vectors and re-normalize the result."""
    stacked = np.vstack(vecs).astype(np.float32)
    mean = stacked.mean(axis=0)
    n = np.linalg.norm(mean)
    return (mean / n) if n > 0 else mean


def _analyze_face_quality(face_obj) -> tuple:
    """Analyze face detection quality and anatomical occlusion.

    Returns:
        warnings: list of human-readable strings
        flags: dict of anatomical state:
            - 'has_eyes': bool
            - 'has_mouth': bool
            - 'lower_occluded': bool (mask / hand over mouth)
            - 'upper_occluded': bool (hat / sunglasses / eyes covered)
    """
    warnings = []
    conf = face_obj.get("confidence", 1.0)
    area = face_obj.get("facial_area", {})
    w = area.get("w", 999)
    h = area.get("h", 999)
    left_eye   = area.get("left_eye")
    right_eye  = area.get("right_eye")
    mouth_left = area.get("mouth_left")
    mouth_right = area.get("mouth_right")

    has_eyes  = (left_eye is not None and right_eye is not None)
    has_mouth = (mouth_left is not None and mouth_right is not None)

    if conf < 0.5:
        warnings.append(f"Low detection confidence ({conf:.0%}) — image may be blurry or face partially visible")
    if w < 60 or h < 60:
        warnings.append(f"Face region is very small ({w}×{h} px) — results may be unreliable")

    if not has_eyes and has_mouth:
        warnings.append("Eyes/forehead occluded or missing — upper periocular search disabled")
    elif has_eyes and not has_mouth:
        warnings.append("Mouth/jaw occluded (mask/hand detected) — activating periocular upper search")
    elif not has_eyes and not has_mouth:
        warnings.append("Landmark alignment failed (extreme angle or severe occlusion)")
    elif has_eyes:
        dx = abs(left_eye[0] - right_eye[0])
        dy = abs(left_eye[1] - right_eye[1])
        eye_dist = (dx**2 + dy**2) ** 0.5
        if eye_dist < 20:
            warnings.append(f"Inter-eye distance too small ({eye_dist:.0f} px) — image resolution may be low")

    flags = {
        "has_eyes": has_eyes,
        "has_mouth": has_mouth,
        "lower_occluded": has_eyes and not has_mouth,
        "upper_occluded": not has_eyes and has_mouth,
    }
    return warnings, flags


def _quality_warnings(face_obj) -> list:
    """Backwards-compatible wrapper returning only the warning strings list."""
    warnings, _ = _analyze_face_quality(face_obj)
    return warnings


def embed(path: str, model: str, detector: str, return_upper: bool = False,
          return_quality: bool = False):
    """Single-image embedding with Multi-Crop Test-Time Augmentation (TTA).

    At search time, generates embeddings for three augmented views of the
    detected face (original, horizontal flip, CLAHE-enhanced) and averages
    them into a single robust 'super-embedding'.  The indexed database is
    never touched; TTA only affects the query vector.

    Args:
        path: Path to the query image.
        model: DeepFace model name (e.g. 'ArcFace').
        detector: Face detector backend (e.g. 'retinaface').
        return_upper: If True, also return an upper-face (periocular) embedding.
        return_quality: If True, include a list of quality warning strings.

    Returns:
        • (full_emb)                          when return_upper=False, return_quality=False
        • (full_emb, upper_emb)               when return_upper=True,  return_quality=False
        • (full_emb, warnings)                when return_upper=False, return_quality=True
        • (full_emb, upper_emb, warnings)     when return_upper=True,  return_quality=True
        Any embedding may be None on failure.
    """
    try:
        faces = _extract_faces(
            img_path=path,
            detector_backend=detector,
            enforce_detection=False,
            align=True,
            grayscale=False,
            expand_percentage=0,
            max_faces=1,
        )
    except Exception as exc:
        logger.warning("Face extraction failed for %s: %s", path, exc)
        if return_upper and return_quality:
            return None, None, []
        if return_upper or return_quality:
            return None, None
        return None

    if not faces or faces[0].get("confidence", 0.0) <= 0.0:
        if return_upper and return_quality:
            return None, None, []
        if return_upper or return_quality:
            return None, None
        return None

    face_obj = faces[0]
    quality_warnings = _quality_warnings(face_obj)

    face_raw = np.asarray(face_obj["face"], dtype=np.float32)
    if face_raw.ndim != 3 or face_raw.shape[-1] != 3:
        if return_upper and return_quality:
            return None, None, quality_warnings
        if return_upper or return_quality:
            return None, None
        return None

    # ── Build TTA augmented crops ────────────────────────────────────────────
    # face_raw is RGB float32 [0,1]; BGR needed for cv2 / DeepFace.represent
    bgr_full  = face_raw[:, :, ::-1]
    bgr_flip  = cv2.flip(bgr_full, 1)
    bgr_clahe = _apply_clahe(bgr_full)
    tta_full  = [bgr_full, bgr_flip, bgr_clahe]  # 3 augmented views

    upper_face = _get_upper_face(face_raw)
    if upper_face is not None:
        bgr_upper  = upper_face[:, :, ::-1]
        bgr_upper_flip  = cv2.flip(bgr_upper, 1)
        bgr_upper_clahe = _apply_clahe(bgr_upper)
        tta_upper = [bgr_upper, bgr_upper_flip, bgr_upper_clahe]
    else:
        tta_upper = tta_full  # fallback

    # ── Single batched forward pass for all augmentations ───────────────────
    combined = tta_full + (tta_upper if return_upper else [])
    try:
        outs = DeepFace.represent(
            img_path=combined,
            model_name=model,
            detector_backend="skip",
            enforce_detection=False,
            align=True,
            normalization="base",
        )
    except Exception as exc:
        logger.warning("DeepFace represent failed for %s: %s", path, exc)
        if return_upper and return_quality:
            return None, None, quality_warnings
        if return_upper or return_quality:
            return None, None
        return None

    if not outs:
        if return_upper and return_quality:
            return None, None, quality_warnings
        if return_upper or return_quality:
            return None, None
        return None

    def _norm_vec(item):
        if isinstance(item, list):
            item = item[0] if item else None
        if not item or "embedding" not in item:
            return None
        v = np.asarray(item["embedding"], dtype=np.float32)
        n = np.linalg.norm(v)
        return (v / n) if n > 0 else None

    # Average TTA embeddings for full face
    full_vecs = [_norm_vec(outs[i]) for i in range(len(tta_full))]
    full_vecs = [v for v in full_vecs if v is not None]
    emb_full = _avg_normalized(full_vecs) if full_vecs else None

    if return_upper:
        offset = len(tta_full)
        upper_vecs = [_norm_vec(outs[offset + i]) for i in range(len(tta_upper))]
        upper_vecs = [v for v in upper_vecs if v is not None]
        emb_upper = _avg_normalized(upper_vecs) if upper_vecs else emb_full

        if return_quality:
            return emb_full, emb_upper, quality_warnings
        return emb_full, emb_upper

    if return_quality:
        return emb_full, quality_warnings
    return emb_full


def _extract_face_worker(chunk_paths, detector):
    """CPU-bound: find+align one face per image. Run inside pool processes."""
    out = []
    for path in chunk_paths:
        try:
            faces = _extract_faces(
                img_path=path,
                detector_backend=detector,
                enforce_detection=False,
                align=True,
                grayscale=False,
                expand_percentage=0,
                max_faces=1,
            )
            out.append(np.asarray(faces[0]["face"], dtype=np.float32) if faces else None)
        except Exception:  # noqa: BLE001
            out.append(None)
    return out


def _slices(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def embed_many(abs_paths, relpaths, model, detector, workers, progress=None, base_done=0, total=None):
    """Embed many images: parallel CPU face extraction, then batched GPU/CPU
    inference. Returns {relpath: l2-normalized embedding}."""
    if total is None:
        total = base_done + len(relpaths)
    faces = {}

    def done_count():
        return base_done + len(faces)

    pool_workers = max(1, min(workers, 8))
    if pool_workers > 1 and len(relpaths) > 1:
        def extract_single(rel):
            try:
                res = _extract_faces(
                    img_path=abs_paths[rel], detector_backend=detector, enforce_detection=False,
                    align=True, grayscale=False, expand_percentage=0, max_faces=1,
                )
                if res and res[0].get("confidence", 0.0) > 0.0 and res[0].get("face") is not None:
                    face = np.asarray(res[0]["face"], dtype=np.float32)
                else:
                    face = None
            except Exception as exc:
                logger.warning("Face extraction failed for %s: %s", rel, exc)
                face = None
            return rel, face

        with ThreadPoolExecutor(max_workers=pool_workers) as pool:
            futs = {pool.submit(extract_single, rel): rel for rel in relpaths}
            for fut in as_completed(futs):
                rel, face = fut.result()
                if face is not None and face.ndim == 3 and face.shape[-1] == 3:
                    faces[rel] = face
                if progress:
                    progress(done_count(), total, rel, face is not None)
        return _batched_embed(faces, relpaths, model, progress, total)

    for i, rel in enumerate(relpaths, start=1):
        try:
            res = _extract_faces(
                img_path=abs_paths[rel], detector_backend=detector, enforce_detection=False,
                align=True, grayscale=False, expand_percentage=0, max_faces=1,
            )
            if res and res[0].get("confidence", 0.0) > 0.0 and res[0].get("face") is not None:
                face = np.asarray(res[0]["face"], dtype=np.float32)
                if face.ndim == 3 and face.shape[-1] == 3:
                    faces[rel] = face
        except Exception as exc:  # noqa: BLE001
            logger.warning("Face extraction failed for %s: %s", rel, exc)
        if progress:
            progress(done_count(), total, rel, rel in faces)
    return _batched_embed(faces, relpaths, model, progress, total)


def _batched_embed(faces, relpaths, model, progress, total):
    ordered = [r for r in relpaths if r in faces]
    if not ordered:
        return {}, {}
    embeddings_full = {}
    embeddings_upper = {}

    for chunk in _slices(ordered, EMBED_CHUNK):
        # Prepare both full face and upper face crops for the chunk
        full_imgs = [faces[r][:, :, ::-1] for r in chunk]
        upper_imgs = [_get_upper_face(faces[r])[:, :, ::-1] for r in chunk]

        # Combine in single forward pass: [full_0, ..., full_n, upper_0, ..., upper_n]
        combined = full_imgs + upper_imgs
        outs = DeepFace.represent(
            img_path=combined, model_name=model, enforce_detection=False,
            detector_backend="skip", align=True, normalization="base",
        )
        if not outs:
            continue

        n_chunk = len(chunk)
        outs_full = outs[:n_chunk]
        outs_upper = outs[n_chunk:]

        for rel, out_f, out_u in zip(chunk, outs_full, outs_upper):
            if isinstance(out_f, dict):
                out_f = [out_f]
            if isinstance(out_u, dict):
                out_u = [out_u]

            if out_f and "embedding" in out_f[0]:
                emb = np.asarray(out_f[0]["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    embeddings_full[rel] = emb / norm

            if out_u and "embedding" in out_u[0]:
                emb = np.asarray(out_u[0]["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    embeddings_upper[rel] = emb / norm

        if progress:
            progress(total, total, chunk[-1], len(embeddings_full) > 0)
    return embeddings_full, embeddings_upper


def _cluster(vectors, k):
    """k-means over L2-normalized vectors; uses the GPU when available."""
    use_gpu = faiss_gpu_known_supported() and resolve_device() != "cpu"
    n, d = vectors.shape
    sample = vectors[:min(n, CLUSTER_TRAIN_MAX)]
    if len(sample) < n:
        rng = np.random.default_rng(seed=42)
        sample = vectors[rng.choice(n, CLUSTER_TRAIN_MAX, replace=False)]
    for gpu in (use_gpu, False):
        try:
            kmeans = faiss.Kmeans(d, k, niter=30, seed=42, verbose=False, gpu=gpu,
                                  min_points_per_centroid=2)
            kmeans.train(sample)
            centroids = kmeans.centroids.astype("float32")
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning("k-means %s attempt failed (%s); retrying without GPU.", gpu, exc)
            centroids = None
    if centroids is None:
        centroids = sample[np.random.default_rng(42).integers(0, len(sample), k)].copy()
    centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    index = faiss.IndexFlatIP(d)
    index.add(centroids)
    _sims, labels = index.search(vectors, 1)
    return centroids, labels.ravel()


def cache_key(db_root, model: str, detector: str, k: int) -> str:
    norm = normalize_roots(db_root)
    roots_str = ";".join(norm) if norm else str(db_root)
    raw = f"{CACHE_VERSION}|{roots_str}|{model}|{detector}|{k}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def cache_path(db_root, model: str, detector: str, k: int) -> str:
    base = os.path.join(tempfile.gettempdir(), "face_recog_cache")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{cache_key(db_root, model, detector, k)}.npz")


def meta_path(db_root, model: str, detector: str, k: int) -> str:
    return cache_path(db_root, model, detector, k).replace(".npz", ".json")


def _cohort_mean(matrix, query_vec, max_samples=256) -> float:
    """Fast cohort background mean dot product against a sample of gallery vectors."""
    if matrix is None or query_vec is None or len(matrix) == 0:
        return 0.0
    n = len(matrix)
    if n <= max_samples:
        sample = matrix
    else:
        step = max(1, n // max_samples)
        sample = matrix[::step][:max_samples]
    dots = sample @ query_vec
    return float(np.mean(dots))


def _calibrate_similarity(sim: float, cohort_mean: float) -> float:
    """Calibrate similarity against background cohort mean.
    If a query vector has high baseline cosine alignment with generic faces
    (hub / phantom face), this suppresses false positives to noise level while
    preserving high genuine matches.
    """
    if cohort_mean <= 0.25:
        return float(sim)
    denom = max(0.05, 1.0 - cohort_mean)
    calibrated = (sim - cohort_mean) / denom
    return float(np.clip(calibrated, 0.0, 1.0))


class FaceIndex:
    def __init__(self, vectors, relpaths, labels, centroids, stamps,
                 model, detector, num_clusters, db_root, build_stats=None,
                 upper_vectors=None):
        self.vectors = vectors
        self.upper_vectors = upper_vectors if upper_vectors is not None else vectors
        self.relpaths = relpaths
        self.labels = labels
        self.centroids = centroids
        self.stamps = stamps
        self.model = model
        self.detector = detector
        self.num_clusters = num_clusters
        norm = normalize_roots(db_root)
        self.db_root = "; ".join(norm) if norm else str(db_root)
        self.db_roots = norm if norm else [str(db_root)]
        self.build_stats = build_stats or {}
        self.sections = self._build_sections()

    def _flat_index(self, matrix):
        d = matrix.shape[1]
        if faiss_gpu_known_supported() and resolve_device() != "cpu":
            try:
                res = faiss.StandardGpuResources()
                cfg = faiss.GpuIndexFlatConfig()
                cfg.device = 0
                cfg.useFloat16 = True
                index = faiss.GpuIndexFlatIP(res, d, cfg)
                index.add(matrix)
                return index
            except Exception as exc:  # noqa: BLE001
                logger.warning("GPU index failed (%s), falling back to CPU.", exc)
        index = faiss.IndexFlatIP(d)
        index.add(matrix)
        return index

    @classmethod
    def build(cls, db_root, model="Facenet", detector="opencv", num_clusters=8,
              workers=4, progress=None, force=False):
        norm = normalize_roots(db_root)
        if not norm:
            raise ValueError("No valid folder(s) specified.")
        db_root_str = "; ".join(norm)
        manifest = scan_manifest(norm)
        if not manifest:
            raise ValueError("No images found under the specified folder(s).")
        total_files = len(manifest)

        stamps_now = {rel: file_stamp(path) for rel, path in manifest.items()}

        existing = None if force else cls.load(db_root_str, model, detector, num_clusters)
        keep, to_embed, removed = [], [], []
        if existing is not None:
            old_index = {rel: i for i, rel in enumerate(existing.relpaths)}
            old_stamps = dict(zip(existing.relpaths, existing.stamps or []))
            for rel in manifest:
                if rel in old_index and old_stamps.get(rel) == stamps_now[rel]:
                    keep.append((rel, old_index[rel]))
                else:
                    to_embed.append(rel)
            removed = [rel for rel in existing.relpaths if rel not in manifest]
        else:
            to_embed = list(manifest)

        stats = {
            "total_files": total_files,
            "kept": len(keep),
            "new": len(to_embed),
            "removed": len(removed),
            "embedded": 0,
            "incremental": existing is not None and not force,
            "reclustered": True,
        }

        new_emb_full = {}
        new_emb_upper = {}
        if to_embed:
            new_emb_full, new_emb_upper = embed_many(
                manifest, to_embed, model, detector, workers,
                progress=progress, base_done=stats["kept"],
                total=stats["kept"] + len(to_embed),
            )
        stats["embedded"] = len(new_emb_full)

        entries = []
        if existing is not None:
            has_old_upper = hasattr(existing, "upper_vectors") and existing.upper_vectors is not None
            for rel, orow in keep:
                old_upper = existing.upper_vectors[orow] if has_old_upper else existing.vectors[orow]
                entries.append((rel, existing.vectors[orow], old_upper, existing.stamps[orow],
                                int(existing.labels[orow])))
        for rel in to_embed:
            if rel in new_emb_full:
                u_vec = new_emb_upper.get(rel, new_emb_full[rel])
                entries.append((rel, new_emb_full[rel], u_vec, stamps_now[rel], None))
        if not entries:
            raise ValueError("No face embeddings could be extracted from the images.")

        entries.sort(key=lambda e: e[0])
        relpaths = [e[0] for e in entries]
        vectors = np.vstack([e[1] for e in entries]).astype("float32")
        upper_vectors = np.vstack([e[2] for e in entries]).astype("float32")
        stamps = [e[3] for e in entries]

        n, d = vectors.shape
        k = max(1, min(num_clusters, n))
        delta = len(to_embed)
        old_flags = np.asarray([e[4] is not None for e in entries], dtype=bool)
        old_labels_arr = np.asarray([e[4] if e[4] is not None else -1 for e in entries], dtype=np.int64)

        if existing is not None and not force and len(to_embed) and n > delta and \
                delta <= RETRAIN_NEW_MIN and delta <= n * RETRAIN_NEW_RATIO:
            centroids_norm = existing.centroids / (np.linalg.norm(existing.centroids, axis=1, keepdims=True) + 1e-9)
            cidx = faiss.IndexFlatIP(d)
            cidx.add(centroids_norm)
            _sims, new_labels = cidx.search(vectors, 1)
            new_assign = new_labels.ravel()
            labels = np.where(old_flags, old_labels_arr, new_assign)
            centroids = existing.centroids
            stats["reclustered"] = False
        else:
            centroids, labels = _cluster(vectors, k)

        index = cls(vectors, relpaths, labels, centroids, stamps,
                    model, detector, k, db_root_str, build_stats=stats,
                    upper_vectors=upper_vectors)
        index.save()
        return index

    @classmethod
    def load(cls, db_root, model="Facenet", detector="opencv", num_clusters=8):
        norm = normalize_roots(db_root)
        key_target = "; ".join(norm) if norm else str(db_root)
        npz = cache_path(key_target, model, detector, num_clusters)
        mp = meta_path(key_target, model, detector, num_clusters)
        if not (os.path.exists(npz) and os.path.exists(mp)):
            return None
        try:
            with np.load(npz, allow_pickle=True) as data:
                relpaths = list(data["relpaths"])
                vectors = data["vectors"]
                upper_vectors = data["upper_vectors"] if "upper_vectors" in data else vectors
                labels = data["labels"]
                centroids = data["centroids"]
                stamps = list(data["stamps"]) if "stamps" in data else None
        except Exception:  # noqa: BLE001
            return None
        if stamps is None:
            return None
        meta = json.load(open(mp, encoding="utf-8"))
        saved_root = meta.get("db_roots") or meta.get("db_root") or key_target
        return cls(vectors, relpaths, labels, centroids, stamps,
                   meta["model"], meta["detector"], meta["num_clusters"], saved_root,
                   upper_vectors=upper_vectors)

    def save(self):
        npz = cache_path(self.db_root, self.model, self.detector, self.num_clusters)
        mp = meta_path(self.db_root, self.model, self.detector, self.num_clusters)
        np.savez_compressed(npz, vectors=self.vectors,
                            upper_vectors=self.upper_vectors,
                            relpaths=np.asarray(self.relpaths, dtype=object),
                            labels=self.labels, centroids=self.centroids,
                            stamps=np.asarray(self.stamps, dtype=object))
        meta = {
            "db_root": self.db_root,
            "db_roots": self.db_roots,
            "model": self.model,
            "detector": self.detector,
            "num_clusters": int(self.num_clusters),
            "count": len(self.relpaths),
            "dim": int(self.vectors.shape[1]),
        }
        with open(mp, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)

    def _build_sections(self):
        sections = {}
        for label in np.unique(self.labels):
            mask = self.labels == label
            idxs = np.where(mask)[0]
            subsim = self.vectors[idxs] @ self.centroids[label]
            order = np.argsort(-subsim)
            curated = zip(idxs[order], subsim[order])
            members = [(self.relpaths[i], float(s)) for i, s in curated]
            radius = float((1.0 - subsim).max()) if len(subsim) else 0.0
            sections[int(label)] = {"radius": radius, "members": members}
        sections = dict(sorted(sections.items(), key=lambda kv: -len(kv[1]["members"])))
        for i, (label, info) in enumerate(sections.items()):
            info["name"] = f"Section {chr(65 + i)}"
            info["label"] = int(label)
        return sections

    def summary(self):
        return {
            "db_root": self.db_root,
            "model": self.model,
            "detector": self.detector,
            "num_clusters": self.num_clusters,
            "count": len(self.relpaths),
            "dim": int(self.vectors.shape[1]),
        }

    def search(self, query_path, top_n=10, search_mode="fused"):
        """Run similarity search with selectable search_mode:
        - 'auto' / 'fused': Maximum of full-face and upper-face similarity.
        - 'full': Standard full-face vector comparison.
        - 'upper_periocular': Compares eye/forehead/brow regions, eliminating mouth/jaw occlusions.

        Returns (results_list, quality_warnings, effective_mode).
        """
        full_vec, upper_vec, quality_warnings = embed(
            query_path, self.model, self.detector,
            return_upper=True, return_quality=True,
        )
        if full_vec is None:
            raise ValueError("Could not produce a face embedding for the query image.")

        # ── Anatomically-aware auto-strategy:
        effective_mode = (search_mode or "fused").lower()
        if effective_mode in ("fused", "auto"):
            is_lower_occluded = any("mouth/jaw occluded" in w.lower() for w in quality_warnings)
            is_upper_occluded = any("eyes/forehead occluded" in w.lower() for w in quality_warnings)

            if is_lower_occluded:
                # Mask / hand over mouth -> Periocular upper face is the hero
                effective_mode = "fused"
            elif is_upper_occluded:
                # Eyes covered / missing -> Periocular upper is invalid, use full face
                effective_mode = "full"
            else:
                # Clean, unoccluded face -> Full face provides the highest precision
                effective_mode = "full"

        n_items = len(self.relpaths)
        fetch_k = min(top_n * 2, n_items)

        # Precompute cohort baseline means for score calibration
        mu_full = _cohort_mean(self.vectors, full_vec)
        mu_upper = _cohort_mean(self.upper_vectors, upper_vec) if upper_vec is not None else 0.0

        scores = {}
        # 1. Full-face similarity
        if effective_mode in ("full", "fused", "auto"):
            idx_full = self._flat_index(self.vectors)
            sims_f, idxs_f = idx_full.search(full_vec.reshape(1, -1), fetch_k)
            for sim, i in zip(sims_f[0], idxs_f[0]):
                if i >= 0:
                    calib_sim = _calibrate_similarity(float(sim), mu_full)
                    scores[i] = max(scores.get(i, -1.0), calib_sim)

        # 2. Upper-face similarity
        if effective_mode in ("upper_periocular", "fused", "auto") and upper_vec is not None:
            idx_upper = self._flat_index(self.upper_vectors)
            sims_u, idxs_u = idx_upper.search(upper_vec.reshape(1, -1), fetch_k)
            for sim, i in zip(sims_u[0], idxs_u[0]):
                if i >= 0:
                    calib_sim = _calibrate_similarity(float(sim), mu_upper)
                    scores[i] = max(scores.get(i, -1.0), calib_sim)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

        results = []
        for idx, sim in ranked:
            label = int(self.labels[idx])
            section = self.sections[label]["name"]
            results.append({
                "rel": self.relpaths[idx],
                "similarity": float(sim),
                "distance": float(1 - sim),
                "section": section,
                "section_id": label,
            })
        return results, quality_warnings, effective_mode

    def section_matches(self, query_path, sections=None, top_n=10, search_mode="fused"):
        full_vec, upper_vec, quality_warnings = embed(
            query_path, self.model, self.detector,
            return_upper=True, return_quality=True,
        )
        if full_vec is None:
            raise ValueError("Could not produce a face embedding for the query image.")

        effective_mode = (search_mode or "fused").lower()
        if effective_mode in ("fused", "auto"):
            is_lower_occluded = any("mouth/jaw occluded" in w.lower() for w in quality_warnings)
            is_upper_occluded = any("eyes/forehead occluded" in w.lower() for w in quality_warnings)

            if is_lower_occluded:
                effective_mode = "fused"
            elif is_upper_occluded:
                effective_mode = "full"
            else:
                effective_mode = "full"

        mode = effective_mode
        wanted = set(sections) if sections else set(self.sections.keys())
        per = []

        mu_full = _cohort_mean(self.vectors, full_vec)
        mu_upper = _cohort_mean(self.upper_vectors, upper_vec) if upper_vec is not None else 0.0

        for label in wanted:
            ids = np.where(self.labels == label)[0]
            if len(ids) == 0:
                continue

            sub_scores = {}
            fetch_k = min(top_n, len(ids))

            if mode in ("full", "fused", "auto"):
                sub_f = np.asarray([self.vectors[i] for i in ids], dtype="float32")
                sub_idx_f = self._flat_index(sub_f)
                sims_f, local_f = sub_idx_f.search(full_vec.reshape(1, -1), fetch_k)
                for sim, li in zip(sims_f[0], local_f[0]):
                    if li >= 0:
                        gid = ids[li]
                        calib_sim = _calibrate_similarity(float(sim), mu_full)
                        sub_scores[gid] = max(sub_scores.get(gid, -1.0), calib_sim)

            if mode in ("upper_periocular", "fused", "auto") and upper_vec is not None:
                sub_u = np.asarray([self.upper_vectors[i] for i in ids], dtype="float32")
                sub_idx_u = self._flat_index(sub_u)
                sims_u, local_u = sub_idx_u.search(upper_vec.reshape(1, -1), fetch_k)
                for sim, li in zip(sims_u[0], local_u[0]):
                    if li >= 0:
                        gid = ids[li]
                        calib_sim = _calibrate_similarity(float(sim), mu_upper)
                        sub_scores[gid] = max(sub_scores.get(gid, -1.0), calib_sim)

            for gid, sim in sub_scores.items():
                per.append({
                    "rel": self.relpaths[gid],
                    "similarity": float(sim),
                    "distance": float(1 - sim),
                    "section": self.sections[int(label)]["name"],
                    "section_id": int(label),
                })

        per.sort(key=lambda r: r["similarity"], reverse=True)
        return per[:top_n], quality_warnings, effective_mode

    def all_sections(self):
        out = []
        for label, info in self.sections.items():
            out.append({
                "id": info["label"],
                "name": info["name"],
                "count": len(info["members"]),
                "preview": [m[0] for m in info["members"][:4]],
                "members": info["members"],
            })
        return out