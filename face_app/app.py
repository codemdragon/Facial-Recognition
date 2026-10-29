"""
Updated app.py - integrates PostgreSQL via db.py (graceful fallback).
All original API routes preserved. DB recording added to /api/build and /api/search.
"""
import argparse
import logging
import os
import string
import threading
import uuid

import gpu
gpu.apply_device()

import numpy as np  # noqa: E402
from flask import Flask, jsonify, render_template, request, send_file  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from vector_db import DETECTORS, MODELS, FaceIndex, is_image, list_images, normalize_roots, resolve  # noqa: E402
import db  # noqa: E402

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

STATE = {
    "index": None,
    "db_root": None,
    "job": None,
}
STATE_LOCK = threading.Lock()


def _persist_last_build(db_root, model, detector, clusters):
    last = {**gpu.load_settings().get("last_build", {}),
            "db_root": db_root, "model": model, "detector": detector, "clusters": clusters}
    gpu.save_settings({"last_build": last})


def _load_cached_index():
    lb = gpu.load_settings().get("last_build", {})
    db_root = lb.get("db_root")
    if not db_root:
        return None
    norm = normalize_roots(db_root)
    if not norm:
        return None
    index = FaceIndex.load(
        db_root,
        model=lb.get("model", "Facenet"),
        detector=lb.get("detector", "opencv"),
        num_clusters=int(lb.get("clusters", 8)),
    )
    if index is None:
        return None
    with STATE_LOCK:
        STATE["index"] = index
        STATE["db_root"] = index.db_root
    return index

THUMB_SIZE = (420, 420)


def thumbnail_bytes(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im.thumbnail(THUMB_SIZE)
        if im.mode in ("RGBA", "P"):
            im = im.convert("RGB")
        out = os.path.join(UPLOAD_DIR, "_thumb_cache")
        os.makedirs(out, exist_ok=True)
        fp = os.path.join(out, uuid.uuid4().hex + ".jpg")
        im.save(fp, "JPEG", quality=82)
        return fp


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/gpu")
def api_gpu():
    try:
        return jsonify(gpu.runtime_status())
    except Exception as exc:  # noqa: BLE001
        logging.exception("gpu status failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/device", methods=["POST"])
def api_device():
    data = request.get_json(force=True)
    device = data.get("device", "").lower()
    if device not in gpu.MODES:
        return jsonify({"error": "device must be one of: " + ", ".join(gpu.MODES)}), 400
    gpu.save_settings({"device": device})
    return jsonify({
        "ok": True,
        "mode": device,
        "message": "Saved. Restart the app for the device mode to fully take effect.",
    })


@app.route("/api/listdir")
def api_listdir():
    path = request.args.get("path", "")
    path = path.strip()
    entries = []
    drives = []
    parent = None

    if not path or os.path.dirname(path) == path:
        if not path:
            drives = [f"{c}:\\" for c in string.ascii_uppercase if os.path.exists(f"{c}:\\")]
            path = "" if not drives else drives[0]
        if os.path.dirname(path) == path:
            drives = [f"{c}:\\" for c in string.ascii_uppercase if os.path.exists(f"{c}:\\")]
            path = ""

    if path and os.path.isdir(path):
        parent = os.path.dirname(path) if os.path.dirname(path) != path else None
        try:
            names = sorted(os.listdir(path), key=lambda n: (not os.path.isdir(os.path.join(path, n)), n.lower()))
        except PermissionError:
            names = []
        dirs, imgs = [], []
        for name in names:
            full = os.path.join(path, name)
            if os.path.isdir(full):
                dirs.append(name)
            elif is_image(full):
                imgs.append(name)
        entries = [{"name": d, "type": "dir", "path": os.path.join(path, d)} for d in dirs]
        entries += [{"name": i, "type": "img", "path": os.path.join(path, i)} for i in imgs[:20]]

    return jsonify({
        "current": path or "",
        "parent": parent,
        "drives": drives,
        "entries": entries,
        "count_images": len(list_images(path)) if path and os.path.isdir(path) else 0,
    })


@app.route("/api/build", methods=["POST"])
def api_build():
    data = request.get_json(force=True)
    raw_db = data.get("db_root", "")
    norm_roots = normalize_roots(raw_db)
    if not norm_roots:
        return jsonify({"error": "No valid folder(s) specified or folder(s) do not exist."}), 400
    db_root = "; ".join(norm_roots)

    model = data.get("model", "Facenet")
    detector = data.get("detector", "opencv")
    num_clusters = max(1, min(int(data.get("clusters", 8)), 512))
    workers = max(1, min(int(data.get("workers", 4)), 16))
    force = bool(data.get("force", False))
    if model not in MODELS:
        return jsonify({"error": "Unknown model."}), 400
    if detector not in DETECTORS:
        return jsonify({"error": "Unknown detector."}), 400

    with STATE_LOCK:
        if STATE["job"] and STATE["job"]["running"]:
            return jsonify({"error": "An indexing job is already running."}), 409
        STATE["job"] = {
            "running": True,
            "processed": 0,
            "total": 1,
            "matched": 0,
            "error": None,
            "done": False,
            "phase": "scanning",
            "current": "",
        }

    def run():
        def progress(i, total, img, ok):
            with STATE_LOCK:
                job = STATE["job"]
                job["processed"] = i
                job["total"] = total
                job["phase"] = "embedding"
                if ok:
                    job["matched"] += 1
                job["current"] = img

        try:
            index = FaceIndex.build(db_root, model=model, detector=detector,
                                    num_clusters=num_clusters, workers=workers,
                                    progress=progress, force=force)
            _persist_last_build(db_root, model, detector, num_clusters)
            with STATE_LOCK:
                STATE["index"] = index
                STATE["db_root"] = db_root
                STATE["job"]["stats"] = index.build_stats
                STATE["job"]["done"] = True
                STATE["job"]["running"] = False
                STATE["job"]["phase"] = "done"

            # Record build in PostgreSQL
            try:
                sec_dict = {
                    info["name"]: info["members"]
                    for _, info in index.sections.items()
                }
                db.record_index_build(
                    index.db_root, index.model, index.detector,
                    index.num_clusters, len(index.relpaths),
                    index.relpaths, sec_dict,
                )
            except Exception:  # noqa: BLE001
                logging.warning("DB record_index_build failed (non-fatal)")

        except Exception as exc:  # noqa: BLE001
            logging.exception("build failed")
            with STATE_LOCK:
                STATE["job"]["error"] = str(exc)
                STATE["job"]["running"] = False
                STATE["job"]["done"] = True
                STATE["job"]["phase"] = "error"

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return jsonify({"ok": True})


@app.route("/api/build_status")
def api_build_status():
    with STATE_LOCK:
        job = dict(STATE["job"]) if STATE["job"] else None
        index = STATE["index"]
    summary = index.summary() if index else None
    if index:
        summary["sections"] = [{"name": s["name"], "count": s["count"]} for s in index.all_sections()]
    return jsonify({"job": job, "summary": summary, "db_root": STATE["db_root"]})


@app.route("/api/sections")
def api_sections():
    with STATE_LOCK:
        index = STATE["index"]
    if index is None:
        return jsonify({"error": "No index built yet."}), 400
    return jsonify({"sections": index.all_sections(), "summary": index.summary()})


@app.route("/api/search", methods=["POST"])
def api_search():
    with STATE_LOCK:
        index = STATE["index"]
    if index is None:
        return jsonify({"error": "No index built yet. Pick a folder and build the index first."}), 400

    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"error": "No image uploaded."}), 400

    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return jsonify({"error": "Unsupported image type."}), 400

    name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, name)
    file.save(save_path)

    top_n = min(int(request.form.get("top_n", 10)), 200)
    top_n = max(1, top_n)
    min_score = float(request.form.get("min_score", 0.0))
    search_strategy = request.form.get("strategy", "fused")  # 'fused' | 'upper_periocular' | 'full'

    try:
        mode = request.form.get("mode", "all")  # 'all' | 'sections'
        if mode == "sections":
            matches = index.section_matches(save_path, top_n=top_n, search_mode=search_strategy)
        else:
            matches = index.search(save_path, top_n=top_n, search_mode=search_strategy)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

    # Apply min score filter
    if min_score > 0:
        matches = [m for m in matches if m["similarity"] >= min_score / 100.0]

    seen = {}
    for m in matches:
        seen[m["section"]] = seen.get(m["section"], 0) + 1

    # Log search in PostgreSQL
    try:
        db.record_search(file.filename, top_n, min_score / 100.0, matches)
    except Exception:  # noqa: BLE001
        logging.warning("DB record_search failed (non-fatal)")

    thumb = f"/api/image?kind=query&rel={name}"
    return jsonify({
        "matches": matches,
        "sections_seen": seen,
        "query_thumb": thumb,
        "top_n": len(matches),
    })


@app.route("/api/image")
def api_image():
    kind = request.args.get("kind", "db")
    rel = request.args.get("rel", "")
    as_full = request.args.get("full", "") == "1"

    if kind == "query":
        target = os.path.join(UPLOAD_DIR, os.path.basename(rel))
        if not os.path.isfile(target):
            return "", 404
        if as_full:
            return send_file(target)
        return send_file(thumbnail_bytes(target), mimetype="image/jpeg")

    with STATE_LOCK:
        index = STATE["index"]
    if index is None:
        return "", 404
    target = resolve(index.db_root, rel)
    if target is None or not os.path.isfile(target):
        return "", 404
    if as_full:
        return send_file(target)
    return send_file(thumbnail_bytes(target), mimetype="image/jpeg")


@app.route("/api/db/status")
def api_db_status():
    """Check PostgreSQL connection status."""
    return jsonify({
        "enabled": db.is_enabled(),
        "dsn_configured": bool(os.environ.get("CPLC_PG_DSN")),
    })


@app.route("/api/history")
def api_history():
    """Return recent search history from PostgreSQL."""
    limit = min(int(request.args.get("limit", 50)), 200)
    return jsonify({"history": db.get_search_history(limit)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPLC Face Search - DeepFace vector search")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to expose on network)")
    parser.add_argument("--port", type=int, default=5000, help="bind port")
    args = parser.parse_args()

    # Init PostgreSQL (non-fatal if unavailable)
    try:
        db.init_db()
    except Exception:  # noqa: BLE001
        logging.warning("PostgreSQL init failed - running without DB")

    try:
        _load_cached_index()
    except Exception:  # noqa: BLE001
        logging.exception("startup cache load failed")

    try:
        gpu.runtime_status()
    except Exception:  # noqa: BLE001
        logging.exception("startup GPU probe failed")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)