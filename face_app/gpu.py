import json
import os
import shutil
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

MODES = ("auto", "gpu", "cpu")

_FAISS_GPU_OK = None


def load_settings():
    data = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            data = {}
    return data


def save_settings(updates):
    data = load_settings()
    data.update(updates)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return data


def resolve_device():
    env = os.environ.get("FACE_DEVICE", "").strip().lower()
    if env in MODES:
        return env
    dev = load_settings().get("device", "auto")
    return dev if dev in MODES else "auto"


def apply_device():
    """Configure the environment for the requested mode. Must run BEFORE
    tensorflow (and therefore deepface) is imported."""
    dev = resolve_device()
    if dev == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif dev == "gpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    return dev


def faiss_gpu_known_supported():
    """True if the installed faiss build ships with GPU support (Linux
    faiss-gpu wheels only; not available for the Windows faiss-cpu wheel)."""
    global _FAISS_GPU_OK
    if _FAISS_GPU_OK is None:
        try:
            import faiss
            _FAISS_GPU_OK = hasattr(faiss, "StandardGpuResources")
        except Exception:
            _FAISS_GPU_OK = False
    return _FAISS_GPU_OK


def _nvidia_smi():
    exe = shutil.which("nvidia-smi") or os.path.join(
        os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"
    )
    if not os.path.exists(exe if not shutil.which("nvidia-smi") else exe if os.path.isabs(exe) else exe):
        exe = "nvidia-smi"
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append({
                "name": parts[1],
                "memory_total_mb": int(parts[2]),
                "memory_used_mb": int(parts[3]),
                "utilization_pct": parts[4] if len(parts) > 4 else "?",
            })
    return rows


def runtime_status():
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    smi = _nvidia_smi()
    status = {
        "mode": resolve_device(),
        "tensorflow_gpu": len(gpus) > 0,
        "faiss_gpu_supported": faiss_gpu_known_supported(),
        "gpus": smi if smi else [],
        "compute_device": "/device:GPU:0" if gpus else "/device:CPU:0",
    }
    return status