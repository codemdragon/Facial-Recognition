#!/usr/bin/env bash
# One-time GPU setup for the Face Match app on a Linux server with an NVIDIA GPU
# (RTX 3070/4070 on Dell R730/R740 tested fine). Assumes an NVIDIA driver is already
# installed on the host and Python 3.10+ is available.
set -e
cd "$(dirname "$0")/../.."                     # workspace root (contains deepface/, face_app/, .venv/)

echo "==> Checking NVIDIA driver"
nvidia-smi || { echo "nvidia-smi not found. Install the NVIDIA driver (CUDA 12.x supports 3070/4070)."; exit 1; }

echo "==> Creating virtualenv"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

echo "==> Installing GPU tensorflow (bundles CUDA/cuDNN libraries)"
pip install "tensorflow[and-cuda]==2.21.0"
pip install "tf-keras>=2.16"

echo "==> Installing DeepFace (editable from the cloned repo)"
pip install -e ./deepface

echo "==> Pinning OpenCV and adding GPU FAISS"
pip install "opencv-python==4.10.0.84"
pip install "faiss-gpu-cu12>=1.9"
pip install Flask "Pillow>=10.0" gunicorn

echo "==> Setting device mode to gpu"
python - <<'PY'
import json, os
p = os.path.join("face_app", "settings.json")
d = json.load(open(p)) if os.path.exists(p) else {}
d["device"] = "gpu"
json.dump(d, open(p, "w"), indent=2)
PY

echo
echo "setup complete."
echo
echo "First run will auto-download model weights (~90MB per model) into ~/.deepface/weights."
echo "Start it (exposed on the LAN):   python face_app/app.py --host 0.0.0.0 --port 5000"
echo "Or with gunicorn (4 workers):    cd face_app && gunicorn -w 4 -b 0.0.0.0:5000 app:app"
echo
echo "Confirm the GPU is used: open http://<server-ip>:5000 and check the top-right chip."