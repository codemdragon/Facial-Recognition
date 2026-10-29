"""
Install all runtime deps for CPLC Face Search desktop app.
Run: .venv\Scripts\python.exe install_deps.py
"""
import subprocess
import sys

pkgs = [
    "faiss-cpu>=1.9",
    "PyQt5>=5.15",
    "psycopg2-binary>=2.9",
    "Pillow>=10.0",
    "Flask>=3.0",
    "opencv-python==4.10.0.84",
    "numpy",
]

for pkg in pkgs:
    print(f"\n=== {pkg} ===", flush=True)
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg],
        text=True,
    )
    status = "OK" if r.returncode == 0 else "FAILED"
    print(f"  → {status}", flush=True)

print("\n--- Import check ---", flush=True)
for mod in ["faiss", "PyQt5", "psycopg2", "PIL", "flask", "cv2", "numpy"]:
    try:
        __import__(mod)
        print(f"  {mod}: OK", flush=True)
    except ImportError as e:
        print(f"  {mod}: MISSING — {e}", flush=True)
