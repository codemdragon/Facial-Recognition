@echo off
rem One-time GPU setup for the Face Match app on Windows Server with an NVIDIA GPU.
rem NOTE: official GPU tensorflow wheels on Windows require manually placing the CUDA 12 and cuDNN 9
rem DLLs on PATH (Python isn't the only source) - easiest is the Microsoft / NVIDIA "TensorFlow on Windows"
rem setup. If that is painful, prefer the Linux script on the server (recommended).
title Face Match GPU setup (Windows)
cd /d "%~dp0..\.."

echo ==^> Creating virtualenv
py -3 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo ==^> Installing tensorflow (CPU-only works too, GPU is auto-detected if a Python CUDA env is present)
pip install "tensorflow-cpu==2.21.0"
pip install "tf-keras>=2.16"

echo ==^> Installing DeepFace (editable from the cloned repo)
pip install -e .\deepface

echo ==^> Pinning OpenCV and adding FAISS
pip install "opencv-python==4.10.0.84"
pip install "faiss-cpu>=1.9"
pip install Flask "Pillow>=10.0"

echo ==^> Setting device mode to gpu
python -c "import json,os;p=os.path.join('face_app','settings.json');d=json.load(open(p)) if os.path.exists(p) else {};d['device']='gpu';json.dump(d,open(p,'w'),indent=2)"

echo.
echo setup complete. Start it with start_app_gpu.bat
pause