@echo off
title Face Match - GPU (exposed on network)
set FACE_DEVICE=gpu
cd /d "%~dp0face_app"
"..\.venv\Scripts\python.exe" app.py --host 0.0.0.0 --port 5000
pause