@echo off
title CPLC Face Search
setlocal

REM ── Resolve paths ────────────────────────────────────────────────────────────
set "ROOT=%~dp0"
set "FACE_APP=%ROOT%face_app"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"

REM ── Use venv Python directly (no activation needed) ──────────────────────────
if not exist "%VENV_PY%" (
    echo ERROR: venv not found at %VENV_PY%
    echo Run:  python -m venv .venv  then  .venv\Scripts\pip install -r face_app\requirements.txt
    pause
    exit /b 1
)

REM ── Always launch from face_app\ so local imports resolve ────────────────────
cd /d "%FACE_APP%"
"%VENV_PY%" main_app.py %*
