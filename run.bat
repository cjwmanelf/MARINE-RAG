@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo.
echo ============================================================
echo   Marine Manual Search - starting...
echo   First run downloads AI models (may take a few minutes).
echo   Open http://127.0.0.1:7860 when the app is ready.
echo ============================================================
echo.
set "PY=C:\Program Files\Python314\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" app.py
pause
