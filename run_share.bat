@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo.
echo ============================================================
echo   Marine Manual Search - SHARE mode
echo   A public link (https://xxxx.gradio.live) will appear below.
echo   Copy it to share. Closing this window ends the link.
echo   Your local screen opens automatically.
echo   [WARNING] Do NOT expose confidential manuals (anyone with
echo   the link can access). Use sample PDFs only.
echo ============================================================
echo.
set "PY=C:\Program Files\Python314\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" app.py --share --open
pause
