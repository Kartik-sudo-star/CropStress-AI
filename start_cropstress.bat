@echo off
REM CropStress AI - one double-click launcher (Windows)
cd /d "%~dp0"
echo ============================================
echo  CropStress AI - Working Model Launcher
echo ============================================
echo [1/3] Checking trained models...
if not exist "models\multimodal\best_model.pkl" (
  echo Models missing - training now (2-3 min)...
  python scripts\bootstrap_working_model.py
  if errorlevel 1 ( echo TRAINING FAILED & pause & exit /b 1 )
) else ( echo Models found. )
echo [2/3] Starting backend on http://127.0.0.1:8000 ...
echo [3/3] Opening UI in browser...
start "" "http://127.0.0.1:8000/"
python -m uvicorn crop_backend:app --host 127.0.0.1 --port 8000
pause
