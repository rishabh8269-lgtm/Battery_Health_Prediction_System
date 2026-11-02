@echo off
setlocal
cd /d "%~dp0"
echo ==========================================
echo Battery Health Prediction System
echo ==========================================
where py >nul 2>nul
if %errorlevel%==0 (
  set PY=py
) else (
  set PY=python
)
echo Using Python launcher: %PY%
%PY% --version
if errorlevel 1 (
  echo Python was not found. Install Python from python.org and enable PATH, then try again.
  pause
  exit /b 1
)
echo Installing project dependencies (first run may take a few minutes)...
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
  echo Package installation failed. Check your internet connection and Python/pip setup.
  pause
  exit /b 1
)
if not exist "models\battery_soh_model.joblib" (
  echo Training model...
  %PY% train_model.py
  if errorlevel 1 (
    echo Model training failed.
    pause
    exit /b 1
  )
)
echo Launching app...
%PY% -m streamlit run app.py
pause
