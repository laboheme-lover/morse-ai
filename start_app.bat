@echo off
rem Start the sound Morse app in the browser (port 8502; 8501 is used by another app).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
