@echo off
rem Set up this app on a new PC: create .venv, install packages, and download the trained model from Releases.
rem If the download fails, get best_model.pt from the Releases page and put it in models\ by hand.
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Creating the virtual environment with Python 3.11 ...
  py -3.11 -m venv .venv || goto nopython
)
echo Installing packages. torch is the CPU build. This takes a few minutes ...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu || goto nopackages
if exist models\best_model.pt goto ready
echo Downloading the trained model from Releases ...
curl -fL -o models\best_model.pt https://github.com/laboheme-lover/morse-ai/releases/latest/download/best_model.pt
if errorlevel 1 if exist models\best_model.pt del models\best_model.pt
if not exist models\best_model.pt goto nomodel
:ready
echo Ready. Double-click start_app.bat to open the app.
pause
exit /b 0
:nopython
echo Python 3.11 was not found. Install it from https://www.python.org/ and run this again.
pause
exit /b 1
:nopackages
echo Installing packages failed. See the messages above.
pause
exit /b 1
:nomodel
echo Could not download the model. Get best_model.pt from https://github.com/laboheme-lover/morse-ai/releases and put it in the models folder.
pause
exit /b 1
