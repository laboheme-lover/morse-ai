@echo off
rem Set up this app on a new PC: create .venv, install packages, and download the trained model from Releases.
rem The repository is private, so log in with GitHub CLI (gh auth login) first, or put the model in models\ by hand.
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
set GH=gh
where gh >nul 2>nul || set GH="C:\Program Files\GitHub CLI\gh.exe"
echo Downloading the trained model from Releases ...
%GH% release download -R laboheme-lover/morse-ai -p best_model.pt -D models
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
