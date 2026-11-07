@echo off
rem Start Hive Shot Lab. The first run sets up Python packages in .venv (needs internet, a few minutes).
rem Optional: drag a video file onto this file to open it.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Setting up Hive Shot Lab for the first time...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo Could not find Python. Install Python 3.10 or newer from https://www.python.org/downloads/
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e .
    if errorlevel 1 (
        echo Installing the packages failed - see the messages above.
        pause
        exit /b 1
    )
)
start "" ".venv\Scripts\pythonw.exe" -m hive_shot_lab %*
