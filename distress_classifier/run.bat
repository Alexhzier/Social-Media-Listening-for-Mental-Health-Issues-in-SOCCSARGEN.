@echo off
REM Run the distress classifier. Uses Python from your PATH.
cd /d "%~dp0"

python --version 2>nul
if errorlevel 1 (
    echo Python not found on PATH. Install from https://www.python.org/downloads/
    echo Check "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

echo Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo If you see "No matching distribution found for tensorflow", use Python 3.10/3.11/3.12 64-bit.
    pause
    exit /b 1
)

echo.
echo Training model...
python main.py
if errorlevel 1 (
    pause
    exit /b 1
)

echo.
echo Done. Test with: python predict.py "Your text here." --model_dir output
pause
