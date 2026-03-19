@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHON_EXE=D:\anaconda\envs\normal\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    echo Please confirm the Conda environment path is correct.
    pause
    exit /b 1
)

if not exist ".env" (
    echo [ERROR] Missing .env file in project root.
    echo Please copy .env.example to .env and fill in your API settings first.
    pause
    exit /b 1
)

echo [INFO] Project root: %CD%
echo [INFO] Using Python: %PYTHON_EXE%
echo [INFO] Starting Folder RAG Local...

"%PYTHON_EXE%" app.py
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Application exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
