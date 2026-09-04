@echo off
set DIR=%~dp0
cd /d "%DIR%Backend"

echo ==================================================
echo     Launching MediKiosk (Ministry of AYUSH)
echo ==================================================

if not exist "venv" (
    echo [Setup] Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate
echo [Setup] Installing required packages...
pip install -r requirements.txt --quiet

echo [MediKiosk] Opening browser at http://localhost:8000 ...
start http://localhost:8000

echo [MediKiosk] Server live!
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
