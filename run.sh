#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/Backend"

echo "=================================================="
echo "    Launching MediKiosk (Ministry of AYUSH)"
echo "=================================================="

# Check and setup Python venv
if [ ! -d "venv" ]; then
    echo "[Setup] Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "[Setup] Checking required packages..."
pip install -r requirements.txt --quiet

echo "[MediKiosk] Opening browser at http://localhost:8000 ..."
sleep 1 && open http://localhost:8000 &

echo "[MediKiosk] Server live! (Press Ctrl+C to stop)"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
