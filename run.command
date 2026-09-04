#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/Backend"

echo "=================================================="
echo "    Launching MediKiosk (Ministry of AYUSH)"
echo "=================================================="

if [ ! -d "venv" ]; then
    echo "[Setup] Setting up virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "[Setup] Checking packages..."
pip install -r requirements.txt --quiet

echo "[MediKiosk] Opening http://localhost:8000 in your browser..."
(sleep 1.5 && open http://localhost:8000) &

echo "[MediKiosk] Server live! (Close this window to stop)"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
