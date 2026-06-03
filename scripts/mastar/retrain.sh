#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "[MaStar Retrain] Step 1: Training hybrid network..."
python scripts/mastar/train.py

echo "[MaStar Retrain] Step 2: Evaluating on validation split..."
python scripts/mastar/evaluate.py

echo "[MaStar Retrain] Step 3: Running XAI Jacobian analysis..."
python scripts/mastar/xai_analysis.py

echo "[MaStar Retrain] Step 4: Cross-domain evaluation on SDSS DR17 spectra..."
python scripts/mastar/evaluate_mastar.py
