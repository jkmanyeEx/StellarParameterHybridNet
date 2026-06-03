#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "[MaStar Workflow] Step 1: Preprocessing spectral flux..."
python src/data/mastar/preprocess_flux.py

echo "[MaStar Workflow] Step 2: Extracting physical absorption line features..."
python src/data/mastar/extract_features.py

echo "[MaStar Workflow] Step 3: Aligning catalog labels..."
python src/data/mastar/extract_labels.py

echo "[MaStar Workflow] Step 4: Training hybrid network..."
python scripts/mastar/train.py

echo "[MaStar Workflow] Step 5: Evaluating on validation split..."
python scripts/mastar/evaluate.py

echo "[MaStar Workflow] Step 6: Running XAI Jacobian analysis..."
python scripts/mastar/xai_analysis.py

echo "[MaStar Workflow] Step 7: Cross-domain evaluation on SDSS DR17 spectra..."
python scripts/mastar/evaluate_mastar.py
