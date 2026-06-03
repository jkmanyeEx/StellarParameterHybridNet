#!/bin/bash
set -e

# Resolve scripts directory location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🚀 [MaStar Retrain] Step 1: Run hybrid network training..."
python scripts/mastar/train.py

echo "🚀 [MaStar Retrain] Step 2: Evaluate model on validation split..."
python scripts/mastar/evaluate.py

echo "🚀 [MaStar Retrain] Step 3: Run XAI attribution analysis..."
python scripts/mastar/xai_analysis.py

echo "🚀 [MaStar Retrain] Step 4: Bulk evaluation on validation spectra..."
python scripts/mastar/evaluate_mastar.py

