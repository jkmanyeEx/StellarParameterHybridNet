#!/bin/bash
set -e

# Resolve scripts directory location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🚀 [MaStar Workflow] Step 1: Preprocess fluxes..."
python src/data/mastar/preprocess_flux.py

echo "🚀 [MaStar Workflow] Step 2: Extract physical features..."
python src/data/mastar/extract_features.py

echo "🚀 [MaStar Workflow] Step 3: Match/Generate catalog labels..."
python src/data/mastar/extract_labels.py

echo "🚀 [MaStar Workflow] Step 4: Run hybrid network training..."
python scripts/mastar/train.py

echo "🚀 [MaStar Workflow] Step 5: Evaluate model on validation split..."
python scripts/mastar/evaluate.py

echo "🚀 [MaStar Workflow] Step 6: Run XAI attribution analysis..."
python scripts/mastar/xai_analysis.py

echo "🚀 [MaStar Workflow] Step 7: Bulk evaluation on validation spectra..."
python scripts/mastar/evaluate_mastar.py

