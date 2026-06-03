#!/bin/bash
set -e

# Resolve scripts directory location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🚀 [GALAH Workflow] Step 1: Preprocess fluxes..."
python src/data/galah/preprocess_flux.py

echo "🚀 [GALAH Workflow] Step 2: Extract physical features..."
python src/data/galah/extract_features.py

echo "🚀 [GALAH Workflow] Step 3: Match/Generate catalog labels..."
python src/data/galah/extract_labels.py

echo "🚀 [GALAH Workflow] Step 4: Run hybrid network training..."
python scripts/galah/train.py

echo "🚀 [GALAH Workflow] Step 5: Evaluate model on validation split..."
python scripts/galah/evaluate.py
