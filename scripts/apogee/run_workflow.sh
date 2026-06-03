#!/bin/bash
set -e

# Resolve scripts directory location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🚀 [APOGEE Workflow] Step 1: Preprocess fluxes..."
python src/data/apogee/preprocess_flux.py

echo "🚀 [APOGEE Workflow] Step 2: Extract physical features..."
python src/data/apogee/extract_features.py

echo "🚀 [APOGEE Workflow] Step 3: Match/Generate catalog labels..."
python src/data/apogee/extract_labels.py

echo "🚀 [APOGEE Workflow] Step 4: Run hybrid network training..."
python scripts/apogee/train.py

echo "🚀 [APOGEE Workflow] Step 5: Evaluate model on validation split..."
python scripts/apogee/evaluate.py
