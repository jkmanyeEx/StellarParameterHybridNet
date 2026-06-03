#!/bin/bash
set -e

# Resolve scripts directory location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🚀 [APOGEE Retrain] Step 1: Run hybrid network training..."
python scripts/apogee/train.py

echo "🚀 [APOGEE Retrain] Step 2: Evaluate model on validation split..."
python scripts/apogee/evaluate.py
