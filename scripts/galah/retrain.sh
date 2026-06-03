#!/bin/bash
set -e

# Resolve scripts directory location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🚀 [GALAH Retrain] Step 1: Run hybrid network training..."
python scripts/galah/train.py

echo "🚀 [GALAH Retrain] Step 2: Evaluate model on validation split..."
python scripts/galah/evaluate.py
