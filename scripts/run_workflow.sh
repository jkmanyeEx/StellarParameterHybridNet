source .venv/bin/activate

python src/data/preprocess_flux.py
python src/data/extract_features.py
python src/data/extract_labels.py

python scripts/train.py
python scripts/evaluate.py
python scripts/xai_analysis.py
python scripts/evaluate_mastar.py
