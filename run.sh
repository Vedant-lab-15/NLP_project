#!/usr/bin/env bash
# run.sh — MedExtract-Lite one-click launcher
set -e

echo "=== MedExtract-Lite Setup ==="

pip install -r requirements.txt --quiet

echo "--- Generating sample data ---"
python data/generate_data.py

echo "--- Running tests ---"
pytest tests/test_ner.py -v

echo "--- Launching Streamlit app ---"
streamlit run app.py
