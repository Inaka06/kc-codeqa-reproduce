#!/bin/bash
set -e
echo "Running Zero-Shot Baseline evaluation..."
python src/evaluate.py --adapter_path "" --prefix test_zero_shot
python src/report.py
echo "Zero-shot evaluation complete!"
