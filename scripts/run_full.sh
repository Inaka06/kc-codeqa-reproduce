#!/bin/bash
# Usage: ./scripts/run_full.sh [laptop|vps] [sanity]
set -e

PROFILE=${1:-vps}
SANITY_FLAG=""
if [ "$2" == "sanity" ]; then
    SANITY_FLAG="--sanity"
fi

echo "Starting full reproduction pipeline with profile: $PROFILE"

# 1. Prepare Data
echo ">>> Step 1: Data Preparation"
python src/prepare_data.py

# 2. Train LoRA
echo ">>> Step 2: Fine-tuning LoRA"
python src/train_lora.py --profile configs/$PROFILE.yaml $SANITY_FLAG

# 3. Evaluate (Greedy)
echo ">>> Step 3: Final Evaluation (Greedy)"
python src/evaluate.py --prefix test_greedy --beams 1

# 4. Evaluate (Beam 4)
echo ">>> Step 4: Final Evaluation (Beam 4)"
python src/evaluate.py --prefix test_beam4 --beams 4

# 5. Generate Report
echo ">>> Step 5: Generating Report"
python src/report.py

echo "Pipeline finished! Check runs/REPORT.md for results."
