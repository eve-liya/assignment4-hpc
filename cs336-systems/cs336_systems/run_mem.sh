#!/usr/bin/env bash
set -euo pipefail

# create logs directory
mkdir -p logs

# hyperparameters
DEVICE=cuda
VOCAB_SIZE=1000
SEQ_LEN=64
STEPS=5
BATCH=8

# configurations to sweep
MODEL_SIZES=(small medium large)
MODES=(forward train)

for SIZE in "${MODEL_SIZES[@]}"; do
    for MODE in "${MODES[@]}"; do
        LOGFILE="logs/membench_${SIZE}_${MODE}.log"
  
        echo "=== Model=$SIZE MODE=$MODE ===" | tee "$LOGFILE"

        CMD=(python memory_profile.py
            --model-size "$SIZE"
            --mode "$MODE"
        )

        echo "${CMD[@]}" | tee -a "$LOGFILE"
        "${CMD[@]}" 2>&1 | tee -a "$LOGFILE"
        echo "" | tee -a "$LOGFILE"
done
done

echo "All benchmarks complete. Logs are in logs/"
