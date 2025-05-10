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
NormImpls=(compiled)

for SIZE in "${MODEL_SIZES[@]}"; do
    for NORM_IMPL in "${NormImpls[@]}"; do
        LOGFILE="logs/normbench_${SIZE}_${NORM_IMPL}.log"
  
        echo "=== ALL COMPILED Model=$SIZE NormImpl=$NORM_IMPL ===" | tee "$LOGFILE"

        CMD=(python benchmarking.py
             --model-size "$SIZE"
             --vocab-size "$VOCAB_SIZE"
             --seq-len    "$SEQ_LEN"
             --batch-size "$BATCH"
             --steps      "$STEPS"
             --mode       "forward"
             --device     "$DEVICE"
             --norm       "$NORM_IMPL"
             --compile
        )

        echo "${CMD[@]}" | tee -a "$LOGFILE"
        "${CMD[@]}" 2>&1 | tee -a "$LOGFILE"
        echo "" | tee -a "$LOGFILE"
done
done

echo "All benchmarks complete. Logs are in logs/"
