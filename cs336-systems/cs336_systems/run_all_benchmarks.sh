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
MODES=(forward backward both)
MIXED_FLAGS=("" "--mixed"
WARMUPS=(0 1)

for SIZE in "${MODEL_SIZES[@]}"; do
  for MODE in "${MODES[@]}"; do
    for MIX in "${MIXED_FLAGS[@]}"; do
      MIX_TAG="fp32"
      if [[ "$MIX" == "--mixed" ]]; then
        MIX_TAG="fp16"
      fi

      for W in "${WARMUPS[@]}"; do
        LOGFILE="logs/bench_${SIZE}_${MODE}_${MIX_TAG}_warmup${W}.log"
        echo "=== Model=$SIZE Mode=$MODE Precision=$MIX_TAG Warmup=$W ===" | tee "$LOGFILE"

        CMD=(python benchmarking.py
             --model-size "$SIZE"
             --vocab-size "$VOCAB_SIZE"
             --seq-len    "$SEQ_LEN"
             --batch-size "$BATCH"
             --warmup     "$W"
             --steps      "$STEPS"
             --mode       "$MODE"
             --device     "$DEVICE"
        )
        [[ -n "$MIX" ]] && CMD+=("$MIX")

        # profiler only on XL
        if [[ "$SIZE" == "xl" ]]; then
          CMD+=(--profile)
        fi

        echo "${CMD[@]}" | tee -a "$LOGFILE"
        "${CMD[@]}" 2>&1 | tee -a "$LOGFILE"
        echo "" | tee -a "$LOGFILE"
      done

    done
  done
done

echo "All benchmarks complete. Logs are in logs/"
