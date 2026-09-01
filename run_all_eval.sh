#!/bin/bash
# Sequential evaluation of all fine-tuned models + base model.
# Run AFTER training finishes (GPU must not be shared). Log to data_root/logs/.
set -e
cd "$(dirname "$0")"
mkdir -p "$(pwd)/logs"
LOG="$(pwd)/logs/eval_all.log"
for ds in clean garbled duplicate unrelated keyword mixed base; do
    echo "===== $(date '+%F %T') EVAL $ds =====" | tee -a "$LOG"
    python scripts/evaluate.py --dataset "$ds" 2>&1 | tee -a "$LOG"
    echo "===== $(date '+%F %T') DONE $ds =====" | tee -a "$LOG"
done
echo "ALL EVAL DONE at $(date '+%F %T')" | tee -a "$LOG"
