#!/bin/bash
# Run all 6 training runs sequentially (default tag). Log to data_root/logs/.
set -e
cd "$(dirname "$0")"
mkdir -p "$(pwd)/logs"
LOG="$(pwd)/logs/train_all.log"
for ds in clean garbled duplicate unrelated keyword mixed; do
    echo "===== $(date '+%F %T') START $ds =====" | tee -a "$LOG"
    python scripts/train.py --dataset "$ds" --epochs 5 2>&1 | tee -a "$LOG"
    echo "===== $(date '+%F %T') DONE $ds =====" | tee -a "$LOG"
done
echo "ALL DONE at $(date '+%F %T')" | tee -a "$LOG"
