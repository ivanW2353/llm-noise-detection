#!/bin/bash
# Run all 6 training runs sequentially. Logs to data_root/train_all.log.
set -e
cd /root/noisedetect
for ds in clean garbled duplicate unrelated keyword mixed; do
    echo "===== $(date '+%F %T') START $ds ====="
    python scripts/train.py --dataset "$ds" --epochs 5
    echo "===== $(date '+%F %T') DONE $ds ====="
done
echo "ALL DONE at $(date '+%F %T')"
