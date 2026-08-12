#!/bin/bash
# Sequential evaluation of all fine-tuned models + base model.
# Run AFTER training finishes (GPU must not be shared).
set -e
cd /root/noisedetect
for ds in clean garbled duplicate unrelated keyword mixed base; do
    echo "===== $(date '+%F %T') EVAL $ds ====="
    python scripts/evaluate.py --dataset "$ds"
    echo "===== $(date '+%F %T') DONE $ds ====="
done
echo "ALL EVAL DONE at $(date '+%F %T')"
