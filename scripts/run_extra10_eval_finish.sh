#!/usr/bin/env bash
# Completes the 3 tasks (bbh, truthfulqa, winogrande) that crashed on every
# extra10 eval (BBH_DIR was missing on the new server; fixed now).
# Resumable: only missing tasks run. Log to data_root/logs/.
set -u
LOG=/root/autodl-tmp/noisedetect/logs/extra10_eval_finish.log
exec >> "$LOG" 2>&1
cd /root/noisedetect
for ds in template truncation near_duplicate mixed; do
    echo "===== $(date '+%F %T') completing $ds ====="
    python3 scripts/evaluate.py --dataset "$ds" --tag extra10 --tasks bbh,truthfulqa,winogrande
done
echo "===== $(date '+%F %T') eval completion DONE ====="
