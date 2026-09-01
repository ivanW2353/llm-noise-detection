#!/usr/bin/env bash
# extra10 tail pipeline: waits for mixed training to finish, then runs
# eval -> detection analysis -> token-level analysis -> IFD, sequentially.
# Usage: bash scripts/run_extra10_after.sh   (start in a tmux window)
set -u
DATA_ROOT=/root/autodl-tmp/noisedetect
LOG=$DATA_ROOT/logs/extra10_after.log
RUNS=$DATA_ROOT/runs/extra10
DS=("template" "truncation" "near_duplicate" "mixed")
ok() { echo "===== [$1] $(date '+%F %T') $2 ====="; }

exec >> "$LOG" 2>&1

ok WAIT "waiting for runs/extra10/mixed/summary.json ..."
while [ ! -f "$RUNS/mixed/summary.json" ]; do sleep 30; done
ok WAIT "mixed training done"

ok EVAL "evaluating 4 models (resumable, ~90 min each)"
cd /root/noisedetect
for ds in "${DS[@]}"; do
    python3 scripts/evaluate.py --dataset "$ds" --tag extra10 || echo "WARN: eval $ds failed (rc=$?)"
done

ok DETECTION "analyze_detection --tag extra10"
python3 scripts/analyze_detection.py --tag extra10 --datasets template,truncation,near_duplicate,mixed

ok TOKEN "analyze_token_level --tag extra10"
for ds in "${DS[@]}"; do
    python3 scripts/analyze_token_level.py --dataset "$ds" --tag extra10 || echo "WARN: token-level $ds failed (rc=$?)"
done

ok IFD "compute_ifd (config override tag=extra10)"
cat > "$DATA_ROOT/config_extra10.yaml" <<'EOF'
paths:
  model: /root/autodl-tmp/Qwen2.5-3B-Instruct
  data_root: /root/autodl-tmp/noisedetect
  repo_root: /root/noisedetect
  experiment_tag: extra10
EOF
for ds in "${DS[@]}"; do
    python3 scripts/compute_ifd.py --config "$DATA_ROOT/config_extra10.yaml" --dataset "$ds" || echo "WARN: ifd $ds failed (rc=$?)"
done

ok DONE "extra10 tail pipeline finished"
