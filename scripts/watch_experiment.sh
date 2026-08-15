#!/bin/bash
# Watch an experiment log and chain post-processing steps when it finishes.
#
#   bash scripts/watch_experiment.sh --log <path> --tag <tag> [--interval 300]
#
# On "ALL DONE": runs the dose-response comparison (compare_ratios.py),
# commits results and pushes to GitHub (with retry on transient TLS errors).
set -u
LOG=""
TAG=""
INTERVAL=300

while [ $# -gt 0 ]; do
    case "$1" in
        --log) LOG="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        *) echo "unknown: $1"; exit 1 ;;
    esac
done
[ -n "$LOG" ] && [ -n "$TAG" ] || { echo "usage: watch_experiment.sh --log <path> --tag <tag>"; exit 1; }

echo "[watch] monitoring $LOG for tag $TAG (every ${INTERVAL}s)"
while true; do
    if grep -q "ALL DONE" "$LOG" 2>/dev/null; then
        echo "[watch] $(date '+%F %T') pipeline DONE -> post-steps"
        cd /root/noisedetect || exit 1
        python3 scripts/compare_ratios.py --tags ratio10,"$TAG"
        git add -A
        git commit -q -m "Auto post-pipeline: dose-response comparison ratio10 vs $TAG" 2>/dev/null || true
        for i in 1 2 3 4 5; do
            git push origin master 2>/dev/null && { echo "[watch] pushed"; break; }
            echo "[watch] push failed (try $i), retrying in 60s"
            sleep 60
        done
        echo "[watch] $(date '+%F %T') all post-steps finished"
        exit 0
    fi
    STAGE=$(grep -E "train |EVAL |ALL DONE" "$LOG" 2>/dev/null | tail -1 | tr -s ' ')
    echo "[watch] $(date '+%F %T') | $STAGE"
    sleep "$INTERVAL"
done
