#!/bin/bash
# One-command pipeline: a single experiment completes in one go.
#
#   bash run_experiment.sh --ratio 0.20 --tag ratio20             # full: build → train → eval → all analyses
#   bash run_experiment.sh --ratio 0.20 --tag ratio20 --reuse-clean
#   bash run_experiment.sh --ratio 0.20 --tag ratio20 --with-extra
#   bash run_experiment.sh --tag ratio20 --train-only|--eval-only|--analyze-only
#
# Everything is auto-skipped once done (per-dataset train summaries, 7-task
# eval completeness, analysis outputs), so re-running the same command resumes
# and completes an interrupted experiment — safe to run any time, including
# while GPU time frees up in stages.
set -e
cd "$(dirname "$0")"

RATIO=""
TAG=""
MODE="full"
REUSE_CLEAN=""
WITH_EXTRA=""

while [ $# -gt 0 ]; do
    case "$1" in
        --ratio) RATIO="$2"; shift 2 ;;
        --tag)   TAG="$2"; shift 2 ;;
        --reuse-clean) REUSE_CLEAN="1"; shift ;;
        --with-extra) WITH_EXTRA="1"; shift ;;
        --train-only) MODE="train"; shift ;;
        --eval-only)  MODE="eval"; shift ;;
        --analyze-only) MODE="analyze"; shift ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$TAG" ]; then
    [ -z "$RATIO" ] && { echo "usage: bash run_experiment.sh --ratio 0.20 [--tag ratio20] [--reuse-clean] [--with-extra] [--train-only|--eval-only|--analyze-only]"; exit 1; }
    TAG=$(python3 -c "print('ratio%d' % round(float('$RATIO')*100))")
fi

DATA="/root/autodl-tmp/noisedetect"
RUNDIR="$DATA/runs/$TAG"
LOG="$DATA/logs/experiment_${TAG}.log"
mkdir -p "$(dirname "$LOG")"

has_summary() { [ -f "$RUNDIR/$1/summary.json" ]; }
eval_complete() {
    local f="/root/noisedetect/results/eval/eval_${TAG}_$1.json"
    [ -f "$f" ] && python3 -c "
import json,sys
r=json.load(open('$f'))
tasks=[k for k,v in r.items() if isinstance(v,dict) and 'acc' in v]
sys.exit(0 if len(tasks)>=7 else 1)" 2>/dev/null
}
trained_ds() {  # trained noise datasets (excludes clean)
    for d in "$RUNDIR"/*/; do
        [ -f "$d/summary.json" ] || continue
        b=$(basename "$d")
        [ "$b" = "clean" ] && continue
        echo "$b"
    done
}
analysis_done() {  # analysis_done <result-prefix> — all trained datasets have the output
    local prefix="$1"
    for ds in $(trained_ds); do
        [ -f "/root/noisedetect/results/${prefix}_${TAG}_${ds}.jsonl" ] || return 1
    done
    return 0
}
stage() {  # stage <name> <cmd...>
    echo "===== [$TAG] $(date '+%F %T') $1 =====" | tee -a "$LOG"
    shift
    "$@" 2>&1 | tee -a "$LOG"
}

echo "===== [$TAG] $(date '+%F %T') run_experiment.sh started (MODE=$MODE) =====" | tee -a "$LOG"

if [ "$MODE" = "full" ] || [ "$MODE" = "train" ]; then
    stage "building datasets (ratio=$RATIO)" python3 scripts/make_noise.py --ratio "$RATIO" --tag "$TAG" \
        $( [ -n "$WITH_EXTRA" ] && echo --with-extra )
    echo "===== [$TAG] $(date '+%F %T') training =====" | tee -a "$LOG"
    TRAIN_LIST="clean garbled duplicate unrelated keyword mixed"
    [ -n "$WITH_EXTRA" ] && TRAIN_LIST="$TRAIN_LIST template truncation near_duplicate"
    for ds in $TRAIN_LIST; do
        if [ "$ds" = "clean" ] && [ -n "$REUSE_CLEAN" ]; then
            # clean dataset is identical across ratios (same seed/order): reuse its run
            SRC="$DATA/runs/ratio10/clean"
            DST="$RUNDIR/clean"
            if [ -d "$SRC" ] && [ ! -d "$DST" ]; then
                mkdir -p "$RUNDIR"
                cp -r "$SRC" "$DST"
                echo "----- [$TAG] $(date '+%F %T') reuse clean run from ratio10 -----" | tee -a "$LOG"
                continue
            fi
        fi
        if has_summary "$ds"; then
            echo "----- [$TAG] $(date '+%F %T') skip train $ds (already trained) -----" | tee -a "$LOG"
            continue
        fi
        echo "----- [$TAG] $(date '+%F %T') train $ds -----" | tee -a "$LOG"
        python3 scripts/train.py --dataset "$ds" --tag "$TAG" 2>&1 | tee -a "$LOG"
        echo "----- [$TAG] $(date '+%F %T') train $ds done -----" | tee -a "$LOG"
    done
    echo "===== [$TAG] $(date '+%F %T') training section done =====" | tee -a "$LOG"
fi

if [ "$MODE" = "full" ] || [ "$MODE" = "eval" ]; then
    echo "===== [$TAG] $(date '+%F %T') evaluation =====" | tee -a "$LOG"
    EVAL_LIST="clean garbled duplicate unrelated keyword mixed base"
    [ -n "$WITH_EXTRA" ] && EVAL_LIST="$EVAL_LIST template truncation near_duplicate"
    if [ -n "$REUSE_CLEAN" ]; then
        # clean model is identical to the default run: reuse its eval results
        if [ -f "results/eval/eval_ratio10_clean.json" ]; then
            cp -n "results/eval/eval_ratio10_clean.json" "results/eval/eval_${TAG}_clean.json"
            cp -n "results/eval/eval_raw_ratio10_clean.jsonl" "results/eval/eval_raw_${TAG}_clean.jsonl" 2>/dev/null || true
            echo "----- [$TAG] $(date '+%F %T') reuse clean eval from ratio10 -----" | tee -a "$LOG"
        fi
        EVAL_LIST="garbled duplicate unrelated keyword mixed"
    fi
    # base model is tag-independent: reuse the default run's eval if present
    if [ -f "results/eval/eval_ratio10_base.json" ] && [ ! -f "results/eval/eval_${TAG}_base.json" ]; then
        cp -n "results/eval/eval_ratio10_base.json" "results/eval/eval_${TAG}_base.json"
        cp -n "results/eval/eval_raw_ratio10_base.jsonl" "results/eval/eval_raw_${TAG}_base.jsonl" 2>/dev/null || true
        echo "----- [$TAG] $(date '+%F %T') reuse base eval from ratio10 -----" | tee -a "$LOG"
    fi
    for ds in $EVAL_LIST; do
        if eval_complete "$ds"; then
            echo "----- [$TAG] $(date '+%F %T') skip eval $ds (already complete) -----" | tee -a "$LOG"
            continue
        fi
        echo "----- [$TAG] $(date '+%F %T') eval $ds -----" | tee -a "$LOG"
        python3 scripts/evaluate.py --dataset "$ds" --tag "$TAG" 2>&1 | tee -a "$LOG"
        echo "----- [$TAG] $(date '+%F %T') eval $ds done -----" | tee -a "$LOG"
    done
    echo "===== [$TAG] $(date '+%F %T') evaluation section done =====" | tee -a "$LOG"
fi

if [ "$MODE" = "full" ] || [ "$MODE" = "analyze" ]; then
    stage "detection analysis" python3 scripts/analyze_detection.py --tag "$TAG"
    if [ -n "$(trained_ds)" ] && analysis_done "token_level/token_level"; then
        echo "----- [$TAG] $(date '+%F %T') skip token-level analysis (all datasets done) -----" | tee -a "$LOG"
    else
        stage "token-level analysis" python3 scripts/analyze_token_level.py --tag "$TAG"
    fi
    if [ -n "$(trained_ds)" ] && analysis_done "ifd"; then
        echo "----- [$TAG] $(date '+%F %T') skip IFD analysis (all datasets done) -----" | tee -a "$LOG"
    else
        stage "IFD analysis" python3 scripts/compute_ifd.py --tag "$TAG"
    fi
fi

echo "===== [$TAG] $(date '+%F %T') ALL DONE =====" | tee -a "$LOG"
