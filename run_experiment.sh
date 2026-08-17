#!/bin/bash
# One-command interface for running the full pipeline with a custom noise ratio.
#
#   bash run_experiment.sh --ratio 0.20            # build data + train + eval + analyze
#   bash run_experiment.sh --ratio 0.20 --train-only
#   bash run_experiment.sh --ratio 0.20 --eval-only
#   bash run_experiment.sh --ratio 0.20 --analyze-only
#
# Each experiment is isolated under an experiment tag (default "ratio<NN>"),
# so different ratios never overwrite each other:
#   data : <data_root>/data/ratio20/train/<dataset>/
#   runs : <data_root>/runs/ratio20/<dataset>/
#   eval : <repo>/results/eval_ratio20_<dataset>.json
#   ana  : <repo>/results/*_ratio20.*
set -e
cd "$(dirname "$0")"
LOG="/root/autodl-tmp/noisedetect/logs/experiment_${TAG:-default}.log"
mkdir -p "$(dirname "$LOG")"
echo "===== [$TAG] $(date '+%F %T') run_experiment.sh started (MODE=$MODE) =====" | tee -a "$LOG"

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

[ -z "$RATIO" ] && [ "$MODE" = "full" ] && { echo "usage: bash run_experiment.sh --ratio 0.20 [--tag ratio20] [--train-only|--eval-only|--analyze-only]"; exit 1; }

if [ -z "$TAG" ]; then
    PCT=$(python3 -c "print(int(round(float('$RATIO')*100)))")
    TAG="ratio${PCT}"
fi

if [ "$MODE" = "full" ] || [ "$MODE" = "train" ]; then
{
    echo "===== [$TAG] $(date '+%F %T') building datasets (ratio=$RATIO) ====="
    SHORTCUT_ARG=""
    [ -n "$WITH_EXTRA" ] && SHORTCUT_ARG="--with-extra"
    python3 scripts/make_noise.py --ratio "$RATIO" --tag "$TAG" $SHORTCUT_ARG
    echo "===== [$TAG] $(date '+%F %T') training (5 noisy runs + clean) ====="
    TRAIN_LIST="clean garbled duplicate unrelated keyword mixed"
    [ -n "$WITH_EXTRA" ] && TRAIN_LIST="$TRAIN_LIST template truncation near_duplicate"
    for ds in $TRAIN_LIST; do
        if [ "$ds" = "clean" ] && [ -n "$REUSE_CLEAN" ]; then
            # the clean dataset is identical across ratios (same seed/order),
            # so its run (metrics, LoRA, TB) can be reused from the default run
            SRC="/root/autodl-tmp/noisedetect/runs/ratio10/clean"
            DST="/root/autodl-tmp/noisedetect/runs/$TAG/clean"
            if [ -d "$SRC" ] && [ ! -d "$DST" ]; then
                mkdir -p "/root/autodl-tmp/noisedetect/runs/$TAG"
                cp -r "$SRC" "$DST"
                echo "----- [$TAG] $(date '+%F %T') reuse clean run from default tag -----"
                continue
            fi
        fi
        echo "----- [$TAG] $(date '+%F %T') train $ds -----"
        python3 scripts/train.py --dataset "$ds" --tag "$TAG"
        echo "----- [$TAG] $(date '+%F %T') train $ds done -----"
    done
    echo "===== [$TAG] $(date '+%F %T') training section done ====="
} 2>&1 | tee -a "$LOG"
fi

if [ "$MODE" = "full" ] || [ "$MODE" = "eval" ]; then
{
    echo "===== [$TAG] $(date '+%F %T') evaluation ====="
    EVAL_LIST="clean garbled duplicate unrelated keyword mixed base"
    [ -n "$WITH_EXTRA" ] && EVAL_LIST="$EVAL_LIST template truncation near_duplicate"
    if [ -n "$REUSE_CLEAN" ]; then
        # clean model is identical to the default run: reuse its eval results
        if [ -f "results/eval/eval_ratio10_clean.json" ]; then
            cp -n "results/eval/eval_ratio10_clean.json" "results/eval/eval_${TAG}_clean.json"
            cp -n "results/eval/eval_raw_ratio10_clean.jsonl" "results/eval/eval_raw_${TAG}_clean.jsonl" 2>/dev/null || true
            echo "----- [$TAG] $(date '+%F %T') reuse clean eval from ratio10 -----"
        fi
        EVAL_LIST="garbled duplicate unrelated keyword mixed"
    fi
    # base model is tag-independent: reuse the default run's eval if present
    if [ -f "results/eval/eval_ratio10_base.json" ] && [ ! -f "results/eval/eval_${TAG}_base.json" ]; then
        cp -n "results/eval/eval_ratio10_base.json" "results/eval/eval_${TAG}_base.json"
        cp -n "results/eval/eval_raw_ratio10_base.jsonl" "results/eval/eval_raw_${TAG}_base.jsonl" 2>/dev/null || true
        echo "----- [$TAG] $(date '+%F %T') reuse base eval from ratio10 -----"
    fi
    for ds in $EVAL_LIST; do
        echo "----- [$TAG] $(date '+%F %T') eval $ds -----"
        python3 scripts/evaluate.py --dataset "$ds" --tag "$TAG"
        echo "----- [$TAG] $(date '+%F %T') eval $ds done -----"
    done
    echo "===== [$TAG] $(date '+%F %T') evaluation section done ====="
} 2>&1 | tee -a "$LOG"
fi

if [ "$MODE" = "full" ] || [ "$MODE" = "analyze" ]; then
{
    echo "===== [$TAG] $(date '+%F %T') detection analysis ====="
    python3 scripts/analyze_detection.py --tag "$TAG"
    echo "===== [$TAG] $(date '+%F %T') token-level analysis ====="
    python3 scripts/analyze_token_level.py --tag "$TAG"
} 2>&1 | tee -a "$LOG"
fi

echo "===== [$TAG] $(date '+%F %T') ALL DONE ====="
