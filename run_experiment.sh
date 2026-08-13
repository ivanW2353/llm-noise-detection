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

RATIO=""
TAG=""
MODE="full"
REUSE_CLEAN=""

while [ $# -gt 0 ]; do
    case "$1" in
        --ratio) RATIO="$2"; shift 2 ;;
        --tag)   TAG="$2"; shift 2 ;;
        --reuse-clean) REUSE_CLEAN="1"; shift ;;
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
    echo "===== [$TAG] building datasets (ratio=$RATIO) ====="
    python3 scripts/make_noise.py --ratio "$RATIO" --tag "$TAG"
    echo "===== [$TAG] training (5 noisy runs + clean) ====="
    for ds in clean garbled duplicate unrelated keyword mixed; do
        if [ "$ds" = "clean" ] && [ -n "$REUSE_CLEAN" ]; then
            # the clean dataset is identical across ratios (same seed/order),
            # so its run (metrics, LoRA, TB) can be reused from the default tag
            SRC="/root/autodl-tmp/noisedetect/runs/clean"
            DST="/root/autodl-tmp/noisedetect/runs/$TAG/clean"
            if [ -d "$SRC" ] && [ ! -d "$DST" ]; then
                mkdir -p "/root/autodl-tmp/noisedetect/runs/$TAG"
                cp -r "$SRC" "$DST"
                echo "----- [$TAG] reuse clean run from default tag -----"
                continue
            fi
        fi
        echo "----- [$TAG] train $ds -----"
        python3 scripts/train.py --dataset "$ds" --tag "$TAG"
    done
fi

if [ "$MODE" = "full" ] || [ "$MODE" = "eval" ]; then
    echo "===== [$TAG] evaluation ====="
    for ds in clean garbled duplicate unrelated keyword mixed base; do
        echo "----- [$TAG] eval $ds -----"
        python3 scripts/evaluate.py --dataset "$ds" --tag "$TAG"
    done
fi

if [ "$MODE" = "full" ] || [ "$MODE" = "analyze" ]; then
    echo "===== [$TAG] detection analysis ====="
    python3 scripts/analyze_detection.py --tag "$TAG"
    echo "===== [$TAG] token-level analysis ====="
    python3 scripts/analyze_token_level.py --tag "$TAG"
fi

echo "===== [$TAG] ALL DONE ====="
