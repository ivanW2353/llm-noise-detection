#!/usr/bin/env bash
# Run one tagged experiment: data -> train -> evaluate -> analysis -> report.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <tag> [noise_types]" >&2
    echo "Example: $0 ratio05 garbled,duplicate,unrelated,keyword" >&2
    exit 2
fi

TAG=$1
NOISE_TYPES=${2:-garbled,duplicate,unrelated,keyword}
IFS=',' read -r -a REQUESTED_TYPES <<< "$NOISE_TYPES"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

has_extra=false
for noise_type in "${REQUESTED_TYPES[@]}"; do
    case "$noise_type" in
        template|truncation|near_duplicate) has_extra=true ;;
    esac
done

build_args=(--tag "$TAG")
if $has_extra; then
    build_args+=(--with-extra)
fi
python scripts/1_data/make_noise.py "${build_args[@]}"

datasets=(clean)
for noise_type in "${REQUESTED_TYPES[@]}"; do
    [[ "$noise_type" == "clean" || "$noise_type" == "mixed" ]] && continue
    datasets+=("$noise_type")
done
datasets+=(mixed)

# Preserve order while removing duplicate dataset names.
declare -A seen=()
for dataset in "${datasets[@]}"; do
    [[ ${seen[$dataset]+yes} ]] && continue
    seen[$dataset]=1
    echo "=== train: $dataset ($TAG) ==="
    python scripts/2_train/train.py --tag "$TAG" --dataset "$dataset"
    echo "=== evaluate: $dataset ($TAG) ==="
    python scripts/2_train/evaluate.py --tag "$TAG" --dataset "$dataset"
done

echo "=== analysis: $TAG ==="
python scripts/3_analysis/analyze_detection.py --tag "$TAG"
python scripts/3_analysis/analyze_unsupervised.py --tag "$TAG"
python scripts/3_analysis/analyze_memorization.py --tag "$TAG"
python scripts/3_analysis/analyze_transfer.py --tags "$TAG"

python scripts/4_reports/generate_report_tables.py > docs/report_tables.md

echo "Experiment complete: $TAG"
echo "Results: results/$TAG/"
echo "Runs: runs/$TAG/"
