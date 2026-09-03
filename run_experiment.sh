#!/usr/bin/env bash
set -euo pipefail
TAG=${1:-ratio10}; DATASETS=${2:-clean,garbled,duplicate,unrelated,keyword,mixed}; SOURCE=${3:?source JSONL required}
python cli.py data --tag "$TAG" --source "$SOURCE" --datasets "$DATASETS"
IFS=, read -ra ITEMS <<< "$DATASETS"
for d in "${ITEMS[@]}"; do python cli.py train --tag "$TAG" --dataset "$d" --model mock; python cli.py evaluate --tag "$TAG" --dataset "$d"; done
