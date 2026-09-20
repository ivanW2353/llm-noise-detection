#!/usr/bin/env bash
set -euo pipefail
python cli.py train --tag ratio10 --dataset cleaning_gain_targeted --train-file data/ratio10/cleaning_gain/unrelated/train_targeted.jsonl --model hf-lora
python cli.py train --tag ratio10 --dataset cleaning_gain_random --train-file data/ratio10/cleaning_gain/unrelated/train_random.jsonl --model hf-lora
