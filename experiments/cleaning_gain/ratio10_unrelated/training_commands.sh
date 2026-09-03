#!/bin/bash

# Training commands for cleaning gain experiment

echo 'Training targeted cleaned model...'
python scripts/train_sft.py \
    --train-file /root/noisedetect/experiments/cleaning_gain/ratio10_unrelated/train_targeted.jsonl \
    --model /root/autodl-tmp/Qwen2.5-3B-Instruct \
    --output-dir /root/noisedetect/models/cleaning_exp/ratio10_unrelated_targeted \
    --epochs 5 \
    --lr 0.0002 \
    --micro-batch 1 \
    --grad-accum 16 \
    --lora-r 32 \
    --lora-alpha 64

echo 'Training random cleaned model...'
python scripts/train_sft.py \
    --train-file /root/noisedetect/experiments/cleaning_gain/ratio10_unrelated/train_random.jsonl \
    --model /root/autodl-tmp/Qwen2.5-3B-Instruct \
    --output-dir /root/noisedetect/models/cleaning_exp/ratio10_unrelated_random \
    --epochs 5 \
    --lr 0.0002 \
    --micro-batch 1 \
    --grad-accum 16 \
    --lora-r 32 \
    --lora-alpha 64
