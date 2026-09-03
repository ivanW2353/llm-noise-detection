#!/bin/bash
# Monitor the cleaning experiment and run sequentially

echo "=== Cleaning Gain Experiment Monitor ==="
echo "Task 1: garbled_scored (remove top-10% by detection score)"
echo "Task 2: garbled_random (remove random 10%)"
echo ""

# Wait for first training to complete
echo "[$(date +%H:%M:%S)] Waiting for clean_scored to complete..."
while tmux has-session -t clean_scored 2>/dev/null; do
    sleep 60
    if [ -f logs/clean_scored.log ]; then
        tail -3 logs/clean_scored.log | grep -E "epoch|step" | tail -1
    fi
done

echo "[$(date +%H:%M:%S)] ✓ clean_scored finished"
echo ""

# Check if it succeeded
if grep -q "save_pretrained" logs/clean_scored.log; then
    echo "✓ Model saved successfully"
else
    echo "⚠ Training may have failed, check logs/clean_scored.log"
    exit 1
fi

# Start second training
echo "[$(date +%H:%M:%S)] Starting clean_random..."
tmux new-session -d -s clean_random "python scripts/train.py --tag ratio10_clean --dataset garbled_random 2>&1 | tee logs/clean_random.log"

# Monitor second training
while tmux has-session -t clean_random 2>/dev/null; do
    sleep 60
    if [ -f logs/clean_random.log ]; then
        tail -3 logs/clean_random.log | grep -E "epoch|step" | tail -1
    fi
done

echo "[$(date +%H:%M:%S)] ✓ clean_random finished"
echo ""

# Summary
echo "=== Both trainings complete ==="
echo "Next steps:"
echo "  1. Run evaluation: python scripts/evaluate.py --tag ratio10_clean --datasets garbled_scored,garbled_random"
echo "  2. Compare against baseline: python scripts/compare_cleaning_gains.py"
