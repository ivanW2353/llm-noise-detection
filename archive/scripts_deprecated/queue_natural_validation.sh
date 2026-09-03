#!/bin/bash
# Queue natural validation after both cleaning experiments finish

echo "=== Natural Signal Validation Queue ==="
echo "Will run after clean_scored + clean_random complete"
echo ""

# Wait for monitor script to finish (it runs both trainings sequentially)
echo "[$(date +%H:%M:%S)] Waiting for monitor session to exit..."
while tmux has-session -t monitor 2>/dev/null; do
    sleep 120
    # Show latest status from either active log
    if tmux has-session -t clean_scored 2>/dev/null; then
        echo "  [clean_scored] $(grep -E 'epoch [0-9]+ step' logs/clean_scored.log | tail -1 | awk '{print $3, $4, $5}')"
    elif tmux has-session -t clean_random 2>/dev/null; then
        echo "  [clean_random] $(grep -E 'epoch [0-9]+ step' logs/clean_random.log | tail -1 | awk '{print $3, $4, $5}')"
    fi
done

echo "[$(date +%H:%M:%S)] ✓ Both training runs complete"
echo ""

# Check GPU is free
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
echo ""

# Run natural validation with ratio10_clean model (20k samples, ~30min)
echo "[$(date +%H:%M:%S)] Starting natural_signal_validation.py --n 20000"
tmux new-session -d -s natural_valid "python scripts/natural_signal_validation.py --model clean --n 20000 2>&1 | tee logs/natural_valid.log"

echo "✓ Queued in tmux session 'natural_valid'"
echo "Monitor: tmux attach -t natural_valid"
echo "Output: tail -f logs/natural_valid.log"
