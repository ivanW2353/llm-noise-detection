#!/bin/bash
# Full finalization chain: wait for trainings -> evaluate both models ->
# compare cleaning gains -> natural validation -> shutdown.
#
# Runs unattended. All output to logs/finalize.log
set -u
cd /root/noisedetect

LOG=logs/finalize.log
exec > >(tee -a "$LOG") 2>&1

echo "================================================================"
echo "[$(date '+%F %T')] FINALIZATION CHAIN STARTED"
echo "================================================================"

# ---------------------------------------------------------------- Step 1
# Wait for both training runs (monitor session drives them sequentially)
echo ""
echo "[$(date '+%F %T')] [1/5] Waiting for training runs to finish..."
while tmux has-session -t monitor 2>/dev/null \
   || tmux has-session -t clean_scored 2>/dev/null \
   || tmux has-session -t clean_random 2>/dev/null; do
    sleep 120
done
echo "[$(date '+%F %T')] ✓ All training sessions exited"

# Verify both LoRA adapters exist before spending GPU hours on eval
MISSING=0
for ds in garbled_scored garbled_random; do
    if [ -f "runs/ratio10_clean/$ds/lora/adapter_model.safetensors" ]; then
        echo "  ✓ $ds adapter present"
    else
        echo "  ✗ $ds adapter MISSING — training likely failed"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "[$(date '+%F %T')] ⚠ ABORTING: not all models trained."
    echo "Server will NOT be shut down so the failure can be inspected."
    echo "Check logs/clean_scored.log and logs/clean_random.log"
    exit 1
fi

# ---------------------------------------------------------------- Step 2
echo ""
echo "[$(date '+%F %T')] [2/5] Evaluating both cleaned models (7 benchmarks each)..."
for ds in garbled_scored garbled_random; do
    echo "  --- evaluating $ds ---"
    python scripts/evaluate.py --tag ratio10_clean --dataset "$ds"
    if [ $? -ne 0 ]; then
        echo "  ⚠ evaluation failed for $ds — aborting, no shutdown"
        exit 1
    fi
done
echo "[$(date '+%F %T')] ✓ Both evaluations complete"

# ---------------------------------------------------------------- Step 3
echo ""
echo "[$(date '+%F %T')] [3/5] Comparing cleaning gains (targeted vs random)..."
python - <<'PYEOF'
import json, os
import pandas as pd

rows = []
eval_dir = "results/eval"
for ds in ["garbled_scored", "garbled_random"]:
    # evaluate.py writes results/eval/<tag>_<dataset>.json (or similar);
    # search for the matching summary file
    cands = [p for p in os.listdir(eval_dir)
             if ds in p and p.endswith(".json") and "raw" not in p]
    if not cands:
        print(f"  ⚠ no eval summary found for {ds}")
        continue
    path = os.path.join(eval_dir, sorted(cands)[-1])
    res = json.load(open(path))
    row = {"model": ds}
    for task, r in res.items():
        if isinstance(r, dict) and "acc" in r:
            row[task] = r["acc"]
    rows.append(row)

if len(rows) == 2:
    df = pd.DataFrame(rows).set_index("model")
    # Baseline: the uncleaned garbled model from the main experiment
    try:
        base = pd.read_csv("results/ratio10/eval_comparison.csv").set_index("model")
        for ref in ["garbled", "clean"]:
            if ref in base.index:
                df.loc[f"[ref] {ref}"] = base.loc[ref]
    except Exception as e:
        print(f"  (could not load ratio10 reference rows: {e})")

    print()
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    df.to_csv("results/cleaning_gain_comparison.csv")
    print()
    print("  -> results/cleaning_gain_comparison.csv")

    if "garbled_scored" in df.index and "garbled_random" in df.index:
        print()
        print("  Targeted cleaning minus random dropout:")
        for task in df.columns:
            d = df.loc["garbled_scored", task] - df.loc["garbled_random", task]
            print(f"    {task:14s} {d:+.4f}")
        print()
        print("  Positive = targeted (score-based) cleaning beat random dropout.")
else:
    print(f"  ⚠ expected 2 result rows, got {len(rows)} — skipping comparison")
PYEOF

# ---------------------------------------------------------------- Step 4
echo ""
echo "[$(date '+%F %T')] [4/5] Natural-data signal validation (20k samples)..."
python scripts/natural_signal_validation.py --model clean --n 20000
echo "[$(date '+%F %T')] ✓ Natural validation complete"

# ---------------------------------------------------------------- Step 5
echo ""
echo "================================================================"
echo "[$(date '+%F %T')] [5/5] ALL TASKS COMPLETE"
echo "================================================================"
echo "Artifacts:"
echo "  results/cleaning_gain_comparison.csv   cleaning gain: targeted vs random"
echo "  results/eval/*garbled_scored*.json     eval of score-cleaned model"
echo "  results/eval/*garbled_random*.json     eval of randomly-cleaned model"
echo "  logs/finalize.log                      this log"
echo ""
echo "Shutting down in 60s. Cancel with: tmux kill-session -t finalize"
sleep 60

echo "[$(date '+%F %T')] Shutting down."
sync
shutdown -h now
