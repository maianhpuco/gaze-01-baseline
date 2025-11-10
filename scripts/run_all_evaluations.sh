#!/bin/bash

# Master script to evaluate all 3 models on test set
# This will run all evaluations sequentially and collect results

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                  EVALUATING ALL MODELS ON TEST SET                         ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate quynhng

cd /home/qtnguy50/gaze-01-baseline

# Create results directory
mkdir -p runs/test_results

echo "Starting evaluation..."
echo

# ============================================================================
# 1. GRADIA
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                           1. EVALUATING GRADIA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Use only GPU 0 to be considerate of labmates
export CUDA_VISIBLE_DEVICES=0
echo "Using GPU 0 only (considerate mode for labmates)"
echo
python -u main_train_gradia.py \
    --config configs/data_egd-cxr.yaml \
    --attention_weight 0.0 \
    --iou_samples 50 \
    --evaluate \
    --checkpoint runs/checkpoints/gradia_best.pt 2>&1 | tee runs/test_results/gradia_test.log

echo
echo "✓ GRADIA evaluation complete"
echo

# ============================================================================
# 2. UNET
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                            2. EVALUATING UNET"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

python -u evaluate_unet.py 2>&1 | tee runs/test_results/unet_test.log

echo
echo "✓ UNET evaluation complete"
echo

# ============================================================================
# 3. TEMPORAL
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                          3. EVALUATING TEMPORAL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

python -u evaluate_temporal.py 2>&1 | tee runs/test_results/temporal_test.log

echo
echo "✓ TEMPORAL evaluation complete"
echo

# ============================================================================
# SUMMARY
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                          EVALUATION SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Extract results
echo "Extracting test results..."
echo

python3 << 'EOF'
import re
from pathlib import Path

results = {}

# Extract GRADIA results
gradia_log = Path("runs/test_results/gradia_test.log")
if gradia_log.exists():
    with open(gradia_log) as f:
        content = f.read()
    match = re.search(r'\[TEST\]\s+loss=([\d.]+)\s+acc=([\d.]+)\s+macroAUC=([\d.]+)', content)
    if match:
        results['GRADIA'] = {
            'loss': float(match.group(1)),
            'acc': float(match.group(2)),
            'auc': float(match.group(3))
        }

# Extract UNET results
unet_log = Path("runs/test_results/unet_test.log")
if unet_log.exists():
    with open(unet_log) as f:
        content = f.read()
    match = re.search(r'Macro AUC:\s+([\d.]+)', content)
    acc_match = re.search(r'Accuracy:\s+([\d.]+)', content)
    loss_match = re.search(r'Loss:\s+([\d.]+)', content)
    if match and acc_match and loss_match:
        results['UNET'] = {
            'loss': float(loss_match.group(1)),
            'acc': float(acc_match.group(1)),
            'auc': float(match.group(1))
        }

# Extract TEMPORAL results  
temporal_log = Path("runs/test_results/temporal_test.log")
if temporal_log.exists():
    with open(temporal_log) as f:
        content = f.read()
    match = re.search(r'Macro AUC:\s+([\d.]+)', content)
    acc_match = re.search(r'Accuracy:\s+([\d.]+)', content)
    loss_match = re.search(r'Loss:\s+([\d.]+)', content)
    if match and acc_match and loss_match:
        results['TEMPORAL'] = {
            'loss': float(loss_match.group(1)),
            'acc': float(acc_match.group(1)),
            'auc': float(match.group(1))
        }

# Display results
print("╔════════════════════════════════════════════════════════════════════════════╗")
print("║                         TEST SET FINAL RESULTS                             ║")
print("╚════════════════════════════════════════════════════════════════════════════╝")
print()
print("┌────────────┬──────────────┬──────────────┬──────────────┬──────────────────┐")
print("│ Model      │ Test Loss    │ Test Acc     │ Test macroAUC│ Rank             │")
print("├────────────┼──────────────┼──────────────┼──────────────┼──────────────────┤")

# Sort by AUC
sorted_models = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)

for rank, (name, metrics) in enumerate(sorted_models, 1):
    star = " ⭐" if rank == 1 else ("  🥈" if rank == 2 else "  🥉")
    print(f"│ {name:<10} │   {metrics['loss']:8.4f}   │   {metrics['acc']:8.3f}   │    {metrics['auc']:8.3f}  │ #{rank}{star:14}│")

print("└────────────┴──────────────┴──────────────┴──────────────┴──────────────────┘")
print()

# Save to file
with open("runs/test_results/summary.txt", "w") as f:
    f.write("TEST SET RESULTS SUMMARY\n")
    f.write("="*80 + "\n\n")
    for name, metrics in sorted_models:
        f.write(f"{name}:\n")
        f.write(f"  Loss:      {metrics['loss']:.4f}\n")
        f.write(f"  Accuracy:  {metrics['acc']:.3f}\n")
        f.write(f"  MacroAUC:  {metrics['auc']:.3f}\n")
        f.write("\n")

print("✅ Results saved to: runs/test_results/summary.txt")
print()

EOF

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                      ✅ ALL EVALUATIONS COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "Individual logs saved to:"
echo "  - runs/test_results/gradia_test.log"
echo "  - runs/test_results/unet_test.log"
echo "  - runs/test_results/temporal_test.log"
echo
echo "Summary: runs/test_results/summary.txt"
echo

