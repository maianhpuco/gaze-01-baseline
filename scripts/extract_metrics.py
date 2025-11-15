#!/usr/bin/env python3
"""
Extract per-class metrics (Accuracy, Precision, Recall, AUC) for 3 classes
from evaluation logs and output a summary table.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

def extract_temporal_metrics(log_file):
    """Extract metrics from temporal model evaluation"""
    metrics = defaultdict(dict)
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract per-class metrics
    classes = ['CHF', 'pneumonia', 'Normal']
    for cls in classes:
        # AUC
        auc_match = re.search(rf'{cls} AUC: ([\d.]+)', content)
        if auc_match:
            metrics[cls]['AUC'] = float(auc_match.group(1))
        
        # Precision
        prec_match = re.search(rf'{cls} Precision: ([\d.]+)', content)
        if prec_match:
            metrics[cls]['Precision'] = float(prec_match.group(1))
        
        # Recall
        recall_match = re.search(rf'{cls} Recall: ([\d.]+)', content)
        if recall_match:
            metrics[cls]['Recall'] = float(recall_match.group(1))
    
    # Overall accuracy
    acc_match = re.search(r'multiclass accuracy: ([\d.]+)', content)
    if acc_match:
        metrics['Overall']['Accuracy'] = float(acc_match.group(1))
    
    return metrics

def extract_unet_metrics(log_file):
    """Extract metrics from UNet model evaluation"""
    metrics = defaultdict(dict)
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract per-class metrics
    classes = ['CHF', 'pneumonia', 'Normal']
    for cls in classes:
        # AUC
        auc_match = re.search(rf'{cls} AUC: ([\d.]+)', content)
        if auc_match:
            metrics[cls]['AUC'] = float(auc_match.group(1))
        
        # Precision
        prec_match = re.search(rf'{cls} Precision: ([\d.]+)', content)
        if prec_match:
            metrics[cls]['Precision'] = float(prec_match.group(1))
        
        # Recall
        recall_match = re.search(rf'{cls} Recall: ([\d.]+)', content)
        if recall_match:
            metrics[cls]['Recall'] = float(recall_match.group(1))
    
    # Overall accuracy
    acc_match = re.search(r'multiclass accuracy: ([\d.]+)', content)
    if acc_match:
        metrics['Overall']['Accuracy'] = float(acc_match.group(1))
    
    return metrics

def extract_gradia_metrics(log_file):
    """Extract metrics from GRADIA model evaluation"""
    metrics = defaultdict(dict)
    with open(log_file, 'r') as f:
        content = f.read()
    
    # GRADIA outputs macro metrics, we'll use those for all classes
    # Overall accuracy
    acc_match = re.search(r'acc=([\d.]+)%', content)
    if acc_match:
        metrics['Overall']['Accuracy'] = float(acc_match.group(1)) / 100.0
    
    # Macro AUC
    auc_match = re.search(r'AUC=([\d.]+)', content)
    if auc_match:
        auc_val = float(auc_match.group(1))
        for cls in ['CHF', 'pneumonia', 'Normal']:
            metrics[cls]['AUC'] = auc_val  # Use macro AUC for all classes
    
    # Macro Precision
    prec_match = re.search(r'precision=([\d.]+)', content)
    if prec_match:
        prec_val = float(prec_match.group(1))
        for cls in ['CHF', 'pneumonia', 'Normal']:
            metrics[cls]['Precision'] = prec_val
    
    # Macro Recall
    recall_match = re.search(r'recall=([\d.]+)', content)
    if recall_match:
        recall_val = float(recall_match.group(1))
        for cls in ['CHF', 'pneumonia', 'Normal']:
            metrics[cls]['Recall'] = recall_val
    
    return metrics

def print_summary_table(all_metrics):
    """Print a formatted summary table"""
    print("\n" + "="*100)
    print("PER-CLASS METRICS SUMMARY - All Models")
    print("="*100)
    
    models = ['TEMPORAL', 'UNET', 'GRADIA']
    classes = ['CHF', 'pneumonia', 'Normal']
    metric_types = ['Accuracy', 'Precision', 'Recall', 'AUC']
    
    # Print header
    print(f"\n{'Model':<12} {'Class':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'AUC':<12}")
    print("-" * 100)
    
    for model in models:
        if model not in all_metrics:
            continue
        
        model_metrics = all_metrics[model]
        overall_acc = model_metrics.get('Overall', {}).get('Accuracy', 0.0)
        
        for i, cls in enumerate(classes):
            cls_metrics = model_metrics.get(cls, {})
            acc = cls_metrics.get('Accuracy', overall_acc if i == 0 else '')
            prec = cls_metrics.get('Precision', 'N/A')
            recall = cls_metrics.get('Recall', 'N/A')
            auc = cls_metrics.get('AUC', 'N/A')
            
            # Format values
            acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
            prec_str = f"{prec:.4f}" if isinstance(prec, float) else str(prec)
            recall_str = f"{recall:.4f}" if isinstance(recall, float) else str(recall)
            auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
            
            model_name = model if i == 0 else ""
            print(f"{model_name:<12} {cls:<12} {acc_str:<12} {prec_str:<12} {recall_str:<12} {auc_str:<12}")
        
        print("-" * 100)
    
    print("\n" + "="*100)

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_metrics.py <log_file>")
        sys.exit(1)
    
    log_file = Path(sys.argv[1])
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    all_metrics = {}
    
    # Extract metrics for each model
    content = log_file.read_text()
    
    # Check which models are in the log
    if 'temporal' in content.lower() or 'TEMPORAL' in content:
        all_metrics['TEMPORAL'] = extract_temporal_metrics(log_file)
    
    if 'unet' in content.lower() or 'UNET' in content:
        all_metrics['UNET'] = extract_unet_metrics(log_file)
    
    if 'gradia' in content.lower() or 'GRADIA' in content:
        all_metrics['GRADIA'] = extract_gradia_metrics(log_file)
    
    # Print summary table
    print_summary_table(all_metrics)
    
    # Also save to file
    output_file = log_file.parent / f"{log_file.stem}_metrics_summary.txt"
    with open(output_file, 'w') as f:
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_summary_table(all_metrics)
        f.write(buf.getvalue())
    
    print(f"\nSummary saved to: {output_file}")

if __name__ == "__main__":
    main()

