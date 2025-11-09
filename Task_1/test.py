import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
from datetime import datetime

import torchmetrics
from torchmetrics.functional.classification import multiclass_precision_recall_curve

from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from Model.model import SANDClassifier
from Model.utils import collate_fn_1d, plot_confusion_matrix, plot_precision_recall_curve, plot_precision_recall_curve, prepare_test_data
from Model.dataset import SandDataset 


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSI = 5


@torch.no_grad()
def evaluate(model, data_loader, device, metrics: dict):
    model.eval()
    total_confidence = 0.0
    
    for m in metrics.values(): m.reset()
    
    all_probs = []
    all_labels = []

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        transient=True
    )
    
    with progress:
        task = progress.add_task("[cyan]Evaluating...", total=len(data_loader))
            
        for waveforms, labels in data_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            logits = model(waveforms)
            
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1) 
            
            metrics['f1'].update(preds, labels)
            metrics['acc'].update(preds, labels)
            metrics['cm'].update(preds, labels)
            
            all_probs.append(probs)
            all_labels.append(labels)
            
            confidences = torch.max(probs, dim=1).values
            total_confidence += torch.mean(confidences).item()
            
            progress.update(task, advance=1)
    
    avg_conf = total_confidence / len(data_loader)
    
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    
    epoch_f1 = metrics['f1'].compute()
    epoch_acc = metrics['acc'].compute()
    epoch_cm_data = metrics['cm'].compute()
    
    precision, recall, _ = multiclass_precision_recall_curve(
        all_probs, all_labels, num_classes=NUM_CLASSI
    )
    
    return epoch_acc, epoch_f1, avg_conf, epoch_cm_data, (precision, recall)

def Test(opt):
    
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(f"outputs/test-{now_str}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Results will be saved to: {output_dir} ---")
    print(f"Using device: {DEVICE}")
    print(f"--- 1. Loading Test Data (CSV: {opt.csv_path}) ---")
    
    try:
        test_file_list, label_map = prepare_test_data(opt.csv_path, opt.audio_path)
    except Exception as e:
        print(f"Error preparing data: {e}")
        return

    test_dataset = SandDataset(file_list=test_file_list, label_map=label_map, target_sample_rate=opt.target_sample)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, collate_fn=collate_fn_1d
    )
    
    print("--- 2. Initializing Model ---")
    model = SANDClassifier(num_classes=NUM_CLASSI).to(DEVICE)
    
    print(f"--- 3. Loading Model Weights from: {opt.model_path} ---")
    model_path = Path(opt.model_path)
    if not model_path.is_file():
        print(f"Error: Model file '{opt.model_path}' not found.")
        return
        
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint) 
            
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    
    print("--- 4. Initializing Metrics ---")
    test_metrics = {
        "acc": torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSI).to(DEVICE),
        "f1": torchmetrics.F1Score(task="multiclass", num_classes=NUM_CLASSI, average="macro").to(DEVICE),
        "cm": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=NUM_CLASSI).to(DEVICE)
    }

    print("\n--- 5. Starting Evaluation ---")
    
    test_acc, test_f1, test_conf, test_cm, (pr_precision, pr_recall) = evaluate(
        model, test_loader, DEVICE, test_metrics
    )
    
    print("\n--- Evaluation Completed ---")
    
    print("\n--- Final Results ---")
    print(f"  Accuracy:         {test_acc.item():.4f}")
    print(f"  Macro F1-Score:   {test_f1.item():.4f}")
    print(f"  Avg. Confidence:  {test_conf:.4f}")
    
    print("\n--- Saving Plots ---")
    plot_confusion_matrix(test_cm, epoch="FINAL", save_dir=output_dir, num_classes=NUM_CLASSI)
    print(f"Confusion Matrix saved to '{output_dir}/confusion_matrix_FINAL.png'")
    
    plot_precision_recall_curve(pr_precision, pr_recall, classes=NUM_CLASSI, save_dir=output_dir,num_classes=NUM_CLASSI)
    print(f"Precision-Recall Curve saved to '{output_dir}/precision_recall_curve.png'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str, help='Path to the TEST CSV/Excel file', required=True)
    parser.add_argument('--audio-path', type=str, help='Path to the ROOT audio folder', required=True)
    parser.add_argument('--model-path', type=str, help='Path to the trained model .pth file (e.g., best.pth)', required=True)
    parser.add_argument('--target-sample', type=int, help='Audio Sample Rate (default: 16000)', default=16000) 
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of data loading workers')
    opt = parser.parse_args()
    
    Test(opt)