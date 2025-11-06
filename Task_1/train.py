import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torchmetrics
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from Model.utils import data_adaptation, collate_fn_1d, calculate_weight_classes
from Model.model import SANDClassifier

LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSI = 5

Path("outputs").mkdir(exist_ok=True)

def plot_confusion_matrix(cm_tensor, epoch):
    cm_numpy = cm_tensor.cpu().numpy()
    cm_norm = np.around(cm_numpy.astype('float') / cm_numpy.sum(axis=1)[:, np.newaxis], decimals=2)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=range(1, NUM_CLASSI + 1),
        yticklabels=range(1, NUM_CLASSI + 1)
    )
    plt.ylabel('True Label (1-5)')
    plt.xlabel('Predicted Label (1-5)')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.savefig(f"outputs/confusion_matrix_epoch_{epoch}.png")
    plt.close(fig)


def train_one_epoch(model, data_loader, loss_fn, optimizer, device, progress: Progress, epoch_task, train_acc_metric):
    model.train()
    total_loss = 0.0
    train_acc_metric.reset()
    
    task = progress.add_task(
        "  [green]Training...", 
        total=len(data_loader), 
        metrics="Loss: 0.0",
        transient=True
    )
    
    for i, (waveforms, labels) in enumerate(data_loader):
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        logits = model(waveforms)
        loss = loss_fn(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        total_loss += current_loss
        
        preds = torch.argmax(logits, dim=1)
        train_acc_metric.update(preds, labels)

        progress.update(task, advance=1, metrics=f"Loss: {current_loss:.4f}")
        
    progress.remove_task(task)
    avg_epoch_acc = train_acc_metric.compute()
    avg_epoch_loss = total_loss / len(data_loader)
    return avg_epoch_loss, avg_epoch_acc

@torch.no_grad()
def validate_epoch(model, data_loader, loss_fn, device, progress_bar: Progress, epoch_task_id, metrics: dict):
    model.eval()
    total_loss = 0.0
    total_confidence = 0.0
    for m in metrics.values(): m.reset()
    
    task = progress_bar.add_task(
        "  [cyan]Validating...", 
        total=len(data_loader), 
        metrics="Loss: 0.0", 
        transient=True
    )
    
    for waveforms, labels in data_loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        logits = model(waveforms)
        loss = loss_fn(logits, labels)
        
        current_loss = loss.item()
        total_loss += current_loss
        
        preds = torch.argmax(logits, dim=1)
        metrics['f1'].update(preds, labels)
        metrics['acc'].update(preds, labels)
        metrics['cm'].update(preds, labels)
        
        probs = torch.softmax(logits, dim=1) 
        confidences = torch.max(probs, dim=1).values
        total_confidence += torch.mean(confidences).item()
        

        progress_bar.update(task, advance=1, metrics=f"Loss: {current_loss:.4f}")
    
    progress_bar.remove_task(task)
    
    avg_loss = total_loss / len(data_loader)
    avg_conf = total_confidence / len(data_loader)
    epoch_f1 = metrics['f1'].compute()
    epoch_acc = metrics['acc'].compute()
    epoch_cm_data = metrics['cm'].compute()
    
    return avg_loss, epoch_acc, epoch_f1, avg_conf, epoch_cm_data


def Train(opt):
    
    Path("outputs").mkdir(exist_ok=True)
    print(f"Using device: {DEVICE}")

    print("--- 1. Preparing Data ---")
    train_dataset, val_dataset = data_adaptation(opt.csv_path, opt.audio_path, opt.target_sample)
    
    print("--- 2. Creating DataLoaders ---")
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, collate_fn=collate_fn_1d
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, collate_fn=collate_fn_1d
    )
    
    print("--- 3. Initializing Model ---")
    model = SANDClassifier(num_classes=NUM_CLASSI).to(DEVICE)
    
    print("--- 4. Initializing Loss and Optimizer ---")
    weight_loss = calculate_weight_classes(opt.csv_path).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weight_loss)
    optimizer = optim.Adam(model.head.parameters(), lr=LEARNING_RATE)
    
    print("--- 5. Initializing Metrics ---")
    train_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSI).to(DEVICE)
    val_metrics = {
        "acc": torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSI).to(DEVICE),
        "f1": torchmetrics.F1Score(task="multiclass", num_classes=NUM_CLASSI, average="macro").to(DEVICE),
        "cm": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=NUM_CLASSI).to(DEVICE)
    }
    print("Metrics ready (F1-Score userà 'average=macro').")
    
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("[blue]{task.fields[metrics]}", justify="right"),
    )
    
    print("\n--- 7. Starting Training ---")
    with progress:
        epoch_task = progress.add_task(
            "[magenta]Epoch",
            total=opt.epochs,
            metrics=""
        )
        
        for epoch in range(1, opt.epochs + 1):
            
            avg_train_loss, avg_train_acc = train_one_epoch(
                model, train_loader, loss_fn, optimizer, DEVICE, progress, epoch_task, train_acc_metric
            )
            
            avg_val_loss, epoch_acc, epoch_f1, avg_conf, epoch_cm = validate_epoch(
                model, val_loader, loss_fn, DEVICE, progress, epoch_task, val_metrics
            )
            
            plot_confusion_matrix(epoch_cm, epoch)
            
            metrics_str = (
                f"Tr.L: {avg_train_loss:.4f}, Tr.A: {avg_train_acc:.4f} | "
                f"Val.L: {avg_val_loss:.4f}, Val.Acc: {epoch_acc:.4f}, Val.F1: {epoch_f1:.4f}, Val.Conf: {avg_conf:.4f}"
            )
            
            progress.update(
                epoch_task, 
                advance=1, 
                metrics=metrics_str
            )
            
    print(f"\n--- Training Completed ---")
    print(f"Confusion matrix plots saved to 'outputs/' directory.")
    torch.save(model.state_dict(), "SAND_Task1.pth")
    print("Final model saved to 'SAND_Task1.pth'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str,help='CSV/Excel Path location',required=True)
    parser.add_argument('--audio-path', type=str,help='Audio Path location',required=True)
    parser.add_argument('--target-sample', type=int, help='Audio Sample Rate', default=16000) 
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
    parser.add_argument('--batch-size', type=int, default=16,help='Batch size')
    parser.add_argument ('--num-workers', type=int, default=4, help='Number of workers for the dataset')
    opt = parser.parse_args()
    
    Train(opt)