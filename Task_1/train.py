import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import torchmetrics
from datetime import datetime 

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich import print
from Model.utils import data_adaptation, collate_fn_1d, calculate_weight_classes, plot_confusion_matrix,  plot_metrics_history
from Model.model import SANDClassifier

LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSI = 5
EARLY_STOPPING_PATIENCE = 20


def train_one_epoch(model, data_loader, loss_fn, optimizer, device, progress: Progress, epoch_task, train_acc_metric):
    model.train()
    total_loss = 0.0
    train_acc_metric.reset()
    task = progress.add_task("  [green]Training...", total=len(data_loader), metrics="Loss: 0.0", transient=True)
    
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
    task = progress_bar.add_task("  [cyan]Validating...", total=len(data_loader), metrics="Loss: 0.0", transient=True)
    
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
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = Path(f"outputs/train-{now_str}")
    cm_dir = exp_dir / "confusion_matrices"
    ckpt_dir = exp_dir / "checkpoints"
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir()
    ckpt_dir.mkdir()
    
    print(f"--- Dati di output salvati in: {exp_dir} ---")
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

    start_epoch = 1
    best_val_f1 = -1.0
    epochs_no_improve = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    if opt.resume:
        resume_path = Path(opt.resume)
        if resume_path.is_file():
            print(f"--- Resuming training from checkpoint: {resume_path} ---")
            checkpoint = torch.load(resume_path, map_location=DEVICE)
            
            if 'model_state_dict' in checkpoint:
                print("Checkpoint completo trovato. Carico modello, ottimizzatore e cronologia.")
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1 
                if 'best_val_f1' in checkpoint:
                     best_val_f1 = checkpoint['best_val_f1']
                if 'history' in checkpoint:
                    history = checkpoint['history']
                if 'epochs_no_improve' in checkpoint: # <-- NUOVO
                     epochs_no_improve = checkpoint['epochs_no_improve']
            else:
                print("Checkpoint 'legacy' (solo pesi) trovato. Carico solo il modello.")
                model.load_state_dict(checkpoint)
                start_epoch = 1 
                 
            print(f"Resuming from Epoch {start_epoch}. Best F1 so far: {best_val_f1:.4f}")
        else:
            print(f"Attenzione: Checkpoint '{opt.resume}' non trovato. Inizio da zero.")
    else:
        print("--- Starting training from scratch (Epoch 1) ---")
    
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
        TextColumn("[blue]{task.fields[metrics]}", justify="right"),
    )
    
    print("\n--- 7. Starting Training ---")
    with progress:
        epoch_task = progress.add_task(
            "[magenta]Epoch",
            total=opt.epochs,
            metrics="",
            completed=start_epoch - 1 
        )
        
        for epoch in range(start_epoch, opt.epochs + 1):
            
            avg_train_loss, avg_train_acc = train_one_epoch(
                model, train_loader, loss_fn, optimizer, DEVICE, progress, epoch_task, train_acc_metric
            )
            
            avg_val_loss, epoch_acc, epoch_f1, avg_conf, epoch_cm = validate_epoch(
                model, val_loader, loss_fn, DEVICE, progress, epoch_task, val_metrics
            )
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(avg_train_acc.item()) 
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(epoch_acc.item())
            history['val_f1'].append(epoch_f1.item())

            metrics_str = (
                f"Tr.L: {avg_train_loss:.4f}, Tr.A: {avg_train_acc:.4f} | "
                f"Val.L: {avg_val_loss:.4f}, Val.Acc: {epoch_acc:.4f}, Val.F1: {epoch_f1:.4f}, Val.Conf: {avg_conf:.4f}"
            )
            
            
            is_best = epoch_f1 > best_val_f1
            if is_best:
                best_val_f1 = epoch_f1
                best_model_path = ckpt_dir / "best.pth"
                torch.save(model.state_dict(), best_model_path)
                metrics_str += " [yellow](Best Model Saved!)"
                epochs_no_improve = 0 # <-- NUOVO: Azzera il contatore
            else:
                epochs_no_improve += 1 # <-- NUOVO: Incrementa il contatore
                metrics_str += f" [red](Patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE})"
            
            # Salva checkpoint periodico (ogni 10 epoche)
            if epoch % 10 == 0 or epoch == opt.epochs:
                checkpoint_path = ckpt_dir / f"epoch_{epoch}.pth"
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_f1': best_val_f1,
                    'history': history,
                    'epochs_no_improve': epochs_no_improve
                }
                torch.save(checkpoint, checkpoint_path)

            progress.update(epoch_task, advance=1, metrics=metrics_str)
            
    print(f"\n--- Training Completed ---")
    
    print("Generating metrics history plot...")
    plot_metrics_history(history, opt.epochs, exp_dir)
    
    print(f"Saving final model to 'last.pth'...")
    last_checkpoint_path = ckpt_dir / "last.pth"
    checkpoint = {
        'epoch': opt.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_f1': best_val_f1,
        'history': history
    }
    torch.save(checkpoint, last_checkpoint_path)

    print(f"Plots saved to '{exp_dir}' and '{cm_dir}'.")
    print(f"Checkpoints saved to '{ckpt_dir}'.")
    print(f"Best model (F1: {best_val_f1:.4f}) saved to '{ckpt_dir}/best.pth'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str,help='Path to the TEST CSV/Excel file',required=True)
    parser.add_argument('--audio-path', type=str,help='Path to the ROOT audio folder',required=True)
    parser.add_argument('--target-sample', type=int, help='Audio Sample Rate', default=16000) 
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
    parser.add_argument('--batch-size', type=int, default=32,help='Batch size for training')
    parser.add_argument ('--num-workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--resume',type=str, default=None, help='Resume with a checkpoint (path to .pth file)')
    opt = parser.parse_args()
    
    Train(opt)