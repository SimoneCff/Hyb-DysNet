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
from datetime import datetime

# Import per Rich
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich import print 

# Import dei tuoi moduli
from Model.utils import data_adaptation, collate_fn_1d, calculate_weight_classes
from Model.model import crea_modello 

# --- 1. CONFIGURAZIONE ---
LEARNING_RATE = 1e-5 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSI = 5
CHECKPOINT_DIR_BASE = Path("outputs") 
EARLY_STOPPING_PATIENCE = 20 

# --- 2. FUNZIONI DI PLOT (plot_confusion_matrix è invariata) ---
def plot_confusion_matrix(cm_tensor, epoch, save_dir: Path):
    cm_numpy = cm_tensor.cpu().numpy()
    cm_sum = cm_numpy.sum(axis=1)[:, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.nan_to_num(cm_numpy.astype('float') / cm_sum, nan=0.0)
    cm_norm = np.around(cm_norm, decimals=2)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=range(1, NUM_CLASSI + 1),
        yticklabels=range(1, NUM_CLASSI + 1)
    )
    plt.ylabel('True Label (1-5)')
    plt.xlabel('Predicted Label (1-5)')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    # Il nome del file ora userà l'epoch (es. "BEST")
    plt.savefig(save_dir / f"confusion_matrix_{epoch}.png") 
    plt.close(fig)

def plot_metrics_history(history: dict, save_dir: Path):
    num_epochs_run = len(history['train_loss'])
    if num_epochs_run == 0: return
    epochs_range = range(1, num_epochs_run + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    ax1.plot(epochs_range, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs_range, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True)
    ax2.plot(epochs_range, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs_range, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(True)
    ax3.plot(epochs_range, history['val_f1'], 'r-', label='Validation F1-Score (Macro)')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('F1-Score (Macro)')
    ax3.legend(); ax3.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "metrics_history.png")
    plt.close(fig)

# --- 3. FUNZIONE LOSS E PREDIZIONE PROTOTIPICA (Nessuna modifica) ---
def prototypical_step(embeddings: torch.Tensor, labels: torch.Tensor, device: torch.device):
    prototypes = torch.zeros(NUM_CLASSI, embeddings.shape[1], device=device)
    class_counts = torch.zeros(NUM_CLASSI, device=device)
    
    for c in range(NUM_CLASSI):
        class_embeddings = embeddings[labels == c]
        if class_embeddings.shape[0] > 0:
            prototypes[c] = class_embeddings.mean(dim=0)
            class_counts[c] = class_embeddings.shape[0]
            
    valid_class_indices = torch.where(class_counts > 0)[0]
    
    if len(valid_class_indices) <= 1:
        return None, None, None 
        
    valid_prototypes = prototypes[valid_class_indices]
    
    distances = torch.cdist(embeddings, valid_prototypes, p=2.0) ** 2
    
    label_map = {original.item(): new for new, original in enumerate(valid_class_indices)}
    
    valid_mask = torch.isin(labels, valid_class_indices)
    valid_labels = labels[valid_mask]
    valid_distances = distances[valid_mask]
    
    if valid_labels.shape[0] == 0:
        return None, None, None
        
    mapped_labels = torch.tensor([label_map[l.item()] for l in valid_labels], device=device, dtype=torch.long)
    
    logits = -valid_distances
    
    preds_indices = torch.argmin(distances, dim=1)
    preds_original = valid_class_indices[preds_indices]
    
    return logits, mapped_labels, preds_original

# --- 4. FUNZIONE DI TRAINING (Nessuna modifica) ---
def train_one_epoch(model, data_loader, loss_fn, optimizer, device, progress: Progress, epoch_task, train_acc_metric):
    model.train()
    total_loss = 0.0
    train_acc_metric.reset()
    task = progress.add_task("  [green]Training...", total=len(data_loader), metrics="Loss: 0.0", transient=True)
    
    batches_skipped = 0
    for i, (waveforms, labels) in enumerate(data_loader):
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        embeddings = model(waveforms)
        logits, mapped_labels, preds = prototypical_step(embeddings, labels, device)
        
        if logits is None: 
            progress.update(task, advance=1, metrics="Loss: SKIPPED")
            batches_skipped += 1
            continue
            
        loss = loss_fn(logits, mapped_labels) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        total_loss += current_loss
        
        train_acc_metric.update(preds, labels)
        progress.update(task, advance=1, metrics=f"Loss: {current_loss:.4f}")
        
    progress.remove_task(task)
    
    num_valid_batches = len(data_loader) - batches_skipped
    avg_epoch_acc = train_acc_metric.compute()
    avg_epoch_loss = total_loss / num_valid_batches if num_valid_batches > 0 else 0.0
    return avg_epoch_loss, avg_epoch_acc

# --- 5. FUNZIONE DI VALIDAZIONE (Nessuna modifica) ---
@torch.no_grad()
def validate_epoch(model, data_loader, loss_fn, device, progress_bar: Progress, epoch_task_id, metrics: dict):
    model.eval()
    total_loss = 0.0
    total_confidence = 0.0 
    for m in metrics.values(): m.reset()
    task = progress_bar.add_task("  [cyan]Validating...", total=len(data_loader), metrics="Loss: 0.0", transient=True)
    
    batches_skipped = 0
    for waveforms, labels in data_loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        embeddings = model(waveforms)
        logits, mapped_labels, preds = prototypical_step(embeddings, labels, device)
        
        if logits is None:
            progress_bar.update(task, advance=1, metrics="Loss: SKIPPED")
            batches_skipped += 1
            continue
            
        loss = loss_fn(logits, mapped_labels)
        current_loss = loss.item()
        total_loss += current_loss
        
        metrics['f1'].update(preds, labels)
        metrics['acc'].update(preds, labels)
        metrics['cm'].update(preds, labels)
        
        distances = -logits
        min_distances = torch.min(distances, dim=1).values
        confidences = 1.0 / (min_distances + 1e-6)
        total_confidence += torch.mean(confidences).item()
        
        progress_bar.update(task, advance=1, metrics=f"Loss: {current_loss:.4f}")
    
    progress_bar.remove_task(task)
    
    num_valid_batches = len(data_loader) - batches_skipped
    if num_valid_batches == 0:
        print("[bold red]Attenzione: Nessun batch di validazione valido (tutti saltati).[/bold red]")
        return 0, 0, 0, 0, torch.zeros(NUM_CLASSI, NUM_CLASSI)

    avg_loss = total_loss / num_valid_batches
    avg_conf = total_confidence / num_valid_batches
    epoch_f1 = metrics['f1'].compute()
    epoch_acc = metrics['acc'].compute()
    epoch_cm_data = metrics['cm'].compute()
    
    return avg_loss, epoch_acc, epoch_f1, avg_conf, epoch_cm_data

# --- 6. FUNZIONE DI TRAINING PRINCIPALE (MODIFICATA) ---
def Train(opt):
    
    # --- MODIFICA: Setup cartelle ---
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = Path(f"outputs/train-FSL-{now_str}")
    ckpt_dir = exp_dir / "checkpoints"
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir()
    
    print(f"--- Dati di output salvati in: {exp_dir} ---")
    print(f"--- [bold yellow]MODALITÀ FEW-SHOT LEARNING (PROTOTYPICAL)[/bold yellow] ---")
    print(f"Using device: {DEVICE}")

    print("--- 1. Preparing Data ---")
    train_dataset, val_dataset = data_adaptation(
        opt.csv_path, 
        opt.audio_path, 
        opt.target_sample
    )
    
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
    model = crea_modello(num_classes=NUM_CLASSI, freeze=False).to(DEVICE)
    
    print("--- 4. Initializing Loss and Optimizer ---")
    loss_fn = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("--- 5. Initializing Metrics ---")
    train_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSI).to(DEVICE)
    val_metrics = {
        "acc": torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSI).to(DEVICE),
        "f1": torchmetrics.F1Score(task="multiclass", num_classes=NUM_CLASSI, average="macro").to(DEVICE),
        "cm": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=NUM_CLASSI).to(DEVICE)
    }
    print("Metrics ready (F1-Score userà 'average=macro').")

    # --- 6. LOGICA DI RESUME (Modificata) ---
    start_epoch = 1
    best_val_f1 = -1.0
    epochs_no_improve = 0 
    best_cm_data = None # <-- NUOVO: Salva la migliore CM
    
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
                print("Checkpoint completo trovato.")
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
                start_epoch = checkpoint['epoch'] + 1 
                if 'best_val_f1' in checkpoint:
                     best_val_f1 = checkpoint['best_val_f1']
                if 'history' in checkpoint:
                    history = checkpoint['history']
                if 'epochs_no_improve' in checkpoint:
                     epochs_no_improve = checkpoint['epochs_no_improve']
                if 'best_cm_data' in checkpoint: # <-- NUOVO
                     best_cm_data = checkpoint['best_cm_data']
            else:
                print("Checkpoint 'legacy' (solo pesi) trovato.")
                model.load_state_dict(checkpoint, strict=False)
                start_epoch = 1 
                 
            print(f"Resuming from Epoch {start_epoch}. Best F1 so far: {best_val_f1:.4f}")
        else:
            print(f"Attenzione: Checkpoint '{opt.resume}' non trovato. Inizio da zero.")
    else:
        print("--- Starting training from scratch (Epoch 1) ---")
    
    # --- 7. DEFINIZIONE BARRA DI PROGRESSO (Nessuna modifica) ---
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
            
            # --- RIMOSSO: Plot CM ad ogni epoca ---
            # plot_confusion_matrix(epoch_cm, epoch, cm_dir) 
            
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
                best_cm_data = epoch_cm # <-- NUOVO: Salva i dati della CM migliore
                best_model_path = ckpt_dir / "best.pth"
                torch.save(model.state_dict(), best_model_path)
                metrics_str += " [yellow](Best Model Saved!)"
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                metrics_str += f" [red](Patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE})"
            
            if epoch % 10 == 0 or epoch == opt.epochs:
                checkpoint_path = ckpt_dir / f"epoch_{epoch}.pth"
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_f1': best_val_f1,
                    'history': history,
                    'epochs_no_improve': epochs_no_improve,
                    'best_cm_data': best_cm_data # <-- NUOVO: Salva la CM migliore
                }
                torch.save(checkpoint, checkpoint_path)

            progress.update(epoch_task, advance=1, metrics=metrics_str)
            
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n[bold red]EARLY STOPPING![/bold red] No improvement in Val F1-Score for {EARLY_STOPPING_PATIENCE} epochs.")
                print(f"Best F1-Score was {best_val_f1:.4f}. Stopping at epoch {epoch}.")
                break 
            
    print(f"\n--- Training Completed ---")
    
    total_epochs_run = len(history['train_loss'])
    if total_epochs_run > 0:
        print("Generating metrics history plot...")
        plot_metrics_history(history, exp_dir)
        print(f"Metrics plot saved to '{exp_dir}/metrics_history.png'.")
        
        # --- NUOVO: Plotta la CM migliore alla fine ---
        print("Generating best confusion matrix...")
        if best_cm_data is not None:
            plot_confusion_matrix(best_cm_data, epoch="BEST", save_dir=exp_dir)
            print(f"Best confusion matrix saved to '{exp_dir}/confusion_matrix_BEST.png'.")
        else:
            print("[yellow]No best confusion matrix to save (model did not improve).[/yellow]")
        # --- FINE BLOCCO ---
    
    print(f"Saving final model to 'last.pth'...")
    last_checkpoint_path = ckpt_dir / "last.pth"
    checkpoint = {
        'epoch': epoch, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_f1': best_val_f1,
        'history': history,
        'epochs_no_improve': epochs_no_improve,
        'best_cm_data': best_cm_data # Salva anche nel last
    }
    torch.save(checkpoint, last_checkpoint_path)

    # print(f"Confusion matrix plots saved to '{cm_dir}'.") # <-- RIMOSSO
    print(f"Checkpoints saved to '{ckpt_dir}'.")
    print(f"Best model (F1: {best_val_f1:.4f}) saved to '{ckpt_dir}/best.pth'.")

# --- 8. PARSER ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAND Challenge - Task 1 Training (Few-Shot Prototypical)")
    parser.add_argument('--csv-path', type=str,help='CSV/Excel Path location',required=True)
    parser.add_argument('--audio-path', type=str,help='Audio Path location',required=True)
    parser.add_argument('--target-sample', type=int, help='Audio Sample Rate', default=16000) 
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=150) 
    parser.add_argument('--batch-size', type=int, default=32,help='Batch size (Default: 32 per V100 32GB)')
    parser.add_argument ('--num-workers', type=int, default=8, help='Number of workers (Default: 8, abbinalo a --cpus-per-task)')
    parser.add_argument('--resume',type=str, default=None, help='resume with a checkpoint (path to .pth file)')
    opt = parser.parse_args()
    
    Train(opt)