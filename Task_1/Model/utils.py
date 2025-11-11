from pathlib import Path
from sklearn.model_selection import train_test_split
from .dataset import SandDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import re
import torch
import pandas as pd
import numpy as np
import seaborn as sns

def data_adaptation(csv_path: str, audio_root: str, target_sampler: int, test_size: float = 0.25, random_state: int = 42):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        try:
            df = pd.read_excel(csv_path)
        except Exception as e:
            print(f"Errore: Erorr on reading {csv_path} as CSV/Excel.")
            print(f"Details: {e}")
            return None, None
            
    if 'ID' not in df.columns or 'Class' not in df.columns:
        print(f"No 'ID' and 'Class' are in the file.")
        return None, None

    label_map = pd.Series(df.Class.values, index=df.ID).to_dict()

    all_ids = df['ID']
    all_labels = df['Class']
    
    #This is done for having a 75/25 data for test and validation on the dataset.
    train_ids, val_ids = train_test_split(
        all_ids,
        test_size=test_size,       
        random_state=random_state, 
        stratify=all_labels        
    )
    
    train_id_set = set(train_ids)
    val_id_set = set(val_ids)
    
    print(f"Split created: {len(train_id_set)}of Train, {len(val_id_set)} of Validation.")

    audio_root_path = Path(audio_root)
    if not audio_root_path.is_dir():
        print(f"Errore: Audio folder'{audio_root}' does not exist.")
        return None, None
        
    all_audio_files = list(audio_root_path.rglob("*.wav"))
    
    train_file_list = []
    val_file_list = []
    
    id_extractor = re.compile(r"(ID\d+)")

    for file_path in all_audio_files:
        match = id_extractor.search(file_path.name)
        if not match:
            continue
        
        id_soggetto = match.group(1)
        
        if id_soggetto in train_id_set:
            train_file_list.append(file_path)
        elif id_soggetto in val_id_set:
            val_file_list.append(file_path)
            
    print("--- 3. Creating Dataset instance ---")
    train_dataset = SandDataset(
        file_list=train_file_list, 
        label_map=label_map, 
        target_sample_rate=target_sampler,
        is_training=True
    )
    val_dataset = SandDataset(
        file_list=val_file_list, 
        label_map=label_map, 
        target_sample_rate=target_sampler,
        is_training=False # 
    )
    
    return train_dataset, val_dataset

def collate_fn_1d(batch):
  
    batch = [b for b in batch if b[1] != -1]
    if not batch:
        return torch.empty(0), torch.empty(0)

    waveforms, labels = zip(*batch)
    waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels)
    
    return waveforms_padded, labels

def calculate_weight_classes(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_excel(csv_path)
    
    labels = df['Class'].values

    weights = compute_class_weight(
        class_weight='balanced',  
        classes=np.unique(labels),
        y=labels
    )
    
    return torch.tensor(weights, dtype=torch.float32)


def plot_confusion_matrix(cm_tensor, epoch, save_dir: Path, num_classes):
    cm_numpy = cm_tensor.cpu().numpy()
    cm_sum = cm_numpy.sum(axis=1)[:, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.nan_to_num(cm_numpy.astype('float') / cm_sum, nan=0.0)
    cm_norm = np.around(cm_norm, decimals=2)
    
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=range(1, num_classes + 1),
        yticklabels=range(1, num_classes + 1)
    )
    plt.ylabel('True Label (1-5)')
    plt.xlabel('Predicted Label (1-5)')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.savefig(save_dir / f"confusion_matrix_epoch_{epoch}.png") 
    plt.close(fig)


def plot_metrics_history(history: dict, epochs: int, save_dir: Path): # Accetta la cartella
    epochs_range = range(1, epochs + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    ax1.plot(epochs_range, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs_range, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs_range, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs_range, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    ax3.plot(epochs_range, history['val_f1'], 'r-', label='Validation F1-Score (Macro)')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('F1-Score (Macro)')
    ax3.set_title('Validation F1-Score')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / "metrics_history.png") 
    plt.close(fig)
    
def plot_precision_recall_curve(precision, recall, classes, save_dir: Path, num_classes):
    fig = plt.figure(figsize=(12, 8))
    for i in range(num_classes):
        plt.plot(recall[i].cpu(), precision[i].cpu(), label=f'Class {i+1}')
        
    plt.title('Precision-Recall Curve per Class')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "precision_recall_curve.png")
    plt.close(fig)

def prepare_test_data(csv_path: str, audio_root: str):
    print(f"--- 1. Loading test CSV from: {csv_path} ---")
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_excel(csv_path)
        
    if 'ID' not in df.columns or 'Class' not in df.columns:
        raise ValueError("CSV must contain 'ID' and 'Class' columns for validation.")

    label_map = pd.Series(df.Class.values, index=df.ID).to_dict()
    print(f"Label map created for {len(label_map)} patients.")

    print(f"--- 2. Scanning audio files in: {audio_root} ---")
    audio_root_path = Path(audio_root)
    if not audio_root_path.is_dir():
        raise FileNotFoundError(f"Audio root folder '{audio_root}' not found.")
        
    all_audio_files = list(audio_root_path.rglob("*.wav"))
    
    id_extractor = re.compile(r"(ID\d+)")
    test_file_list = []
    test_id_set = set(label_map.keys())
    
    for file_path in all_audio_files:
        match = id_extractor.search(file_path.name)
        if match and match.group(1) in test_id_set:
            test_file_list.append(file_path)
            
    print(f"Found {len(test_file_list)} .wav files corresponding to the test CSV.")
    
    return test_file_list, label_map