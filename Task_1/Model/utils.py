from pathlib import Path
from sklearn.model_selection import train_test_split
from .dataset import SandDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils.class_weight import compute_class_weight

import re
import torch
import pandas as pd
import numpy as np

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
            
    print("--- 3. Creazione Istanze Dataset PyTorch ---")
    train_dataset = SandDataset(file_list=train_file_list, label_map=label_map, target_sample_rate=target_sampler)
    val_dataset = SandDataset(file_list=val_file_list, label_map=label_map, target_sample_rate=target_sampler)
    
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