"""
train.py

Addestra un modello Few-Shot (Prototypical Networks)
utilizzando un backbone ResNet18 su Spettrogrammi Mel.

Questo script usa il 'dataset.py' auto-sufficiente e non
richiede file .pkl esterni.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

# Import FSL
from easyfsl.methods import PrototypicalNetworks
from easyfsl.samplers import TaskSampler 
from easyfsl.modules import resnet18


try:
    from dataset import SandDataset
except ImportError:
    print("Errore: Assicurati che 'dataset.py' (Versione Spettrogrammi) sia nella stessa cartella.")
    exit(1)

# --- 1. Funzione Main ---
def main():
    
    # --- Configurazione ---
    DEVICE = torch.device("mps")
    
    # Percorsi
    XLSX_PATH = "task1/sand_task_1.xlsx"
    AUDIO_DIR = "task1/training"
    
    # Parametri FSL
    N_WAY = 5
    N_SHOT = 5
    N_QUERY = 10
    N_EPOCHS = 20
    N_TASKS_PER_EPOCH = 100
    
    # --- 1. Inizializza i Dataset ---
    print("Inizializzazione Datasets (auto-sufficiente)...")
    
    # La classe ora fa tutto il lavoro di lettura dell'Excel
    train_dataset = SandDataset(
        xlsx_file_path=XLSX_PATH,
        sheet_name="Training Baseline - Task 1",
        audio_dir=AUDIO_DIR,
        is_training=True  # Attiva SpecAugment
    )
    
    val_dataset = SandDataset(
        xlsx_file_path=XLSX_PATH,
        sheet_name="Validation Baseline - Task 1",
        audio_dir=AUDIO_DIR,
        is_training=False # Disattiva Augmentation
    )

    # --- 2. Inizializza i Sampler e Loader FSL ---
    print("Inizializzazione Sampler FSL...")
    train_sampler = TaskSampler(
        train_dataset, 
        n_way=N_WAY, 
        n_shot=N_SHOT, 
        n_query=N_QUERY, 
        n_tasks=N_TASKS_PER_EPOCH
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=8,
        collate_fn=train_sampler.episodic_collate_fn # <-- ADD THIS LINE
    )

    val_sampler = TaskSampler(
        val_dataset,
        n_way=N_WAY,
        n_shot=N_SHOT,
        n_query=N_QUERY,
        n_tasks=N_TASKS_PER_EPOCH
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=8,
        collate_fn=val_sampler.episodic_collate_fn # <-- ADD THIS LINE
    )

    # --- 3. Inizializza Modello e Optimizer ---
    print("Caricamento backbone ResNet18...")
    # Usiamo ResNet18 perché il nostro Dataset ora crea IMMAGINI
    backbone = resnet18()
    model = PrototypicalNetworks(backbone=backbone).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    criterion = nn.CrossEntropyLoss()

    # --- 4. Loop di Training ---
    print("\n--- Inizio Training FSL (Spettrogrammi + ResNet) ---")
    best_val_acc = 0
    
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}") as pbar:
            for batch in pbar:
                # Unpack the 5 items from the batch
                support_images, support_labels, query_images, query_labels, _class_ids = batch

                # Move only the Tensors to the device (MPS)
                support_images = support_images.to(DEVICE)
                support_labels = support_labels.to(DEVICE)
                query_images = query_images.to(DEVICE)
                query_labels = query_labels.to(DEVICE)
                # We ignore _class_ids, which is a list and stays on the CPU
                
                # 1. Calcola i prototipi usando il support set
                model.process_support_set(support_images, support_labels)
                
                # 2. Ottieni i punteggi (logits) per il query set
                query_logits = model(query_images)
                
                # 3. Calcola il loss
                loss = criterion(query_logits, query_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{total_loss / (pbar.n + 1):.4f}")

        # --- 5. Loop di Validazione ---
        model.eval()
        total_val_acc = 0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Validation Epoch {epoch+1}") as pbar_val:
                for batch in pbar_val:
                    # Unpack the 5 items from the batch
                    support_images, support_labels, query_images, query_labels, _class_ids = batch

                    # Move only the Tensors to the device (MPS)
                    support_images = support_images.to(DEVICE)
                    support_labels = support_labels.to(DEVICE)
                    query_images = query_images.to(DEVICE)
                    query_labels = query_labels.to(DEVICE)
                    
                    # 1. Calcola i prototipi usando il support set
                    model.process_support_set(support_images, support_labels)
                    
                    # 2. Ottieni i punteggi (logits) per il query set
                    query_logits = model(query_images)

                    # 3. Calcola l'accuratezza manualmente
                    _, predictions = torch.max(query_logits, 1)
                    accuracy = (predictions == query_labels).float().mean().item()
                    
                    total_val_acc += accuracy
                    pbar_val.set_postfix(acc=f"{total_val_acc / (pbar_val.n + 1):.4f}")
        
        avg_val_acc = total_val_acc / len(val_loader)
        if avg_val_acc > best_val_acc:
            print(f"Nuova best accuracy: {avg_val_acc:.4f} (salvataggio modello)")
            best_val_acc = avg_val_acc
            torch.save(backbone.state_dict(), "fsl_resnet18_backbone_best.pth")

    print(f"\nTraining completato. Migliore accuratezza di validazione: {best_val_acc:.4f}")

# Proteggi l'esecuzione per il multiprocessing
if __name__ == "__main__":
    main()